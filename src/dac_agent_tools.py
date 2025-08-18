from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from langchain_core.utils.function_calling import convert_to_openai_tool

from art import Trajectory
from pydantic import BaseModel, Field

from src.dac_agent import AgentNode
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger
from src.types import Message, UserMessage, AssistantMessage, ToolMessage


logger = create_logger(__name__)


class delegate_sub_task(BaseModel):
    """
    Calls a sub-agent LLM with the given input prompt. Used to delegate sub-tasks to a sub-agent.
    Useful for breaking down complex tasks. Do not delegate the entire task to the sub-agent.
    Use the sub-agent for task decomposition. You may use this tool multiple times.

    Returns:
        str: The output from the sub-agent.
    """

    input: str = Field(description="Input prompt to the sub-agent")


SUB_TASK_TOOL: ChatCompletionToolParam = convert_to_openai_tool(delegate_sub_task, strict=True)  # type: ignore


class AgentToolNode(AgentNode):
    def create_sub_agent(self):
        return AgentToolNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        kwargs.setdefault("parallel_tool_calls", False)
        if not self.stop_criteria.should_stop(self.current_depth):
            tools = kwargs.setdefault("tools", [])
            tools.append(SUB_TASK_TOOL)

        return await super()._call(messages=messages, **kwargs)

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Start a conversation with the agent using the provided prompt.

        Args:
            prompt (Message): The initial message to start the conversation.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.

        Returns:
            Trajectory: The trajectory of the conversation, including messages and choices.
                This trajectory is used to train an `art.TrainableModel` model.
        """
        if prompt["role"] != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt['role']}.")

        self.trajectory.messages_and_choices.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            self.trajectory.messages_and_choices.append(completion.choices[0])

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract tasks from the response
            tasks_inputs = AgentToolNode.parse_tasks(self.trajectory.messages()[-1])
            if not tasks_inputs or self.stop_criteria.should_stop(self.current_depth):
                break

            task_responses: list[AssistantMessage] = []
            for task in tasks_inputs:
                # create a sub-agent and get answer the task
                sub_agent = self.create_sub_agent()
                resp = await sub_agent.answer(task, verbose, **kwargs)
                task_responses.append(resp)

                # update metrics from sub-agent
                self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)

            tool_calls = completion.choices[0].message.tool_calls or []
            assert len(tool_calls) == len(task_responses)

            for tool_call, task_resp in zip(tool_calls, task_responses):
                content = task_resp.get("content")
                assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"
                tool_message = ToolMessage(role="tool", tool_call_id=tool_call.id, content=content)
                self.trajectory.messages_and_choices.append(tool_message)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            self.stop_criteria.update_round(num_tasks=len(tasks_inputs))

        self.trajectory.finish()
        return self.trajectory

    @staticmethod
    def parse_tasks(message: Message) -> list[UserMessage]:
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        tool_calls = message.get("tool_calls", [])
        tasks = []

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]

            if fn_name == delegate_sub_task.__name__:
                content = delegate_sub_task.model_validate_json(fn_args).input.strip()
                task = UserMessage(role="user", content=content)
                tasks.append(task)

            else:
                logger.error(f"Unexpected tool call name: {tc['function']['name']}")
                raise ValueError("Tool call must be 'call_sub_agent' to extract tasks.")

        return tasks

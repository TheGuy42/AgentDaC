from openai.types.chat.chat_completion import ChatCompletion
from langchain_core.utils.function_calling import convert_to_openai_tool

from art import Trajectory
from pydantic import BaseModel, Field

from src.dac_agent import AgentNode
from src.configs.prompts import get_prompt
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger
from src.openai_types import Message, UserMessage, AssistantMessage, ToolMessage


logger = create_logger(__name__)


class call_sub_agent(BaseModel):
    """
    Creates a new sub-agent LLM and calls it with the provided prompt.

    Returns:
        str: The output from the sub-agent.
    """

    prompt: str = Field(description="Input prompt to a sub-agent. All characters must be properly escaped.")

    def to_task(self) -> UserMessage:
        """
        Converts the tool call to a UserMessage for the sub-agent.
        """
        return UserMessage(role="user", content=self.prompt.strip())


# TODO: test
schema_text_base64 = """
{
    "type": "function",
    "function": {
        "name": "call_sub_agent",
        "description": "Creates a new sub-agent LLM and calls it with the provided prompt.\nReturns the output from the sub-agent.",
        "parameters": {
            "properties": {
                "prompt": {
                    "description": "Base64 (UTF-8) input prompt to a sub-agent.",
                    "type": "string",
                    "contentEncoding": "base64"
                }
            },
            "required": [
                "prompt"
            ],
            "type": "object"
        }
    }
}
"""


class AgentToolNode(AgentNode):
    TOOLS = [
        convert_to_openai_tool(call_sub_agent, strict=False),
    ]

    def create_sub_agent(self):
        return AgentToolNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
        )

    # TODO: need to debug and understand what are the inputs that the model receives.
    # TODO: As far as i can tell, even if the passed tools are empty the model can still issue tool calls.
    # TODO: "strict" is not supported yet resulting sometimes in not properly formatted tool calls.
    # Note that actually thats fine, because strict mode allows only specific strings as inputs, see:
    # https://platform.openai.com/docs/guides/structured-outputs?context=with_parse&type-restrictions=string-restrictions#supported-schemas
    # TODO: Another very big problem is that the model doesnt properly escape the content argument of the tool call,
    # resulting in parsing errors on the server side, which fails quietly and actually returns the tool call in the
    # content field, which ends the conversation. Perhaps to resolve it we need to pass
    # --guided-decoding-backend outlines:no-fallback to engine args or something like that
    # TODO: Maybe the base64 schema will work, idk
    # TODO: perhaps custom tools will work, see https://platform.openai.com/docs/guides/function-calling#custom-tools
    # it seems promising. Its not supported in client.chat.completions.create() unfortunately.
    # and the new API client.responses.create() doesnt return Choice objects
    # TODO: another possible approach is to disable hermes tool parser and parse it ourselves. 
    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        # By default allow only a single tool call in the response
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs.setdefault("stop", [Markers.TOOL_CALL_END, Markers.ANSWER_END, Markers.TASK_END])
        kwargs.setdefault("parallel_tool_calls", False)  # NOTE: currently vLLM ignores this flag

        if not self.decomp_config.should_stop(self.current_depth):
            kwargs["tool_choices"] = "auto"
            tools = kwargs.setdefault("tools", [])
            tools.extend(self.TOOLS)
        else:
            kwargs["tool_choices"] = "none"

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

        should_break = False

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

            if should_break or len(tasks_inputs) == 0:
                break  # No tasks to delegate, so last message

            task_responses: list[AssistantMessage] = []

            if self.decomp_config.should_stop(self.current_depth):
                mock_answer = get_prompt(self.prompt_config.tasks_depleted)
                if mock_answer is None:
                    break

                should_break = True
                for task in tasks_inputs:
                    # Provide mock answer indicating no more tasks available
                    task_responses.append(AssistantMessage(role="assistant", content=mock_answer))

            else:
                for task in tasks_inputs:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    resp = await sub_agent.answer(task, verbose, **kwargs)
                    task_responses.append(resp)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)

            tool_calls = completion.choices[0].message.tool_calls or []
            assert len(tool_calls) == len(task_responses)

            # Create tool responses for each task
            for tool_call, task_resp in zip(tool_calls, task_responses):
                content = task_resp.get("content")
                assert isinstance(content, str), "Task response content must be a string."

                tool_message = ToolMessage(role="tool", tool_call_id=tool_call.id, content=content)
                self.trajectory.messages_and_choices.append(tool_message)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

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

            if fn_name == call_sub_agent.__name__:
                tool_instance = call_sub_agent.model_validate_json(fn_args)
                tasks.append(tool_instance.to_task())

            else:
                logger.error(f"Unexpected tool call name: {tc['function']['name']}")
                raise ValueError("Unrecognized tool call name.")

        return tasks

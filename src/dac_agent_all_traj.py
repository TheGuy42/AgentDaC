from openai.types.chat.chat_completion import ChatCompletion
from openai import AsyncOpenAI
from art.types import Message
from art import Trajectory

from src.dac_agent import AgentNode, patch_completion, ChatMessage, PromptConfig, StopCriteria
from src.configs.markers import Markers

from src.utils import text as text_utils
from src.utils.visualize import trajectory_string, message_string
from src.configs.markers import Markers
from src.configs.prompts import get_prompt
from src.utils.logging import create_logger
from src.utils.io import save_base_model



class AllTrajSingleTaskAgentNode(AgentNode):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        current_depth: int = 0,
        include_histories: bool = False,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            stop_criteria=stop_criteria,
            current_depth=current_depth,
            include_histories=include_histories,
        )
        self.sub_trajectories:list[Trajectory] = []

    def create_sub_agent(self):
        return AllTrajSingleTaskAgentNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        # format kwargs
        extra_body = kwargs.pop("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        stop = kwargs.pop("stop", [Markers.TASK_END, Markers.ANSWER_END])

        completion = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )

        return patch_completion(completion)
    
    async def chat(
        self,
        prompt: ChatMessage,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Start a conversation with the agent using the provided prompt.

        Args:
            prompt (ChatMessage): The initial message to start the conversation.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.

        Returns:
            Trajectory: The trajectory of the conversation, including messages and choices.
                This trajectory is used to train an `art.TrainableModel` model.
        """

        if prompt.role != "user":
            raise ValueError("Prompt role must be 'user' to start the conversation.")

        self.trajectory.messages_and_choices.append(prompt.as_openai())

        if verbose:
            last_message = self.trajectory.messages()[-1]
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        should_break = False

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.current_depth))

            # Extract tasks from the response
            response = ChatMessage.model_validate(choice.message, from_attributes=True)
            tasks = self._parse_tasks(response)

            if should_break or len(tasks) == 0:
                break  # No tasks to delegate, so last message

            task_responses = []

            if self.stop_criteria.should_stop(self.current_depth):
                content = get_prompt(self.prompt_config.tasks_depleted)
                if content is None:
                    break

                # Provide mock answers indicating no more tasks available
                resp = ChatMessage(role="user", content=content)
                task_responses = [resp] * len(tasks)
                should_break = True

            else:
                for task in tasks:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    resp = await sub_agent.answer(task, verbose, **kwargs)
                    task_responses.append(resp)
                    self.sub_trajectories.append(sub_agent.trajectory.model_copy())
                    if self.include_histories:
                        # Add the sub-agent's trajectory to the main trajectory
                        sub_agent_trajectory = sub_agent.trajectory
                        history = AgentNode.traj2history(sub_agent_trajectory)
                        self.trajectory.additional_histories.append(history)
                        self.trajectory.additional_histories.extend(sub_agent_trajectory.additional_histories)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks)
            self.metrics["total_tasks"] += len(tasks)

            # Format the task responses
            task_answers = []
            for resp in task_responses:
                answer_text = f"{Markers.ANSWER_START} {resp.content} {Markers.ANSWER_END}"
                task_answers.append(answer_text)

            # Create a new message with the tasks' answers
            tasks_message = ChatMessage(role="user", content="\n".join(task_answers))
            self.trajectory.messages_and_choices.append(tasks_message.as_openai())

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.current_depth))

            self.stop_criteria.update_round(num_tasks=len(tasks))

        self.trajectory.finish()
        return self.trajectory
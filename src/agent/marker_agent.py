from __future__ import annotations

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from art import Trajectory

from src.agent.base import BaseAgent
from src.utils import text as text_utils
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger
from src.openai_types import Message, UserMessage
from src.configs import PromptConfig, DecompConfig
from src.configs.prompts import get_prompt


logger = create_logger(__name__)


class MarkerAgent(BaseAgent):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
        )

    def create_sub_agent(self) -> MarkerAgent:
        return MarkerAgent(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        # By default allow only a single task and answer in the response
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANS_END])
        return await super()._call(messages, **kwargs)

    # NOTE: experimental
    def remaining_budget_string(self) -> str:
        if self.decomp_config.max_tasks is None:
            return "INFO: Unlimited number of tasks available."
        return (
            f"INFO: Number of available tasks: {max(self.decomp_config.max_tasks - self.decomp_config.total_tasks, 0)}"
        )

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt["role"] != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt['role']}.")

        # NOTE: experimental
        if not self.decomp_config.should_stop(self.current_depth):
            # prompt["content"] = f"{prompt.get('content')}\n\n{self.remaining_budget_string()}"
            pass

        self.trajectory.messages_and_choices.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
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
            if completion.usage is not None:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract tasks from the response
            tasks_inputs = self._parse_tasks(self.trajectory.messages()[-1])

            # If no tasks to delegate then last message
            if should_break or len(tasks_inputs) == 0:
                break

            tasks_answers: list[str] = []

            if self.decomp_config.should_stop(self.current_depth):
                mock_answer = get_prompt(self.prompt_config.tasks_depleted)

                # If no mock answer provided then immediately stop
                if mock_answer is None:
                    break

                should_break = True
                for task in tasks_inputs:
                    # Provide mock answer indicating no more tasks available
                    tasks_answers.append(mock_answer)

            else:
                for task in tasks_inputs:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    ans = await sub_agent.answer(task, verbose, **kwargs)
                    tasks_answers.append(ans)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)

            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

            # Create a new message with all tasks' answers
            tasks_answers = [f"{Markers.ANS_START} {ans} {Markers.ANS_END}" for ans in tasks_answers]
            unified_answer = "\n".join(tasks_answers)

            # NOTE: experimental
            # unified_answer = f"{unified_answer}\n\n{self.remaining_budget_string()}"

            joined_message = UserMessage(role="user", content=unified_answer)
            self.trajectory.messages_and_choices.append(joined_message)

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

        # Update final stats
        self.metrics["response_completed"] = choice.finish_reason != "length"
        self.trajectory.finish()

        return self.trajectory

    def parse_answer(self, message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        return text_utils.extract_answer(content)

    def _parse_tasks(self, message: Message) -> list[UserMessage]:
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        tasks = text_utils.extract_tasks(content)
        return [UserMessage(role="user", content=task) for task in tasks]

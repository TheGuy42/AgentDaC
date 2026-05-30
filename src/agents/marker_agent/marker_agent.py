from __future__ import annotations

import asyncio

from src.trajectories import Trajectory, History
from src.agents.base import BaseAgent
from src.utils.visualize import trajectory_string, message_string
import src.agents.marker_agent.markers as markers
from src.agents.marker_agent.markers import Markers
from src.utils.logging import create_logger
from src.aliases import Message, UserMessage, Response


logger = create_logger(__name__)


class MarkerAgent(BaseAgent):
    def _create_subagent(self) -> MarkerAgent:
        return MarkerAgent(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  # NOTE: no support for recursive histories yet
        )

    async def call(self, messages: list[Message], **kwargs) -> Response:
        # By default allow only a single task and answer in the response
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANS_END])
        return await super().call(messages, **kwargs)

    # NOTE: experimental
    def _remaining_budget_string(self) -> str:
        if self.decomp_config.max_tasks is None:
            return "INFO: Unlimited number of tasks available."
        return (
            f"INFO: Number of available tasks: {max(self.decomp_config.max_tasks - self.decomp_config.total_tasks, 0)}"
        )

    def _should_stop(self) -> bool:
        dc = self.decomp_config
        if self.current_depth >= dc.max_depth:
            return True
        if dc.total_tasks >= dc.max_tasks:
            return True
        if dc.total_rounds >= dc.max_rounds:
            return True
        return False

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt["role"] != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt['role']}.")

        # NOTE: experimental
        # if not self._should_stop():
        #     prompt["content"] = f"{prompt.get('content')}\n\n{self._remaining_budget_string()}"

        self.trajectory.messages_and_responses.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        should_break = False

        while True:
            # Call the OpenAI API to get a response
            completion = await self.call(self.trajectory.messages(), **kwargs)
            self.trajectory.messages_and_responses.append(completion)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage is not None:
                self.metrics["direct_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract tasks from the response
            tasks_inputs = self._parse_tasks(self.trajectory.messages()[-1])

            # If no tasks to delegate then last message
            if should_break or len(tasks_inputs) == 0:
                break

            if self._should_stop():
                mock_answer = self.prompt_config.tasks_depleted
                if mock_answer is None:
                    break

                should_break = True
                tasks_answers = [mock_answer] * len(tasks_inputs)

            else:

                async def subagent_forward(task: UserMessage):
                    # create a sub-agent and get answer the task
                    sub_agent = self._create_subagent()
                    answer = await sub_agent.answer(task, verbose, **kwargs)

                    if self.additional_histories:
                        history = History(messages_and_responses=sub_agent.trajectory.messages_and_responses)
                        self.trajectory.additional_histories.append(history)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])
                    return answer

                tasks_answers = await asyncio.gather(*[subagent_forward(task) for task in tasks_inputs])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)

            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

            # Create a new message with all tasks' answers
            tasks_answers = [f"{Markers.ANS_START} {ans} {Markers.ANS_END}" for ans in tasks_answers]
            unified_answer = "\n".join(tasks_answers)

            # NOTE: experimental
            # unified_answer = f"{unified_answer}\n\n{self.remaining_budget_string()}"

            joined_message = UserMessage(role="user", name="sub-agent", content=unified_answer)
            self.trajectory.messages_and_responses.append(joined_message)

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

        # Update final stats
        self.metrics["response_completed"] = completion.choices[0].finish_reason != "length"
        self.trajectory.finish()

        return self.trajectory

    @staticmethod
    def parse_answer(message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        return markers.extract_answer(content)

    def _parse_tasks(self, message: Message) -> list[UserMessage]:
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        tasks = markers.extract_tasks(content)
        return [UserMessage(role="user", content=task) for task in tasks]

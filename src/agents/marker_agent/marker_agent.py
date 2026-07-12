from __future__ import annotations
import asyncio

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.utils.visualize import trajectory_string, message_string
import src.agents.marker_agent.markers as markers
from src.agents.marker_agent.markers import Markers
from src.utils.logging import create_logger
from src.aliases import Message, UserMessage
from src.inference import InferenceResponse


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")


class MarkerAgent(BaseAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.metrics.update(
            {
                f"{prefix}_{counter}": 0
                for counter in ("calls", "tasks", "chats", "responses_completed", "responses_incomplete")
                for prefix in METRIC_PREFIXES
            }
        )
        self.metrics.update(
            {
                "subtree_depth": 0,
                "direct_tokens": 0,
            }
        )

    def _create_subagent(self) -> MarkerAgent:
        return MarkerAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  # NOTE: no support for recursive histories yet
        )

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        # By default allow only a single task and answer in the response
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANS_END])
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super()._call(messages, **kwargs)

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

        self.decomp_config.reset()
        self.trajectory.messages_and_responses.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        should_break = False
        
        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            self.trajectory.messages_and_responses.append(completion)

            # Update metrics
            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1
            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

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
                        self.trajectory.histories.append(sub_agent.trajectory)

                    # Fold in the sub-agent's subtree contribution from this single invocation
                    for q in ("calls", "tasks", "chats", "responses_completed", "responses_incomplete"):
                        self.metrics[f"subtree_{q}"] += sub_agent.metrics[f"subtree_{q}"]

                    child_depth = 1 + sub_agent.metrics["subtree_depth"]
                    self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)
                    return answer

                tasks_answers = await asyncio.gather(*[subagent_forward(task) for task in tasks_inputs])

            # The direct tasks issued by this agent
            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_tasks"] += len(tasks_inputs)

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
        completed = int(completion.finish_reason != "length")
        incomplete = 1 - completed

        # This agent's own response completion, counted across all four quadrants
        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_responses_completed"] += completed
            self.metrics[f"{prefix}_responses_incomplete"] += incomplete
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

from __future__ import annotations
from asyncio import tasks as asyncio_tasks
from enum import StrEnum

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
import src.agents.marker_agent.markers as markers
from src.agents.marker_agent.markers import Markers
from src.utils.logging import create_logger
from src.aliases import Message, UserMessage
from src.inference import InferenceResponse


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "chats")


class MarkerErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"


class MarkerAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return MarkerErrors

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

    def _create_subagent(self) -> MarkerAgent:
        return MarkerAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  
            verbose=self.verbose,
        )

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        # By default allow only a single task and answer in the response
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANS_END])
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super()._call(messages, **kwargs)

    def _should_stop(self) -> bool:
        DC = self.decomp_config
        return DC.is_leaf(self.current_depth) or not DC.has_tasks() or not DC.has_rounds()

    async def chat(self, prompt: Message, **kwargs) -> Trajectory:
        if prompt["role"] != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt['role']}.")

        self.decomp_config.reset()
        self.append_message(prompt)

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self._call(self.trajectory.messages(), **kwargs)
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                self.trajectory.error(kind=MarkerErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            # Extract tasks from the response
            try:
                tasks_inputs = self._parse_tasks(self.trajectory.messages()[-1])
                # TODO: note that markers.extract_tasks never raises when it fails, it simply returns an empty list
                # this should be a place when we also check for malformed parsing of tasks, and insert a corresponding error in the trajectory
                # but no need to return the trajectory here
            except Exception as e:
                logger.error(f"Error parsing tasks from model response: {e}")
                self.trajectory.error(kind=MarkerErrors.PARSE_ERROR, message=str(e))
                return self.trajectory.finish()

            # If no tasks to delegate then last message
            if self._should_stop() or len(tasks_inputs) == 0:
                # TODO: should we add a parse check here for final answer, and add a corresponding error if not found?
                # i guess this is a good place to check for a final answer, and if not found, we can log an error and return the trajectory with an error state
                return self.trajectory.finish()

            async def subagent_forward(task: UserMessage) -> str:
                # create a sub-agent and get answer the task
                sub_agent = self._create_subagent()
                answer = await sub_agent.answer(task, **kwargs)
                if answer is None:
                    return "[error] sub-agent failed to produce an answer."

                if self.additional_histories:
                    self.trajectory.histories.append(sub_agent.trajectory)

                # Fold in the sub-agent's subtree contribution from this single invocation
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)
                return answer

            tasks_answers = await asyncio_tasks.gather(*[subagent_forward(task) for task in tasks_inputs])

            # The direct tasks issued by this agent
            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_tasks"] += len(tasks_inputs)

            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

            # Create a new message with all tasks' answers
            tasks_answers = [f"{Markers.ANS_START} {ans} {Markers.ANS_END}" for ans in tasks_answers]
            unified_answer = "\n".join(tasks_answers)

            joined_message = UserMessage(role="user", name="sub-agent", content=unified_answer)
            self.append_message(joined_message)

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            return None

        return markers.extract_answer(content, strict=True)

    def _parse_tasks(self, message: Message) -> list[UserMessage]:
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        # TODO: note that extract_tasks never raises when it fails, it simply returns an empty list
        # same with extract_answer
        tasks = markers.extract_tasks(content, strict=True)
        return [UserMessage(role="user", content=task) for task in tasks]

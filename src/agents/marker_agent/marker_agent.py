from __future__ import annotations
from asyncio import tasks as asyncio_tasks
from enum import StrEnum

from src.trajectory import Trajectory
from src.configs import DecompConfig, PromptConfig
from src.agents.base import BaseAgent
from src.agents.marker_agent.markers import Markers
from src.agents.marker_agent.parsing import MarkerAction, MarkerParser
from src.utils.logging import create_logger
from src.aliases import Message, UserMessage
from src.inference import InferenceClient, InferenceResponse


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "chats")


class MarkerErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"
    ILLEGAL_ACTION = "illegal_action"


class MarkerAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return MarkerErrors

    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        strict: bool = True,
        additional_histories: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
            verbose=verbose,
        )

        self.strict = strict
        self.parser = MarkerParser(strict=self.strict)

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

    def create_subagent(self) -> MarkerAgent:
        agent = MarkerAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,
            strict=self.strict,
            verbose=self.verbose,
        )

        if self.additional_histories:
            self.trajectory.histories.append(agent.trajectory)

        return agent

    async def call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super().call(messages, **kwargs)

    def _allowed_actions(self) -> set[MarkerAction]:
        DC = self.decomp_config
        allowed = {MarkerAction.ANSWER}  # answering is always allowed
        if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
            allowed.add(MarkerAction.TASK)  # can still delegate
        return allowed

    async def _subagent_forward(self, task_text: str, **kwargs) -> str:
        """Create a sub-agent to handle the task, and return its answer."""
        sub_agent = self.create_subagent()
        answer = await sub_agent.answer(UserMessage(role="user", content=task_text), **kwargs)
        if answer is None:
            return "[error] sub-agent failed to produce an answer."

        # Fold in the sub-agent's subtree contribution from this single invocation
        for q in METRIC_COUNTERS:
            self.metrics[f"subtree_{q}"] += sub_agent.metrics[f"subtree_{q}"]

        child_depth = 1 + sub_agent.metrics["subtree_depth"]
        self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)
        return answer

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
                completion = await self.call(self.trajectory.messages(), stop=self.parser.stop_tags(), **kwargs)
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                self.trajectory.error(kind=MarkerErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Parse model output
                turn = self.parser.parse(completion.content)
            except Exception as e:
                message = f"Failed to parse model output: {e}"
                self.trajectory.error(kind=MarkerErrors.PARSE_ERROR, message=message)
                if not self.decomp_config.has_rounds():
                    return self.trajectory.finish()
                self.append_message(UserMessage(role="user", content=f"[error] {message}"))
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Record recoverable parse errors in the model's output
            if parse_error := self.parser.validate(completion.content):
                self.trajectory.error(kind=MarkerErrors.PARSE_ERROR, message=parse_error)

            allowed = self._allowed_actions()

            # Terminal: answer is present
            if turn.answers is not None:
                if turn.tasks is not None:
                    self.trajectory.error(kind=MarkerErrors.ILLEGAL_ACTION, message="Ambiguous turn: both <task> and <answer> present.")
                return self.trajectory.finish()

            elif turn.tasks is not None:
                # Check if delegating a task is allowed
                if MarkerAction.TASK not in allowed:
                    message = "Delegating a task is not allowed in the current state."
                    self.trajectory.error(kind=MarkerErrors.ILLEGAL_ACTION, message=message)
                    if not self.decomp_config.has_rounds():
                        return self.trajectory.finish()
                    self.append_message(UserMessage(role="user", content=f"[error] {message}"))
                    self.decomp_config.update_round(num_tasks=0)
                    continue

                # The direct tasks issued by this agent
                # TODO: handle case when multiple tasks are issued but we have less available tasks in the decomp_config
                tasks_answers = await asyncio_tasks.gather(*[self._subagent_forward(task, **kwargs) for task in turn.tasks])

                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += len(turn.tasks)

                self.decomp_config.update_round(num_tasks=len(turn.tasks))

                # Report all sub-task answers back to the model
                unified_answer = "\n".join(f"{Markers.ANS_START} {ans} {Markers.ANS_END}" for ans in tasks_answers)
                joined_message = UserMessage(role="user", name="sub-agent", content=unified_answer)
                self.append_message(joined_message)

            else:  # This should never happen, but just in case
                raise ValueError(f"Unexpected turn: {turn}. Allowed actions: {allowed}.")

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            return None

        try:
            turn = MarkerParser(strict=False).parse(content)
            return turn.answers[-1] if turn.answers is not None else None

        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return None

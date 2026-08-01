from __future__ import annotations
from enum import StrEnum

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.agents.perst_agent.actions import TurnAction
from src.agents.regex_agent.regex_agent import GuidedRegex
from src.aliases import Message, UserMessage
from src.configs import PromptConfig, DecompConfig
from src.inference import InferenceClient, InferenceResponse
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "agents", "chats")


class PersistentErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"


class PersistentAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return PersistentErrors

    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
        force_thinking: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
            verbose=verbose,
        )

        self.force_thinking = force_thinking

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

        # We support a persistent sub-agent across chat rounds
        self.sub_agent: PersistentAgent | None = None

    def _create_regex(self) -> GuidedRegex:
        """
        Rules for allowed actions:
        0) If force_thinking is True and this is the first round: must THINK.
        1) If sub_agent is None: cannot ISSUE_TASK.
        2) If at a leaf (depth >= max_depth): cannot ISSUE_FRESH_TASK or ISSUE_TASK.
        3) If rounds remain (total_rounds < max_rounds): may THINK.
        4) If no rounds remain: must ANSWER.
        5) If tasks exhausted (total_tasks >= max_tasks): cannot ISSUE_FRESH_TASK or ISSUE_TASK.
        6) ANSWER is always allowed.
        """

        DC = self.decomp_config

        if self.force_thinking and DC.total_rounds == 0 and DC.has_rounds():
            return GuidedRegex(TurnAction.THINK)

        allowed = [TurnAction.ANSWER]
        if DC.has_rounds():
            allowed.append(TurnAction.THINK)
            if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
                allowed.append(TurnAction.ISSUE_FRESH_TASK)
                if self.sub_agent is not None:
                    allowed.append(TurnAction.ISSUE_TASK)

        return GuidedRegex(*allowed)

    async def call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        regex: GuidedRegex = kwargs.pop("regex")
        kwargs = self.client.update_kwargs(kwargs, regex_schema=regex.model_pattern, include_stop_str_in_output=True)
        return await super().call(messages, **kwargs)

    def create_subagent(self) -> PersistentAgent:
        agent = PersistentAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,
            force_thinking=self.force_thinking,
            verbose=self.verbose,
        )
        
        if self.additional_histories:
            self.trajectory.histories.append(agent.trajectory)
        
        return agent

    async def chat(self, prompt: Message, **kwargs) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self.append_message(prompt)

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            regex = self._create_regex()

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self.call(self.trajectory.messages(), regex=regex, **kwargs)
            except Exception as e:
                logger.debug(f"Error during model call: {e}")
                self.trajectory.error(kind=PersistentErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Parse model output
                turn = regex.parse(completion.content)
            except Exception as e:
                logger.debug(f"Failed to parse model output: {e}")
                self.trajectory.error(kind=PersistentErrors.PARSE_ERROR, message=str(e))
                return self.trajectory.finish()

            # Finish if the model chose to answer
            if turn.action == TurnAction.ANSWER:
                return self.trajectory.finish()

            # If the model chose to think, continue
            elif turn.action == TurnAction.THINK:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Create a new sub-agent
            elif turn.action == TurnAction.ISSUE_FRESH_TASK:
                self.sub_agent = self.create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

            # Issue a sub-task to the current sub-agent
            if turn.action == TurnAction.ISSUE_FRESH_TASK or turn.action == TurnAction.ISSUE_TASK:
                if self.sub_agent is None:
                    raise ValueError("Sub-agent must be created before it can be asked to perform a task.")

                task = UserMessage(role="user", content=turn.text)
                task_answer = await self.sub_agent.answer(task, **kwargs)
                if task_answer is None:
                    task_answer = "[error] sub-agent failed to produce an answer."

                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.append_message(task_response)

                # The direct task issued by this agent
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation.
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += self.sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + self.sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            else:  # Unrecognized action, should be impossible
                raise ValueError(f"Unrecognized action: {turn.action}")

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.debug(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.debug(f"Expected message content to be a string, got {type(content)}")
            return None

        try:
            schema = GuidedRegex(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text

        except Exception as e:
            logger.debug(f"Failed to parse final answer: {e}")
            return None

from __future__ import annotations

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.agents.perst_agent.actions import TurnAction
from src.agents.regex_agent.regex_agent import GuidedRegex
from src.aliases import Message, UserMessage
from src.configs import PromptConfig, DecompConfig
from src.inference import InferenceClient, InferenceResponse
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "agents", "chats", "responses_completed", "responses_incomplete")


class PersistentAgent(BaseAgent):
    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
        force_thinking: bool = False,
    ):
        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
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
        dc = self.decomp_config
        is_leaf = self.current_depth >= dc.max_depth
        has_rounds = dc.total_rounds < dc.max_rounds
        tasks_available = dc.total_tasks < dc.max_tasks

        if self.force_thinking and dc.total_rounds == 0:
            return GuidedRegex(TurnAction.THINK)

        allowed = [TurnAction.ANSWER]
        if has_rounds:
            allowed.append(TurnAction.THINK)
            if (not is_leaf) and tasks_available:
                allowed.append(TurnAction.ISSUE_FRESH_TASK)
                if self.sub_agent is not None:
                    allowed.append(TurnAction.ISSUE_TASK)

        return GuidedRegex(*allowed)

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        regex: GuidedRegex = kwargs.pop("regex")
        kwargs = self.client.update_kwargs(kwargs, regex_schema=regex.model_pattern, include_stop_str_in_output=True)
        return await super()._call(messages, **kwargs)

    def _create_subagent(self) -> PersistentAgent:
        return PersistentAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  # NOTE: no support for recursive histories yet
            force_thinking=self.force_thinking,
        )

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self.trajectory.messages_and_responses.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            # Model turn
            regex = self._create_regex()
            completion = await self._call(self.trajectory.messages(), regex=regex, **kwargs)
            self.trajectory.messages_and_responses.append(completion)

            # Update metrics
            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1
            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract raw content and parse it
            assistant_msg = self.trajectory.messages()[-1]
            turn = regex.parse(assistant_msg.get("content"))

            # Finish if the model chose to answer
            if turn.action == TurnAction.ANSWER:
                self.decomp_config.update_round(num_tasks=0)
                break

            # If the model chose to think, continue
            elif turn.action == TurnAction.THINK:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)

            # Create a new sub-agent
            elif turn.action == TurnAction.ISSUE_FRESH_TASK:
                self.sub_agent = self._create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

                if self.additional_histories:
                    # Each sub-agent defines its own history
                    # The history is dynamically updated as the sub-agent is invoked, as expected.
                    self.trajectory.histories.append(self.sub_agent.trajectory)

            # Issue a sub-task to the current sub-agent
            if turn.action == TurnAction.ISSUE_FRESH_TASK or turn.action == TurnAction.ISSUE_TASK:
                assert self.sub_agent is not None, "Sub-agent must be created before it can be asked to perform a task."

                task = UserMessage(role="user", content=turn.text)
                task_answer = await self.sub_agent.answer(task, verbose, **kwargs)
                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.trajectory.messages_and_responses.append(task_response)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                # The direct task issued by this agent
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation.
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += self.sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + self.sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            # Unrecognized action, stop the agent loop
            if turn.action not in [e for e in TurnAction]:
                logger.info(f"Unhandled action: {turn.action}")
                break

        # Update final stats
        completed = int((completion.finish_reason != "length") and (turn.action == TurnAction.ANSWER))
        incomplete = 1 - completed

        # This agent's own response completion, counted across all four quadrants
        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_responses_completed"] += completed
            self.metrics[f"{prefix}_responses_incomplete"] += incomplete

        self.trajectory.finish()
        return self.trajectory

    def parse_answer(self, message: Message | InferenceResponse) -> str:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            raise ValueError("parse_answer expects an InferenceResponse.")

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        try:
            schema = GuidedRegex(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content

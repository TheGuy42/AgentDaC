from __future__ import annotations
from enum import StrEnum
from typing import Any
from dataclasses import dataclass
import re

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.agents.regex_agent.actions import TurnAction
from src.aliases import Message, UserMessage
from src.inference import InferenceResponse
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "chats")


class RegexErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"


@dataclass
class AgentTurn:
    action: str
    text: str
    raw: str


class GuidedRegex:
    def __init__(self, *actions: str) -> None:
        if not actions:
            raise ValueError("At least one allowed action must be provided.")
        self.actions = actions

        alt = "|".join(re.escape(act) for act in self.actions)
        self.model_pattern = rf"^\s?Action: (?:{alt})\r?\nText: [\s\S]*$"
        self.parse_pattern = rf"^\s?Action: (?P<action>{alt})\r?\nText: (?P<text>[\s\S]*)$"
        self.regex = re.compile(self.parse_pattern)

    def parse(self, content: Any) -> AgentTurn:
        if not isinstance(content, str):
            raise ValueError("Content to parse must be a string.")

        m = self.regex.match(content)
        if not m:
            logger.debug(f"Failed to match content against regex: {self.parse_pattern}")
            logger.debug(f"Raw content was: {content}")
            raise ValueError(f"Content does not match the expected regex pattern. Allowed actions: {self.actions}.")

        action_val = m.group("action").strip()
        text_val = m.group("text").strip()

        if action_val not in self.actions:
            raise ValueError(f"Action {action_val} is not allowed for this turn. Allowed: {self.actions}")

        return AgentTurn(action=action_val, text=text_val, raw=content)


class RegexAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return RegexErrors

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

    def _create_regex(self) -> GuidedRegex:
        """
        Rules for allowed actions:
        1) If at a leaf (depth >= max_depth): cannot ISSUE_TASK.
        2) If rounds remain (total_rounds < max_rounds): may THINK.
        3) If no rounds remain: must ANSWER.
        4) If tasks exhausted (total_tasks >= max_tasks): cannot ISSUE_TASK.
        5) ANSWER is always allowed.
        """
        DC = self.decomp_config
        if DC.is_leaf(self.current_depth):
            DC.max_rounds = 1  # Force only one round at leaf nodes

        allowed = [TurnAction.ANSWER]
        if DC.has_rounds():
            allowed.append(TurnAction.THINK)
            if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
                allowed.append(TurnAction.ISSUE_TASK)

        return GuidedRegex(*allowed)

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        regex: GuidedRegex = kwargs.pop("regex")
        kwargs = self.client.update_kwargs(kwargs, regex_schema=regex.model_pattern, include_stop_str_in_output=True)
        return await super()._call(messages, **kwargs)

    def _create_subagent(self) -> BaseAgent:
        return RegexAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  
            verbose=self.verbose,
        )

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
            # Model turn
            regex = self._create_regex()

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                completion = await self._call(self.trajectory.messages(), regex=regex, **kwargs)
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                self.trajectory.error(kind=RegexErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Extract raw content and parse it
                assistant_msg = self.trajectory.messages()[-1]
                turn = regex.parse(assistant_msg.get("content"))
            except Exception as e:
                logger.error(f"Error parsing model response: {e}")
                self.trajectory.error(kind=RegexErrors.PARSE_ERROR, message=str(e))
                return self.trajectory.finish()

            # Finish if the model chose to answer
            if turn.action == TurnAction.ANSWER:
                return self.trajectory.finish()

            # If the model chose to think, continue
            elif turn.action == TurnAction.THINK:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)

            # Issue a task and get the answer from a sub-agent
            elif turn.action == TurnAction.ISSUE_TASK:
                sub_agent = self._create_subagent()
                task = UserMessage(role="user", content=turn.text)

                task_answer = await sub_agent.answer(task, **kwargs)
                if task_answer is None:
                    task_answer = "[error] sub-agent failed to produce an answer."

                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.append_message(task_response)

                if self.additional_histories:
                    self.trajectory.histories.append(sub_agent.trajectory)

                # The direct task issued by this agent
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            else:  # Unrecognized action, should be impossible
                raise ValueError(f"Unrecognized action: {turn.action}.")

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            return None

        try:
            schema = GuidedRegex(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return None

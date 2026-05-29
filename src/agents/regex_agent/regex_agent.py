from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from openai.types.chat import ChatCompletion

from src.trajectories import Trajectory, History
from src.agents.base import BaseAgent
from src.agents.regex_agent.actions import TurnAction
from src.openai_types import Message, UserMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger
import re


logger = create_logger(__name__)


@dataclass
class AgentTurn:
    action: TurnAction
    text: str
    raw: str


class GuidedRegex:
    def __init__(self, *actions: TurnAction) -> None:
        if not actions:
            raise ValueError("At least one allowed action must be provided.")
        self.actions = actions

        alt = "|".join(re.escape(act.value) for act in self.actions)
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
            raise ValueError(f"Content does not match the required pattern: {self.parse_pattern}")

        action_val = m.group("action").strip()
        text_val = m.group("text").strip()

        action = TurnAction(action_val)
        if action not in self.actions:
            raise ValueError(f"Action {action} is not allowed for this turn. Allowed: {self.actions}")

        return AgentTurn(action=action, text=text_val, raw=content)


class RegexAgent(BaseAgent):
    def _create_regex(self) -> GuidedRegex:
        """
        Rules for allowed actions:
        1) If at a leaf (depth >= max_depth): cannot ISSUE_TASK.
        2) If rounds remain (total_rounds < max_rounds): may THINK.
        3) If no rounds remain: must ANSWER.
        4) If tasks exhausted (total_tasks >= max_tasks): cannot ISSUE_TASK.
        5) ANSWER is always allowed.
        """
        if self.current_depth >= self.decomp_config.max_depth:
            self.decomp_config.max_rounds = 1  # Force only one round at leaf nodes

        dc = self.decomp_config
        is_leaf = self.current_depth >= dc.max_depth
        has_rounds = dc.total_rounds < dc.max_rounds
        tasks_available = dc.total_tasks < dc.max_tasks

        allowed = [TurnAction.ANSWER]
        if has_rounds:
            allowed.append(TurnAction.THINK)
            if (not is_leaf) and tasks_available:
                allowed.append(TurnAction.ISSUE_TASK)

        return GuidedRegex(*allowed)

    async def call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        regex: GuidedRegex = kwargs.pop("regex")
        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        extra_body["guided_regex"] = regex.model_pattern

        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    def _create_subagent(self) -> BaseAgent:
        return RegexAgent(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  # NOTE: no support for recursive histories yet
        )

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.trajectory.messages_and_choices.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        self.metrics.setdefault("direct_thinks", 0)
        self.metrics.setdefault("total_thinks", 0)

        while True:
            # Model turn
            regex = self._create_regex()
            completion = await self.call(self.trajectory.messages(), regex=regex, **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage is not None:
                self.metrics["direct_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract raw content and parse it
            assistant_msg = self.trajectory.messages()[-1]
            turn = regex.parse(assistant_msg.get("content"))

            # Finish if the model chose to answer
            if turn.action == TurnAction.ANSWER:
                break

            # If the model chose to think, continue
            elif turn.action == TurnAction.THINK:
                self.metrics["direct_thinks"] += 1
                self.metrics["total_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)

            # Issue a task and get the answer from a sub-agent
            elif turn.action == TurnAction.ISSUE_TASK:
                sub_agent = self._create_subagent()
                task = UserMessage(role="user", content=turn.text)
                task_answer = await sub_agent.answer(task, verbose, **kwargs)
                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.trajectory.messages_and_choices.append(task_response)

                if self.additional_histories:
                    history = History(messages_and_choices=sub_agent.trajectory.messages_and_choices)
                    self.trajectory.additional_histories.append(history)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                # Update metrics from sub-agent
                self.metrics["direct_tasks"] += 1
                self.metrics["total_tasks"] += 1 + sub_agent.metrics["total_tasks"]
                self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                self.metrics["total_thinks"] += sub_agent.metrics["total_thinks"]
                self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

                self.decomp_config.update_round(num_tasks=1)

            else:
                raise ValueError(f"Unhandled action: {turn.action}")

        # Update final stats
        self.metrics["response_completed"] = (choice.finish_reason != "length") and (turn.action == TurnAction.ANSWER)
        self.trajectory.finish()

        return self.trajectory

    @staticmethod
    def parse_answer(message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
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

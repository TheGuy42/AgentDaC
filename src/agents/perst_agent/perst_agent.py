from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from openai.types.chat import ChatCompletion
from openai import AsyncOpenAI
from art.trajectories import Trajectory, History

from src.agents.base import BaseAgent
from src.agents.perst_agent.actions import TurnAction
from src.openai_types import Message, UserMessage
from src.configs import PromptConfig, DecompConfig
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


class PersistentAgent(BaseAgent):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
        )

        # Additional metrics that we track
        self.metrics.update(
            {
                "direct_thinks": 0,
                "total_thinks": 0,
                "direct_agents": 0,
                "total_agents": 0,
                "responses_completed": 0,
                "responses_incomplete": 0,
            }
        )

        # We support a persistent sub-agent across chat rounds
        self.sub_agent: PersistentAgent | None = None

        # Metrics from only the latest chat()/answer() invocation
        self.latest_metrics: dict[str, int] = {
            "total_tasks": 0,
            "total_calls": 0,
            "total_thinks": 0,
            "total_agents": 0,
        }

    def _create_regex(self) -> GuidedRegex:
        """
        Rules for allowed actions:
        0) If sub_agent is None: cannot ASK_SUBAGENT.
        1) If at a leaf (depth >= max_depth): cannot CREATE_SUBAGENT or ASK_SUBAGENT.
        2) If rounds remain (total_rounds < max_rounds): may THINK.
        3) If no rounds remain: must ANSWER.
        4) If tasks exhausted (total_tasks >= max_tasks): cannot CREATE_SUBAGENT or ASK_SUBAGENT.
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
                allowed.append(TurnAction.CREATE_SUBAGENT)
                if self.sub_agent is not None:
                    allowed.append(TurnAction.ASK_SUBAGENT)

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

    def _create_subagent(self) -> PersistentAgent:
        return PersistentAgent(
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

        self.decomp_config.reset()
        self.trajectory.messages_and_choices.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if "prompt" not in self.metadata and isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Reset metrics of the current run
        for k in self.latest_metrics:
            self.latest_metrics[k] = 0

        while True:
            # Model turn
            regex = self._create_regex()
            completion = await self.call(self.trajectory.messages(), regex=regex, **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            self.latest_metrics["total_calls"] += 1
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
                self.latest_metrics["total_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)

            # Create a new sub-agent if requested
            elif turn.action == TurnAction.CREATE_SUBAGENT:
                self.sub_agent = self._create_subagent()
                self.metrics["direct_agents"] += 1
                self.metrics["total_agents"] += 1
                self.latest_metrics["total_agents"] += 1

                if self.additional_histories:  # Each sub-agent defines its own history
                    self.trajectory.additional_histories.append(History(messages_and_choices=[]))

            if turn.action == TurnAction.CREATE_SUBAGENT or turn.action == TurnAction.ASK_SUBAGENT:
                assert self.sub_agent is not None, "Sub-agent must be created before it can be asked."

                task = UserMessage(role="user", content=turn.text)
                task_answer = await self.sub_agent.answer(task, verbose, **kwargs)
                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.trajectory.messages_and_choices.append(task_response)

                if self.additional_histories:  # Update sub-agent history
                    agent_history = self.trajectory.additional_histories[-1]
                    agent_history.messages_and_choices = self.sub_agent.trajectory.messages_and_choices

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                # Update metrics from sub-agent
                self.metrics["direct_tasks"] += 1
                self.metrics["total_tasks"] += 1
                self.metrics["max_depth"] = max(1 + self.sub_agent.metrics["max_depth"], self.metrics["max_depth"])
                self.latest_metrics["total_tasks"] += 1

                # For cumulative metrics, only add the delta from the
                # last sub-agent invocation, to avoid double-counting
                for metric in ["total_tasks", "total_calls", "total_thinks", "total_agents"]:
                    self.metrics[metric] += self.sub_agent.latest_metrics[metric]
                    self.latest_metrics[metric] += self.sub_agent.latest_metrics[metric]

                self.decomp_config.update_round(num_tasks=1)

        # Update final stats
        completed = int((choice.finish_reason != "length") and (turn.action == TurnAction.ANSWER))
        self.metrics["responses_completed"] += completed
        self.metrics["responses_incomplete"] += 1 - completed

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

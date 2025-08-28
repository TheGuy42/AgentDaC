from __future__ import annotations

import base64
import json
import re
from typing import Any

from enum import Enum
from dataclasses import dataclass

from openai.types.chat import ChatCompletion
from art import Trajectory

from src.dac_agent import AgentNode
from src.configs.prompts import get_prompt
from src.openai_types import Message, UserMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger


logger = create_logger(__name__)


class AgentAction(str, Enum):
    THINK = "think"
    ISSUE_TASK = "issue_task"
    ANSWER = "answer"


@dataclass
class AgentTurn:
    action: AgentAction
    text: str
    raw: dict[str, Any]


class GuidedSchema:
    """
    Pure schema helper: builds the JSON schema and parses a raw JSON string into an AgentTurn.

    Public API:
    - build() -> dict: JSON schema object for response_format
    - parse(content: str) -> AgentTurn: parse a raw assistant content string
    """

    def __init__(self, *actions: AgentAction) -> None:
        self.actions = actions

    # NOTE: its possible that structured output automatically handles special characters and escapes them automatically, test
    def build(self) -> dict[str, Any]:
        """Return the schema descriptor for response_format."""
        turn_schema = {
            "type": "object",
            "additionalProperties": False,
            "description": (
                "Return exactly two fields: 'action' and 'text_b64'.\n"
                "- action: one of " + str([a.value for a in self.actions]) + " for this turn.\n"
                "- text_b64: Base64-encoded UTF-8 text. Always required, regardless of action."
            ),
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [a.value for a in self.actions],
                    "description": "What to do this turn.",
                },
                "text_b64": {
                    "type": "string",
                    "description": (
                        "Base64 (UTF-8) encoded text for ALL actions.\n"
                        "- think: reasoning notes or plan\n"
                        "- issue_task: fully self-contained sub-task prompt\n"
                        "- answer: final answer to the original question"
                    ),
                },
            },
            "required": ["action", "text_b64"],
        }
        return {
            "name": "dac_turn",
            "description": "Schema for a single AgentDaC turn.",  # TODO: should actually be descriptive, as it is used by the model
            "strict": True,  # NOTE: maybe should be false actually
            "schema": turn_schema,
        }

    def parse(self, content: str) -> AgentTurn:
        try:
            obj = json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse guided JSON content: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("Top-level guided JSON is not an object.")

        action_val = obj.get("action")
        if not isinstance(action_val, str):
            raise ValueError("Missing or invalid 'action' field.")
        try:
            action = AgentAction(action_val)
        except Exception as e:
            raise ValueError(f"Invalid action {action_val!r}. Allowed: {self.actions}.") from e

        if action not in self.actions:
            raise ValueError(f"Action {action.value!r} is not allowed for this turn. Allowed: {self.actions}")

        text = self._decode_text_b64(obj.get("text_b64"))
        return AgentTurn(action=action, text=text.strip(), raw=obj)

    @staticmethod
    def _decode_text_b64(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("Field 'text_b64' must be a non-empty string.")
        # Remove all whitespace to be robust to LLM formatting
        cleaned = re.sub(r"\s+", "", value)
        try:
            decoded_bytes = base64.b64decode(cleaned, validate=True)
        except Exception as e:
            raise ValueError(f"Failed to base64-decode 'text_b64': {e}") from e
        try:
            return decoded_bytes.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Decoded 'text_b64' is not valid UTF-8: {e}") from e


class AgentGuidedNode(AgentNode):
    """
    Guided-decoding agent that avoids function/tool calls and uses a very simple
    JSON control schema per assistant turn. The schema supports three actions,
    with exactly one action per turn:

    - think: Spend the turn thinking. The model writes thoughts in `text_b64`.
    - issue_task: Delegate exactly one sub-task. The sub-task prompt is in `text_b64`.
    - answer: Provide the final answer in `text_b64`.

    Design goals:
    - Only one field for content: `text_b64` (UTF-8, base64-encoded) to avoid any
      escaping issues for thinking, sub-task prompts, or final answers.
    - Only one task per turn is allowed by design.
    - A dedicated "think" action lets the model think before deciding to issue
    a task or finalize with an answer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: temporarily here, should be in self.decomp_config
        self.think_turns = 0
        self.max_think_turns = 8

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """Call the model with JSON-schema structured output enabled. Uses all actions by default."""

        kwargs.setdefault("stop", [])
        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)

        if self.decomp_config.should_stop(self.current_depth):
            schema = GuidedSchema(AgentAction.ANSWER)
        elif self.think_turns >= self.max_think_turns:
            schema = GuidedSchema(AgentAction.ISSUE_TASK, AgentAction.ANSWER)
        else:
            schema = GuidedSchema(AgentAction.THINK, AgentAction.ISSUE_TASK, AgentAction.ANSWER)

        schema_descriptor = schema.build()
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema_descriptor}
        extra_body["guided_json"] = schema_descriptor["schema"]

        return await super()._call(messages=messages, **kwargs)

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Orchestrate a conversation using structured outputs:
        - think: append a THINK block and continue the loop
        - issue_task: delegate exactly one sub-task to a sub-agent and feed its answer back
        - answer: finish the conversation

        For now, all actions are permitted each turn.
        """
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')!r}.")

        self.trajectory.messages_and_choices.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Single schema/parser instance per chat loop
        schema = GuidedSchema(AgentAction.THINK, AgentAction.ISSUE_TASK, AgentAction.ANSWER)

        while True:
            # Model turn
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

            try:
                # Extract raw content and parse it
                assistant_msg = self.trajectory.messages()[-1]
                raw_content = assistant_msg.get("content")
                raw_content = raw_content if isinstance(raw_content, str) else ""
                turn = schema.parse(raw_content)

            except Exception as e:
                if choice.finish_reason == "length":
                    break
                else:
                    logger.error(f"Error parsing guided JSON: {e}")
                    raise

            # If the model chose to answer, stop
            if turn.action == AgentAction.ANSWER:
                break

            # If the model chose to think, continue
            elif turn.action == AgentAction.THINK:
                self.think_turns += 1
                if self.think_turns >= self.max_think_turns:
                    logger.warning("Max consecutive think turns reached; stopping to avoid infinite loop.")
                    break
                continue

            elif turn.action == AgentAction.ISSUE_TASK:
                # If we cannot create more tasks, inject a tasks_depleted answer
                if self.decomp_config.should_stop(self.current_depth):
                    answer = get_prompt(self.prompt_config.tasks_depleted)
                    if answer is None:
                        break
                else:
                    # Call sub-agent
                    task_text = turn.text
                    task = UserMessage(role="user", content=task_text)
                    sub_agent = self.create_sub_agent()
                    answer = await sub_agent.answer(task, verbose, **kwargs)

                    # Update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

                # Update metrics for this delegation
                self.metrics["direct_tasks"] += 1
                self.metrics["total_tasks"] += 1

                response = UserMessage(role="user", content=f"{Markers.ANSWER_START} {answer} {Markers.ANSWER_END}")
                self.trajectory.messages_and_choices.append(response)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            else:
                raise ValueError(f"Unhandled action: {turn.action}")

            # Inform decomp state that we used one task this round
            self.decomp_config.update_round(num_tasks=1)

        # Update final stats
        self.metrics["response_completed"] = choice.finish_reason != "length"
        self.trajectory.finish()

        return self.trajectory

    async def answer(self, prompt: Message, verbose: bool = False, **kwargs) -> str:
        trajectory = await self.chat(prompt, verbose=verbose, **kwargs)
        answer_message = self.parse_answer(trajectory.messages()[-1])
        answer_content = answer_message.get("content")

        if not isinstance(answer_content, str):
            raise ValueError("Final answer message has no valid content.")

        try:
            schema = GuidedSchema(AgentAction.ANSWER)
            turn = schema.parse(answer_content)
            return turn.text

        except Exception as e:
            logger.error(f"Error parsing final answer guided JSON: {e}")
            return answer_content.strip()

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
from src.openai_types import Message, UserMessage, AssistantMessage
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
    Owns the JSON schema for a guided turn and all related parsing/validation.

    Public API:
    - schema() -> dict: returns the JSON schema object for response_format
    - parse_turn(message: Message) -> AgentTurn
    - parse_action_and_text(message: Message) -> tuple[GuidedAction, str]
    """

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        # Keep schema as uniform and simple as possible for the LLM.
        # Always return exactly two fields: {"action": "think|issue_task|answer", "text_b64": "..."}
        turn_schema = {
            "type": "object",
            "additionalProperties": False,
            "description": (
                "Return exactly two fields: 'action' and 'text_b64'.\n"
                "- action: one of ['think', 'issue_task', 'answer'] for every turn.\n"
                "- text_b64: Base64-encoded UTF-8 text. Always required, regardless of action."
            ),
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [a.value for a in AgentAction],
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
            "strict": True,
            "schema": turn_schema,
        }

    @staticmethod
    def _ensure_assistant_message(message: Message) -> None:
        if message.get("role") != "assistant":
            raise ValueError(f"Expected assistant role, got {message.get('role')!r}.")

    @staticmethod
    def _parse_json_content(content: Any) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Assistant message content is empty.")
        try:
            obj = json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse guided JSON content: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("Top-level guided JSON is not an object.")
        return obj

    @classmethod
    def _parse_guided_object(cls, message: Message) -> dict[str, Any]:
        cls._ensure_assistant_message(message)
        content = message.get("content")
        return cls._parse_json_content(content)

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

    @classmethod
    def parse_turn(cls, message: Message) -> AgentTurn:
        obj = cls._parse_guided_object(message)

        action_val = obj.get("action")
        if not isinstance(action_val, str):
            raise ValueError("Missing or invalid 'action' field.")
        try:
            action = AgentAction(action_val)
        except Exception:
            expected = [a.value for a in AgentAction]
            raise ValueError(f"Invalid action {action_val!r}. Expected one of {expected}.")

        text = cls._decode_text_b64(obj.get("text_b64"))
        return AgentTurn(action=action, text=text.strip(), raw=obj)


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

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """Call the model with JSON-schema structured output enabled."""
        # Ensure we are not mixing tool-calls with structured outputs
        kwargs = dict(kwargs)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        # For JSON-only outputs, no stop sequences are necessary
        kwargs.setdefault("stop", [])

        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)

        json_schema_obj = GuidedSchema.get_schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema_obj,
        }
        # Some vLLM builds also look for this
        extra_body.setdefault("guided_json", json_schema_obj.get("schema", json_schema_obj))

        return await super()._call(messages=messages, **kwargs)

    def parse_tasks(self, message: Message) -> list[UserMessage]:
        """Extract at most one sub-task from the guided turn."""
        try:
            turn = GuidedSchema.parse_turn(message)
            action, text = turn.action, turn.text
            if action == AgentAction.ISSUE_TASK and text:
                return [UserMessage(role="user", content=text)]
        except Exception:
            pass
        return []

    def parse_answer(self, message: Message) -> AssistantMessage:
        try:
            turn = GuidedSchema.parse_turn(message)
            action, text = turn.action, turn.text
            if action == AgentAction.ANSWER:
                return AssistantMessage(role="assistant", content=text)
        except Exception:
            pass
        return AssistantMessage(role="assistant", content="")


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

        # Prevent unbounded thinking loops
        think_turns = 0
        max_think_turns = 8

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

            assistant_msg = self.trajectory.messages()[-1]
            try:
                turn = GuidedSchema.parse_turn(assistant_msg)
                action = turn.action
            except Exception as e:
                logger.warning(f"Failed to parse guided turn: {e}")
                break

            # If the model chose to answer, finalize and stop
            if action == AgentAction.ANSWER:
                final = turn.text
                self.trajectory.messages_and_choices.append(
                    AssistantMessage(role="assistant", content=f"{Markers.ANSWER_START} {final} {Markers.ANSWER_END}")
                )
                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))
                break

            # If the model chose to think, append the think block and continue
            if action == AgentAction.THINK:
                think_text = turn.text
                think_block = f"{Markers.THINK_START} {think_text} {Markers.THINK_END}"
                self.trajectory.messages_and_choices.append(UserMessage(role="user", content=think_block))
                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))
                think_turns += 1
                if think_turns >= max_think_turns:
                    logger.warning("Max consecutive think turns reached; stopping to avoid infinite loop.")
                    break
                continue

            # action == ISSUE_TASK: delegate exactly one task
            task_text = turn.text
            if not isinstance(task_text, str) or not task_text.strip():
                logger.warning("issue_task selected but no task text present; stopping.")
                break
            task = UserMessage(role="user", content=task_text)

            # If we cannot create more tasks at this depth, inject a tasks_depleted answer
            if self.decomp_config.should_stop(self.current_depth):
                mock_answer = get_prompt(self.prompt_config.tasks_depleted)
                if mock_answer is None:
                    break
                task_resp = AssistantMessage(role="assistant", content=mock_answer)
            else:
                # Call sub-agent
                sub_agent = self.create_sub_agent()
                resp = await sub_agent.answer(task, verbose, **kwargs)
                task_resp = resp

                # Update metrics from sub-agent
                self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics for this delegation
            self.metrics["direct_tasks"] += 1
            self.metrics["total_tasks"] += 1

            # Feed the task answer back as a single user message using ANSWER markers
            content = task_resp.get("content")
            content = content if isinstance(content, str) else ""
            joined_message = UserMessage(
                role="user",
                content=f"{Markers.ANSWER_START} {content} {Markers.ANSWER_END}",
            )
            self.trajectory.messages_and_choices.append(joined_message)

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Inform decomp state that we used one task this round
            self.decomp_config.update_round(num_tasks=1)

        # Update final stats
        self.metrics["response_completed"] = choice.finish_reason != "length"
        self.trajectory.finish()

        return self.trajectory

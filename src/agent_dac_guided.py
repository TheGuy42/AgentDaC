from __future__ import annotations

import base64
import json
import re
from typing import Any

from openai.types.chat import ChatCompletion
from art import Trajectory

from src.dac_agent import AgentNode
from src.configs.prompts import get_prompt
from src.openai_types import Message, UserMessage, AssistantMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger


logger = create_logger(__name__)


def _maybe_decode_base64(s: str) -> str:
    """Try decoding a base64 string; return original if decoding fails."""
    if not isinstance(s, str) or len(s) == 0:
        return s
    b64_pattern = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")
    if len(s) % 4 == 0 and b64_pattern.match(s.strip()):
        try:
            decoded_bytes = base64.b64decode(s, validate=True)
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            return s
    return s


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

    def _json_schema(self) -> dict[str, Any]:
        turn_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["think", "issue_task", "answer"],
                    "description": "What to do this turn.",
                },
                "text_b64": {
                    "type": "string",
                    "description": (
                        "Base64 (UTF-8) encoded text.\n"
                        "- think: reasoning notes or plan\n"
                        "- issue_task: fully self-contained sub-task prompt\n"
                        "- answer: final answer to the original question"
                    ),
                },
            },
            "required": ["action", "text_b64"],
            "oneOf": [
                {"properties": {"action": {"const": "think"}}, "required": ["text_b64"]},
                {"properties": {"action": {"const": "issue_task"}}, "required": ["text_b64"]},
                {"properties": {"action": {"const": "answer"}}, "required": ["text_b64"]},
            ],
        }
        return {
            "name": "dac_turn",
            "strict": True,
            "schema": turn_schema,
        }

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

        json_schema_obj = self._json_schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema_obj,
        }
        # Some vLLM builds also look for this
        extra_body.setdefault("guided_json", json_schema_obj.get("schema", json_schema_obj))

        return await super()._call(messages=messages, **kwargs)

    def _parse_guided_turn(self, message: Message) -> dict[str, Any] | None:
        if message.get("role") != "assistant":
            return None
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            obj = json.loads(content)
            return obj if isinstance(obj, dict) else None
        except Exception as e:
            logger.warning(f"Failed to parse guided JSON content: {e}")
            return None

    def _decode_text_b64(self, obj: dict[str, Any]) -> str:
        raw = obj.get("text_b64")
        if isinstance(raw, str) and raw:
            return _maybe_decode_base64(raw).strip()
        return ""

    def parse_tasks(self, message: Message) -> list[UserMessage]:
        """Extract at most one sub-task from the guided turn."""
        obj = self._parse_guided_turn(message)
        if isinstance(obj, dict) and obj.get("action") == "issue_task":
            text = self._decode_text_b64(obj)
            if text:
                return [UserMessage(role="user", content=text)]
        # Fallback to base extraction (marker-based) if guided failed
        return super().parse_tasks(message)

    def parse_answer(self, message: Message) -> AssistantMessage:
        obj = self._parse_guided_turn(message)
        if isinstance(obj, dict) and obj.get("action") == "answer":
            text = self._decode_text_b64(obj)
            return AssistantMessage(role="assistant", content=text)
        return super().parse_answer(message)

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
            guided_obj = self._parse_guided_turn(assistant_msg) or {}
            action = guided_obj.get("action") if isinstance(guided_obj, dict) else None

            # Unknown/invalid action → stop to avoid loops
            if action not in {"think", "issue_task", "answer"}:
                logger.warning(f"Unexpected or missing action in guided output: {action!r}")
                break

            # If the model chose to answer, finalize and stop
            if action == "answer":
                final = self._decode_text_b64(guided_obj)
                self.trajectory.messages_and_choices.append(
                    AssistantMessage(role="assistant", content=f"{Markers.ANSWER_START} {final} {Markers.ANSWER_END}")
                )
                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))
                break

            # If the model chose to think, append the think block and continue
            if action == "think":
                think_text = self._decode_text_b64(guided_obj)
                think_block = f"{Markers.THINK_START} {think_text} {Markers.THINK_END}"
                self.trajectory.messages_and_choices.append(UserMessage(role="user", content=think_block))
                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))
                think_turns += 1
                if think_turns >= max_think_turns:
                    logger.warning("Max consecutive think turns reached; stopping to avoid infinite loop.")
                    break
                continue

            # action == "issue_task": try to extract exactly one task
            tasks_inputs = self.parse_tasks(assistant_msg)
            if not tasks_inputs:
                # No task provided; stop to avoid infinite loops.
                logger.warning("issue_task selected but no task text present; stopping.")
                break
            # Enforce a single task per turn
            task = tasks_inputs[0]

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

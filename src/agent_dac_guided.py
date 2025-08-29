from __future__ import annotations
from typing import Any, Iterable
from enum import Enum
from dataclasses import dataclass

import json_repair
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
    Pure schema helper: builds the JSON schema and parses a raw JSON string into an AgentTurn.

    Public API:
    - build() -> dict: JSON schema object for response_format
    - parse(content: str) -> AgentTurn: parse a raw assistant content string
    """

    def __init__(self, actions: Iterable[AgentAction]) -> None:
        self.actions = actions

    def build(self) -> dict[str, Any]:
        """Return the schema descriptor for response_format."""
        turn_schema = {
            "type": "object",
            "additionalProperties": False,
            "description": (
                "Return exactly two fields: 'action' and 'text'.\n"
                "- action: one of " + str([a.value for a in self.actions]) + " for this turn.\n"
                "- text: UTF-8 text. Always required, regardless of action."
            ),
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [a.value for a in self.actions],
                    "description": "What to do this turn.",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text content for ALL actions.\n"
                        "- think: reasoning notes or plan\n"
                        "- issue_task: fully self-contained sub-task prompt\n"
                        "- answer: final answer to the original question"
                    ),
                },
            },
            "required": ["action", "text"],
        }
        return {
            "name": "dac_turn",
            "description": "Schema for a single AgentDaC turn.",
            "strict": True,
            "schema": turn_schema,
        }

    def parse(self, content: str) -> AgentTurn:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Assistant content is empty.")
        try:
            obj = json_repair.loads(content)
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
            raise ValueError(
                f"Action {action.value!r} is not allowed for this turn. Allowed: {self.actions}"
            )

        text_val = obj.get("text")
        if not isinstance(text_val, str) or not text_val:
            raise ValueError("Field 'text' must be a non-empty string.")

        return AgentTurn(action=action, text=text_val.strip(), raw=obj)


class AgentGuidedNode(AgentNode):
    """
    Guided-decoding agent that avoids function/tool calls and uses a very simple
    JSON control schema per assistant turn. The schema supports three actions,
    with exactly one action per turn:

    - think: Spend the turn thinking. The model writes thoughts in `text`.
    - issue_task: Delegate exactly one sub-task. The sub-task prompt is in `text`.
    - answer: Provide the final answer in `text`.

    Design goals:
    - Only one field for content: `text` (UTF-8) to avoid ambiguity.
    - Only one task per turn is allowed by design.
    - A dedicated "think" action lets the model think before deciding to issue
    a task or finalize with an answer.
    """

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """Call the model with JSON-schema structured output enabled. Uses all actions by default."""
        # Ensure we are not mixing tool-calls with structured outputs
        kwargs = dict(kwargs)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        # For JSON-only outputs, no stop sequences are necessary
        kwargs.setdefault("stop", [])

        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)

        # Respect a preconfigured response_format; otherwise, set it up
        if "response_format" not in kwargs:
            schema = GuidedSchema([AgentAction.THINK, AgentAction.ISSUE_TASK, AgentAction.ANSWER])
            json_schema_obj = schema.build()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
            # Some vLLM builds also look for this
            extra_body.setdefault("guided_json", json_schema_obj.get("schema", json_schema_obj))
        else:
            # If provided, still try to pass through guided_json for broader compatibility
            rf = kwargs.get("response_format", {})
            js = rf.get("json_schema") if isinstance(rf, dict) else None
            if isinstance(js, dict):
                extra_body.setdefault("guided_json", js.get("schema", js))

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

        # Prevent unbounded thinking loops
        think_turns = 0
        max_think_turns = 8

        # Single schema/parser instance per chat loop
        schema = GuidedSchema([AgentAction.THINK, AgentAction.ISSUE_TASK, AgentAction.ANSWER])
        json_schema_obj = schema.build()
        # Ensure the model is guided by this schema on each call
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault(
            "response_format",
            {"type": "json_schema", "json_schema": json_schema_obj},
        )

        while True:
            # Model turn
            completion = await self._call(self.trajectory.messages(), **call_kwargs)
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
                think_turns += 1
                if think_turns >= max_think_turns:
                    logger.warning("Max consecutive think turns reached; stopping to avoid infinite loop.")
                    break
                continue

            elif turn.action == AgentAction.ISSUE_TASK:
                task_text = turn.text
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
                if isinstance(task_resp, dict):
                    content = task_resp.get("content") or ""
                elif isinstance(task_resp, str):
                    content = task_resp
                else:
                    content = str(task_resp)
                joined_message = UserMessage(
                    role="user",
                    content=f"{Markers.ANSWER_START} {content} {Markers.ANSWER_END}",
                )
                self.trajectory.messages_and_choices.append(joined_message)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                # Inform decomp state that we used one task this round
                self.decomp_config.update_round(num_tasks=1)

            else:
                pass  # TODO: handle unexpected action

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
            schema = GuidedSchema([AgentAction.ANSWER])
            turn = schema.parse(answer_content)
            return turn.text

        except Exception as e:
            logger.error(f"Error parsing final answer guided JSON: {e}")
            return answer_content.strip()

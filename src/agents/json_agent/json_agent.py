from __future__ import annotations
from enum import StrEnum
from typing import Any
from dataclasses import dataclass
import json_repair

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.agents.json_agent.actions import TurnAction
from src.aliases import Message, UserMessage
from src.inference import InferenceResponse
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "chats")


class JsonErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"


@dataclass
class AgentTurn:
    action: str | None
    text: str
    raw: dict[str, Any]


class GuidedJson:
    def __init__(self, *actions: str) -> None:
        self.actions = actions

    def build(self) -> dict[str, Any]:
        json_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": [a for a in self.actions]},
                "text": {"type": "string"},
            },
            "required": ["action", "text"],
        }

        return {
            "name": "assistant_turn",
            "description": "Json schema for a single assistant turn.",
            "strict": True,
            "schema": json_schema,
        }

    def parse(self, content: Any) -> AgentTurn:
        if not isinstance(content, str):
            raise ValueError("Content to parse must be a string.")

        json_obj = json_repair.loads(content, skip_json_loads=True)
        if not isinstance(json_obj, dict):
            logger.debug(f"Failed to parse JSON content: {content}")
            logger.debug(f"Parsed JSON object: {json_obj}")
            raise ValueError(f"Parsed content is not a dictionary, got {type(json_obj)}")

        action_val = json_obj["action"]
        if not isinstance(action_val, str):
            raise ValueError(f"Field 'action' must be a string, got {type(action_val)}.")

        text_val = json_obj["text"]
        if not isinstance(text_val, str):
            raise ValueError(f"Field 'text' must be a string, got {type(text_val)}.")

        action = TurnAction(action_val)
        text = text_val.strip()

        if action not in self.actions:
            raise ValueError(f"Action {action} is not allowed for this turn. Allowed: {self.actions}")

        return AgentTurn(action=action, text=text, raw=json_obj)


class JsonAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return JsonErrors

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

    def _create_schema(self) -> GuidedJson:
        """
        Rules for allowed actions:
        1) If at a leaf (depth >= max_depth): cannot ISSUE_TASK.
        2) If rounds remain (total_rounds < max_rounds): may THINK.
        3) If no rounds remain: must ANSWER.
        4) If tasks exhausted (total_tasks >= max_tasks): cannot ISSUE_TASK.
        5) ANSWER is always allowed.
        """
        DC = self.decomp_config
        allowed = [TurnAction.ANSWER]
        if DC.has_rounds():
            allowed.append(TurnAction.THINK)
            if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
                allowed.append(TurnAction.ISSUE_TASK)

        return GuidedJson(*allowed)

    async def call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        schema: GuidedJson = kwargs.pop("schema")
        schema_descriptor = schema.build()
        kwargs = self.client.update_kwargs(kwargs, json_schema=schema_descriptor, include_stop_str_in_output=True)
        return await super().call(messages, **kwargs)

    def create_subagent(self) -> BaseAgent:
        agent = JsonAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  
            verbose=self.verbose,
        )
        
        if self.additional_histories:
            agent.trajectory.histories = self.trajectory.histories
            
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
            schema = self._create_schema()

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self.call(self.trajectory.messages(), schema=schema, **kwargs)
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                self.trajectory.error(kind=JsonErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Extract raw content and parse it
                assistant_msg = self.trajectory.messages()[-1]
                turn = schema.parse(assistant_msg.get("content"))
            except Exception as e:
                logger.warning(f"Failed to parse model output: {e}")
                self.trajectory.error(kind=JsonErrors.PARSE_ERROR, message=str(e))
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
                sub_agent = self.create_subagent()
                task = UserMessage(role="user", content=turn.text)

                task_answer = await sub_agent.answer(task, **kwargs)
                if task_answer is None:
                    task_answer = "[error] sub-agent failed to produce an answer."

                task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
                self.append_message(task_response)

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
                raise ValueError(f"Unrecognized action: {turn.action}")

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            return None

        try:
            schema = GuidedJson(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text

        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return None

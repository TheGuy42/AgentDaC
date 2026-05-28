from __future__ import annotations
from typing import Any
from dataclasses import dataclass

import json_repair
from openai.types.chat import ChatCompletion
from art.trajectories import Trajectory, History

from src.agents.base import BaseAgent
from src.agents.json_agent.actions import TurnAction
from src.openai_types import Message, UserMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger


logger = create_logger(__name__)


@dataclass
class AgentTurn:
    action: TurnAction
    text: str
    raw: dict[str, Any]


class GuidedJson:
    def __init__(self, *actions: TurnAction) -> None:
        self.actions = actions

    def build(self) -> dict[str, Any]:
        json_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": [a.value for a in self.actions]},
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
    def _create_schema(self) -> GuidedJson:
        """
        Rules for allowed actions:
        1) If at a leaf (depth >= max_depth): cannot ISSUE_TASK.
        2) If rounds remain (total_rounds < max_rounds): may THINK.
        3) If no rounds remain: must ANSWER.
        4) If tasks exhausted (total_tasks >= max_tasks): cannot ISSUE_TASK.
        5) ANSWER is always allowed.
        """

        dc = self.decomp_config
        is_leaf = self.current_depth >= dc.max_depth
        has_rounds = dc.total_rounds < dc.max_rounds
        tasks_available = dc.total_tasks < dc.max_tasks

        allowed = [TurnAction.ANSWER]
        if has_rounds:
            allowed.append(TurnAction.THINK)
            if (not is_leaf) and tasks_available:
                allowed.append(TurnAction.ISSUE_TASK)

        return GuidedJson(*allowed)

    async def call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        schema: GuidedJson = kwargs.pop("schema")
        schema_descriptor = schema.build()

        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema_descriptor}

        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    def _create_subagent(self) -> BaseAgent:
        return JsonAgent(
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
            schema = self._create_schema()
            completion = await self.call(self.trajectory.messages(), schema=schema, **kwargs)
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
            try:
                assistant_msg = self.trajectory.messages()[-1]
                turn = schema.parse(assistant_msg.get("content"))
            except Exception as e:
                logger.warning(f"Failed to parse model output: {e}")
                turn = AgentTurn(action=TurnAction.ERROR, text="", raw={})
                break

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
            schema = GuidedJson(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content

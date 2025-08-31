from __future__ import annotations
from typing import Any
from enum import Enum
from dataclasses import dataclass

import json_repair
from openai.types.chat import ChatCompletion
from art import Trajectory

from src.agents.base import BaseAgent
from src.openai_types import Message, UserMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger


logger = create_logger(__name__)


class TurnAction(str, Enum):
    Think = "think"
    Issue_Task = "issue_task"
    Answer = "answer"


@dataclass
class AgentTurn:
    action: TurnAction
    text: str
    raw: dict[str, Any]


# TODO: guided decoding only works when setting: internal_config["engine_args"]["num_scheduler_steps"] = 1
class GuidedSchema:
    def __init__(self, *actions: TurnAction) -> None:
        self.actions = actions

    # def build(self) -> dict[str, Any]:
    #     """Return the schema descriptor for response_format."""

    #     # TODO: make sure the comments are descriptive enough for the model to understand
    #     # TODO: the comments themselves are decorative only, they do not matter at all. The only thing that matters is the actual schema
    #     # We need to supply the schema in the message to the model in the system prompt with the actual documentation.
    #     turn_schema = {
    #         "type": "object",
    #         "additionalProperties": False,
    #         "description": (
    #             "Return exactly two fields: 'action' and 'text'.\n"
    #             "- action: one of " + str([a.value for a in self.actions]) + " for this turn.\n"
    #             "- text: UTF-8 text. Always required, regardless of action."
    #         ),
    #         "properties": {
    #             "action": {
    #                 "type": "string",
    #                 "enum": [a.value for a in self.actions],
    #                 "description": "What to do this turn.",
    #             },
    #             "text": {
    #                 "type": "string",
    #                 "description": (
    #                     "Text content for the chosen action.\n"
    #                     + "".join(
    #                         [
    #                             f"- {TurnAction.Think.value}: reasoning notes or plan" + "\n"
    #                             if TurnAction.Think in self.actions
    #                             else "",
    #                             f"- {TurnAction.Issue_Task.value}: fully self-contained sub-task prompt" + "\n"
    #                             if TurnAction.Issue_Task in self.actions
    #                             else "",
    #                             f"- {TurnAction.Answer.value}: final answer to the original question" + "\n"
    #                             if TurnAction.Answer in self.actions
    #                             else "",
    #                         ]
    #                     )
    #                 ),
    #             },
    #         },
    #         "required": ["action", "text"],
    #     }
    #     return {
    #         "name": "assistant_turn",
    #         "description": (
    #             "Json schema for a single assistant turn. "
    #             "The assistant must choose one of the allowed actions and provide the corresponding text."
    #         ),
    #         "strict": True,
    #         "schema": turn_schema,
    #     }

    def build(self) -> dict[str, Any]:
        turn_schema = {
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
            "schema": turn_schema,
        }

    def parse(self, content: Any) -> AgentTurn:
        if not isinstance(content, str):
            raise ValueError("Content to parse must be a string.")

        json_obj = json_repair.loads(content)
        if not isinstance(json_obj, dict):
            raise ValueError(f"Parsed content is not a dictionary, got {type(json_obj)}.")

        action_val = json_obj["action"]
        if not isinstance(action_val, str):
            raise ValueError("Field 'action' must be a string.")

        text_val = json_obj["text"]
        if not isinstance(text_val, str):
            raise ValueError("Field 'text' must be a string.")

        action = TurnAction(action_val)
        text = text_val.strip()

        if action not in self.actions:
            raise ValueError(f"Action {action.value!r} is not allowed for this turn. Allowed: {self.actions}")

        return AgentTurn(action=action, text=text, raw=json_obj)


class GuidedAgent(BaseAgent):
    def _create_schema(self) -> GuidedSchema:
        if self.decomp_config.should_stop(self.current_depth):
            schema = GuidedSchema(TurnAction.Answer)
        else:
            schema = GuidedSchema(TurnAction.Think, TurnAction.Issue_Task, TurnAction.Answer)
        return schema

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        schema: GuidedSchema = kwargs.pop("schema")
        schema_descriptor = schema.build()

        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema_descriptor}
        # extra_body["guided_json"] = schema_descriptor["schema"] # NOTE: internally vLLM passes response_format to guided_json

        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    def create_sub_agent(self) -> BaseAgent:
        return GuidedAgent(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
        )

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')!r}.")

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
            completion = await self._call(self.trajectory.messages(), schema=schema, **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage is not None:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract raw content and parse it
            try:
                assistant_msg = self.trajectory.messages()[-1]
                turn = schema.parse(assistant_msg.get("content"))
            except Exception as e:
                logger.warning(f"Failed to parse model output: {e}")
                break

            # Check if we should stop
            if self.decomp_config.should_stop(self.current_depth):
                break

            # Finish if the model chose to answer
            if turn.action == TurnAction.Answer:
                break

            # If the model chose to think, continue
            elif turn.action == TurnAction.Think:
                self.metrics["direct_thinks"] += 1
                self.metrics["total_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)

            # Issue a task and get the answer from a sub-agent
            elif turn.action == TurnAction.Issue_Task:
                sub_agent = self.create_sub_agent()
                task = UserMessage(role="user", content=turn.text)
                task_answer = await sub_agent.answer(task, verbose, **kwargs)

                # Update metrics from sub-agent
                self.metrics["direct_tasks"] += 1
                self.metrics["total_tasks"] += 1 + sub_agent.metrics["total_tasks"]
                self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                self.metrics["total_thinks"] += sub_agent.metrics["total_thinks"]
                self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

                task_response = UserMessage(role="user", content=f"{Markers.ANS_START} {task_answer} {Markers.ANS_END}")
                self.trajectory.messages_and_choices.append(task_response)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                self.decomp_config.update_round(num_tasks=1)

            else:
                raise ValueError(f"Unhandled action: {turn.action}")

        # Update final stats
        self.metrics["response_completed"] = choice.finish_reason != "length"
        self.trajectory.finish()

        return self.trajectory

    def parse_answer(self, message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        try:
            schema = GuidedSchema(TurnAction.Answer)
            turn = schema.parse(content)
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content

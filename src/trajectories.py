from __future__ import annotations
from datetime import datetime
from typing import Any, cast
import dataclasses

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from src.aliases import Message, Choice


@dataclasses.dataclass
class History:
    messages_and_choices: list[Message | Choice]
    tools: list[ChatCompletionToolParam] | None = None

    def messages(self) -> list[Message]:
        return get_messages(self.messages_and_choices)


@dataclasses.dataclass
class Trajectory:
    messages_and_choices: list[Message | Choice]
    tools: list[ChatCompletionToolParam] | None = None
    additional_histories: list[History] = dataclasses.field(default_factory=list)
    reward: float = 0.0
    metrics: dict[str, float | int | bool] = dataclasses.field(default_factory=dict)
    metadata: dict[str, float | int | str | bool | None] = dataclasses.field(default_factory=dict)
    logs: list[str] = dataclasses.field(default_factory=list)
    start_time: datetime = dataclasses.field(default_factory=datetime.now)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def finish(self) -> Trajectory:
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    def messages(self) -> list[Message]:
        return get_messages(self.messages_and_choices)

    def for_logging(self) -> dict[str, Any]:
        loggable_dict: dict[str, Any] = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": [],
            "tools": self.tools,
            "logs": self.logs,
        }
        for message_or_choice in self.messages_and_choices:
            if isinstance(message_or_choice, Choice):
                trainable = True
                message: dict[str, Any] = message_or_choice.message.to_dict()
            else:
                trainable = False
                message = cast(dict[str, Any], message_or_choice)
            loggable_dict["messages"].append({**message, "trainable": trainable})
        return loggable_dict


def get_messages(messages_and_choices: list[Message | Choice]) -> list[Message]:
    messages: list[Message] = []
    for message_or_choice in messages_and_choices:
        if isinstance(message_or_choice, Choice):
            content = message_or_choice.message.content or ""
            tool_calls = message_or_choice.message.tool_calls or []
            assistant_message: Message = cast(
                Message,
                {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": [tool_call.model_dump(mode="json") for tool_call in tool_calls]} if tool_calls else {}),
                },
            )
            messages.append(assistant_message)
        else:
            # Ensure content is always a string for tokenizer chat templates
            msg = dict(message_or_choice)
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)  # type: ignore[arg-type]
    return messages

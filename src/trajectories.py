from __future__ import annotations
from datetime import datetime
from typing import Any, cast
import dataclasses

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from art.trajectories import Trajectory as ArtTrajectory, History as ArtHistory
from src.aliases import Message, Choice


@dataclasses.dataclass
class History:
    messages_and_choices: list[Message | Choice]
    tools: list[ChatCompletionToolParam] | None = None

    def messages(self) -> list[Message]:
        return get_messages(self.messages_and_choices)

    def to_art(self) -> ArtHistory:
        """
        Creates a shallow copy to an `art.trajectories.History` object, used for compatibility.
        """
        return ArtHistory(messages_and_choices=self.messages_and_choices, tools=self.tools)


@dataclasses.dataclass
class Trajectory:
    messages_and_choices: list[Message | Choice]
    tools: list[ChatCompletionToolParam] | None = None
    additional_histories: list[History] = []
    reward: float = 0.0
    metrics: dict[str, float | int | bool] = {}
    metadata: dict[str, float | int | str | bool | None] = {}
    logs: list[str] = []
    start_time: datetime = dataclasses.field(default_factory=datetime.now)

    def to_art(self) -> ArtTrajectory:
        """
        Creates a shallow copy to an `art.trajectories.Trajectory` object, used for compatibility.
        """
        return ArtTrajectory(
            messages_and_choices=self.messages_and_choices,
            tools=self.tools,
            additional_histories=[h.to_art() for h in self.additional_histories],
            reward=self.reward,
            metrics=self.metrics,
            metadata=self.metadata,
            logs=self.logs,
            start_time=self.start_time,
        )

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

from __future__ import annotations
from datetime import datetime
from typing import Any, cast
import dataclasses

from src.aliases import Message, ToolSchema
from src.inference import InferenceResponse


@dataclasses.dataclass(frozen=True)
class TrajectoryError:
    kind: str
    message: str
    step: int

    def __str__(self) -> str:
        return f"[{self.step}] {self.kind}: {self.message}"


@dataclasses.dataclass
class Trajectory:
    messages_and_responses: list[Message | InferenceResponse]
    tools: list[ToolSchema] | None = None
    histories: list[Trajectory] = dataclasses.field(default_factory=list)
    reward: float = 0.0
    metrics: dict[str, float | int | bool] = dataclasses.field(default_factory=dict)
    metadata: dict[str, float | int | str | bool | None] = dataclasses.field(default_factory=dict)
    logs: list[str] = dataclasses.field(default_factory=list)
    errors: list[TrajectoryError] = dataclasses.field(default_factory=list)
    start_time: datetime = dataclasses.field(default_factory=datetime.now)

    def log(self, message: str) -> None:
        message = f"[{len(self.messages_and_responses)}] {message.strip()}"
        self.logs.append(message)

    def error(self, kind: str, message: str) -> None:
        error = TrajectoryError(kind=kind, message=message, step=len(self.messages_and_responses))
        self.errors.append(error)

    def finish(self) -> Trajectory:
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    def messages(self) -> list[Message]:
        return get_messages(self.messages_and_responses)

    def for_logging(self) -> dict[str, Any]:
        result_dict: dict[str, Any] = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": get_messages(self.messages_and_responses),
            "histories": [history.for_logging() for history in self.histories],
            "tools": self.tools,
            "logs": self.logs,
            "errors": [str(error) for error in self.errors],
        }

        # add field "trainable" to each message dict
        for msg, item in zip(result_dict["messages"], self.messages_and_responses):
            msg["trainable"] = isinstance(item, InferenceResponse)

        return result_dict


def get_messages(messages_and_responses: list[Message | InferenceResponse]) -> list[Message]:
    messages: list[Message] = []
    for message_or_response in messages_and_responses:
        if isinstance(message_or_response, InferenceResponse):
            content = message_or_response.content or ""
            tool_calls = message_or_response.tool_calls or []
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
            msg = dict(message_or_response)
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)  # type: ignore[arg-type]
    return messages

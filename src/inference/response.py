from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from src.aliases import Response, Choice


class InferenceResponse(ABC):
    """
    Backend-agnostic view of a single model response.
    """

    @property
    @abstractmethod
    def content(self) -> str | None:
        """The assistant message text."""

    @property
    @abstractmethod
    def tool_calls(self) -> list[Any] | None:
        """Assistant tool calls, or ``None``/empty when there are none."""

    @property
    @abstractmethod
    def finish_reason(self) -> str | None:
        """Why generation stopped (e.g. ``"stop"`` / ``"length"``)."""

    @property
    @abstractmethod
    def total_tokens(self) -> int | None:
        """Total prompt+completion tokens, or ``None`` if unavailable."""


class OAIResponse(InferenceResponse):
    def __init__(self, openai_response: Response) -> None:
        self.openai_response = openai_response

    @property
    def choice(self) -> Choice:
        return self.openai_response.choices[0]

    @property
    def content(self) -> str | None:
        return self.choice.message.content

    @property
    def tool_calls(self) -> list[Any] | None:
        return self.choice.message.tool_calls

    @property
    def finish_reason(self) -> str | None:
        return self.choice.finish_reason

    @property
    def total_tokens(self) -> int | None:
        usage = self.openai_response.usage
        return usage.total_tokens if usage is not None else None

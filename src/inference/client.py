from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from src.aliases import Message


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
        """Assistant tool calls, or `None`/empty when there are none."""

    @property
    @abstractmethod
    def finish_reason(self) -> str | None:
        """Why generation stopped (e.g. `"stop"` / `"length"`)."""

    @property
    @abstractmethod
    def total_tokens(self) -> int | None:
        """Total prompt+completion tokens, or `None` if unavailable."""


class InferenceClient(ABC):
    @abstractmethod
    def update_kwargs(
        self,
        kwargs: dict[str, Any],
        json_schema: dict | None = None,
        regex_schema: str | None = None,
        include_stop_str_in_output: bool | None = None,
    ) -> dict[str, Any]:
        """Updates kwargs with backend-specific parameters.

        Args:
            kwargs (dict[str, Any]): The original kwargs to update.
            json_schema (dict | None): Optional guided JSON schema for the assistant's response.
            regex_schema (str | None): Optional guided regex schema for the assistant's response.
            include_stop_str_in_output (bool | None): Whether to include the stop string in the output.

        Returns:
            dict[str, Any]: The updated kwargs with backend-specific parameters.
        """
        raise NotImplementedError

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> InferenceResponse:
        """Generate a single assistant response for `messages`.

        Args:
            messages (list[Message]): The full conversation so far (OpenAI message dicts).
            **kwargs: Keyword arguments forwarded verbatim to the inference backend's
                sampling parameters. Callers are expected to pass backend-compatible kwargs.

        Returns:
            InferenceResponse: a backend-agnostic response wrapping the raw backend output.
        """
        raise NotImplementedError

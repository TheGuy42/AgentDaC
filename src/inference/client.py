from __future__ import annotations
from abc import ABC, abstractmethod

from src.aliases import Message
from src.inference.response import InferenceResponse

class InferenceClient(ABC):

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

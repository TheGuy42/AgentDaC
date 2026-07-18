from __future__ import annotations
from enum import StrEnum

from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.configs import DecompConfig
from src.aliases import Message
from src.inference import InferenceResponse
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct",)
METRIC_COUNTERS = ("calls", "chats")


class DummyErrors(StrEnum):
    CLIENT_ERROR = "client_error"


class DummyAgent(BaseAgent):
    """
    A simple model wrapper that satisfies the BaseAgent interface.
    Does not create any sub-agents or any custom logic.
    """

    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return DummyErrors

    def __init__(
        self,
        client,
        prompt_config,
        verbose: bool = False,
    ):

        decomp_config = DecompConfig(
            max_depth=0,
            max_rounds=1,
            max_tasks=0,
        )

        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=0,
            additional_histories=False,
            verbose=verbose,
        )

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics["direct_tokens"] = 0

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super()._call(messages, **kwargs)

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
            self.metrics[f"{prefix}_calls"] += 1

        try:
            # Model turn
            completion = await self._call(self.trajectory.messages(), **kwargs)
        except Exception as e:
            logger.error(f"Error during model call: {e}")
            self.trajectory.error(kind=DummyErrors.CLIENT_ERROR, message=str(e))
            return self.trajectory.finish()

        self.append_message(completion)
        if completion.total_tokens is not None:
            self.metrics["direct_tokens"] = completion.total_tokens

        self.decomp_config.update_round(num_tasks=0)
        return self.trajectory.finish()

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            return None

        return content

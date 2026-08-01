from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque

import torch

from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class _ThinkingBudget:
    def __init__(
        self,
        budget: int,
        start_ids: list[int],
        end_ids: list[int],
    ) -> None:
        self.budget = budget
        self.start_ids = tuple(start_ids)
        self.end_ids = tuple(end_ids)

        self.tail: deque[int] = deque(maxlen=max(len(self.start_ids), len(self.end_ids)))

        self.initialized = False
        self.seen_output = 0
        self.in_thinking = False
        self.thinking_tokens = 0

    def _endswith(self, token_ids: tuple[int, ...]) -> bool:
        n = len(token_ids)
        return n <= len(self.tail) and tuple(self.tail)[-n:] == token_ids

    def _consume(self, token_id: int) -> None:
        self.tail.append(token_id)

        if self.in_thinking:
            self.thinking_tokens += 1

            if self._endswith(self.end_ids):
                self.in_thinking = False
                self.thinking_tokens = 0

        elif self._endswith(self.start_ids):
            self.in_thinking = True
            self.thinking_tokens = 0

    def _next_end_token(self) -> int:
        """Continue an already-partial end sequence when possible."""
        tail = tuple(self.tail)

        max_prefix = min(
            len(tail),
            len(self.end_ids) - 1,
        )

        for prefix_len in range(max_prefix, 0, -1):
            if tail[-prefix_len:] == self.end_ids[:prefix_len]:
                return self.end_ids[prefix_len]

        return self.end_ids[0]

    def __call__(
        self,
        prompt_ids: list[int] | None,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.initialized:
            for token_id in prompt_ids or ():
                self._consume(token_id)

            self.initialized = True

        if len(output_ids) < self.seen_output:
            raise RuntimeError("output_ids unexpectedly shrank; this processor assumes non-speculative append-only decoding")

        for token_id in output_ids[self.seen_output :]:
            self._consume(token_id)

        self.seen_output = len(output_ids)

        if self.in_thinking and self.thinking_tokens >= self.budget:
            forced_token_id = self._next_end_token()

            if not 0 <= forced_token_id < logits.shape[-1]:
                raise RuntimeError(f"Reasoning end token {forced_token_id} is outside the vocabulary of size {logits.shape[-1]}")

            logits.fill_(float("-inf"))
            logits[forced_token_id] = 0.0

        return logits


class ThinkingBudgetLogitsProcessor(AdapterLogitsProcessor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(
            vllm_config,
            device,
            is_pin_memory,
        )

        model_config = vllm_config.model_config
        if model_config is None:
            raise ValueError("ThinkingBudgetLogitsProcessor requires a generation model")

        tokenizer = cached_tokenizer_from_config(model_config)
        if tokenizer is None:
            raise ValueError("ThinkingBudgetLogitsProcessor requires tokenizer initialization")

        self.tokenizer = tokenizer

        # Avoid repeatedly tokenizing the same delimiters.
        self._token_ids_cache: dict[str, tuple[int, ...]] = {}

    def _encode(self, text: str) -> tuple[int, ...]:
        cached = self._token_ids_cache.get(text)
        if cached is not None:
            return cached

        token_ids = tuple(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

        if not token_ids:
            raise ValueError(f"Reasoning delimiter {text!r} tokenized to no tokens")

        self._token_ids_cache[text] = token_ids
        return token_ids

    @classmethod
    def validate_params(
        cls,
        sampling_params: SamplingParams,
    ) -> None:
        args = sampling_params.extra_args or {}

        budget = args.get("thinking_token_budget")
        if budget is None:
            return

        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("thinking_token_budget must be a non-negative integer")

        start_str = args.get("thinking_start_str")
        if not isinstance(start_str, str) or not start_str:
            raise ValueError("thinking_start_str must be a non-empty string")

        end_str = args.get("thinking_end_str")
        if not isinstance(end_str, str) or not end_str:
            raise ValueError("thinking_end_str must be a non-empty string")

        if start_str == end_str:
            raise ValueError("thinking_start_str and thinking_end_str must differ")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        args = params.extra_args or {}

        budget = args.get("thinking_token_budget")
        if budget is None:
            return None

        self.validate_params(params)

        start_ids = self._encode(args["thinking_start_str"])
        end_ids = self._encode(args["thinking_end_str"])

        return _ThinkingBudget(
            budget=budget,
            start_ids=list(start_ids),
            end_ids=list(end_ids),
        )

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from transformers import PreTrainedTokenizerBase

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

from src.aliases import Message, Response
from src.inference import InferenceClient, InferenceResponse, OAIResponse


class VerlResponse(OAIResponse):
    def __init__(self, openai_response: Response, verl_response: TokenOutput) -> None:
        super().__init__(openai_response)
        self.verl_response = verl_response


class VerlClient(InferenceClient):
    """
    Text-only client over VERL's token-in/token-out rollout API.
    This client is non-openai based, it is rather based on plain `.generate()` calls.
    """

    def __init__(self, verl_agent: AgentLoopBase, client_id: str | None = None) -> None:
        self.verl_agent = verl_agent
        self.client_id = client_id or uuid4().hex
        self.model_name = self._get_model_name()

    def _get_model_name(self) -> str:
        model_path = self.verl_agent.config.actor_rollout_ref.model.path
        return "/".join(model_path.split("/")[-2:])

    async def chat(self, messages: list[Message], **kwargs: Any) -> InferenceResponse:
        prompt_ids = await self._tokenize(messages)

        output = await self.verl_agent.server_manager.generate(
            request_id=self.client_id,
            prompt_ids=prompt_ids,
            sampling_params=dict(kwargs),
        )

        return await self._build_response(
            prompt_ids=prompt_ids,
            output=output,
            kwargs=kwargs,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.verl_agent.tokenizer  # type: ignore[assignment]

    async def _tokenize(self, messages: list[Message]) -> list[int]:
        encoded = await self.verl_agent.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                list(messages),  # type: ignore[arg-type]
                add_generation_prompt=True,
                tokenize=True,
                **dict(self.verl_agent.apply_chat_template_kwargs or {}),
            ),
        )
        return normalize_token_ids(encoded)

    async def _build_response(
        self,
        *,
        prompt_ids: list[int],
        output: TokenOutput,
        kwargs: dict[str, Any],
    ) -> InferenceResponse:
        content = await self.verl_agent.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                output.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )

        # Minimal ChatCompletion: only the fields we actually use, plus the token ids that
        # `convert` reads (response ids on the choice, prompt ids on the response).
        completion = Response.model_validate(
            {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": self._finish_reason(prompt_ids, output, kwargs),
                        "token_ids": output.token_ids,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(output.token_ids),
                    "total_tokens": len(prompt_ids) + len(output.token_ids),
                },
                "prompt_token_ids": list(prompt_ids),
            }
        )

        return VerlResponse(completion, output)

    def _finish_reason(self, prompt_ids: list[int], output: TokenOutput, kwargs: dict[str, Any]) -> str:
        """Best-effort finish reason inference.

        VERL collapses vLLM finish_reason in {"stop", "length"} into stop_reason="completed",
        so exact recovery is impossible. This infers "length" only when the generated output
        reaches VERL's effective max-token cap.
        """

        if output.stop_reason in {"abort", "aborted"}:
            return "aborted"

        cfg = self.verl_agent.rollout_config

        # If explicitly passed use max_tokens or max_new_tokens
        if kwargs.get("max_tokens") is not None:
            requested_max_tokens = int(kwargs["max_tokens"])
        elif kwargs.get("max_new_tokens") is not None:
            requested_max_tokens = int(kwargs["max_new_tokens"])

        else:
            # Otherwise, infer from the prompt and response lengths in the rollout config
            requested_max_tokens = min(
                cfg.response_length,
                cfg.prompt_length + cfg.response_length - len(prompt_ids),
            )

        max_possible_tokens = cfg.max_model_len - len(prompt_ids)
        effective_max_tokens = max(1, min(requested_max_tokens, max_possible_tokens))

        if len(output.token_ids) < effective_max_tokens:
            return "stop"

        return "length"

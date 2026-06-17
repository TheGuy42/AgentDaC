"""Builders for in-memory test objects (no server / GPU)."""

from __future__ import annotations

from datetime import datetime

import agentlightning as agl
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from src.aliases import Response, UserMessage
from src.trajectory import Trajectory


def make_response(
    *,
    response_id: str,
    content: str,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
) -> Response:
    """Build a ``ChatCompletion`` shaped like AGL's patched vLLM response.

    Mirrors the patched-object extraction path in ``convert._extract_attributes``:
    ``resp.prompt_token_ids`` is a flat list, ``resp.response_token_ids`` is one list per choice.
    """
    resp = ChatCompletion(
        id=response_id,
        object="chat.completion",
        created=0,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )
    # AGL patches these onto the response object (ChatCompletionResponsePatched).
    resp.prompt_token_ids = prompt_token_ids
    resp.response_token_ids = [response_token_ids]
    return resp


def make_trajectory(responses: list[Response], *, reward: float, metrics: dict) -> Trajectory:
    items: list = [UserMessage(role="user", content="prompt")]
    items.extend(responses)
    return Trajectory(
        messages_and_responses=items,
        reward=reward,
        metrics=dict(metrics),
        start_time=datetime(2024, 1, 1, 0, 0, 0),
    )


def make_rollout(*, rollout_id: str = "ro-test", attempt_id: str = "at-test") -> agl.AttemptedRollout:
    return agl.AttemptedRollout(
        rollout_id=rollout_id,
        input={},
        start_time=0.0,
        mode="train",
        attempt=agl.Attempt(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=1,
            start_time=0.0,
        ),
    )

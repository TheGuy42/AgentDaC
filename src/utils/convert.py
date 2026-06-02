from __future__ import annotations
from typing import Any

import agentlightning as agl
from agentlightning import Span, TraceStatus

from src.aliases import Response
from src.trajectory import History, Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


def _extract_attributes(resp: Response) -> dict[str, Any]:

    response_token_ids: list[int]
    prompt_token_ids: list[int]
    
    if hasattr(resp.choices[0], "provider_specific_fields") and ("token_ids" in resp.choices[0].provider_specific_fields):
        # When manually upgrading to vllm==0.11.0 and verl==0.6.1 (latest stable versions)
        # then AGL patch stops working and the token_ids are placed here
        response_token_ids = resp.choices[0].provider_specific_fields["token_ids"] # type: ignore[attr-defined]

    elif hasattr(resp, "response_token_ids"):
        # This attr appears in the patched response object by AGL
        # See `agentlightning.instrumentation.vllm.ChatCompletionResponsePatched`
        response_token_ids = getattr(resp, "response_token_ids")[0]
    
    elif hasattr(resp.choices[0], "token_ids"):
        # This is the original response object structure from OpenAI
        response_token_ids = getattr(resp.choices[0], "token_ids")
    
    else:
        logger.error(f"Response object missing expected token id attributes: {resp.model_dump()}")
        raise ValueError("Unable to extract response token ids from response object")

    if hasattr(resp, "prompt_token_ids"):
        # This attr appears in the original and the patched response object by AGL
        # See `agentlightning.instrumentation.vllm.ChatCompletionResponsePatched`
        prompt_token_ids = getattr(resp, "prompt_token_ids")
    
    else:
        logger.error(f"Response object missing expected token id attributes: {resp.model_dump()}")
        raise ValueError("Unable to extract prompt token ids from response object")

    return {
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response_token_ids,
        "gen_ai.response.id": resp.id,
    }


def convert_trajectory(trajectory: Trajectory, rollout: agl.AttemptedRollout) -> list[Span]:

    spans: list[Span] = []
    base_time = trajectory.start_time.timestamp()
    responses = [r for r in trajectory.messages_and_responses if isinstance(r, Response)]

    for i, resp in enumerate(responses):
        core = agl.SpanCoreFields(
            name="openai.chat.completion",
            status=TraceStatus(status_code="OK"),
            attributes=_extract_attributes(resp),
            start_time=(base_time + i),
            end_time=(base_time + i + 0.5),
        )

        span = Span.from_core_fields(
            core=core,
            rollout_id=rollout.rollout_id,
            attempt_id=rollout.attempt.attempt_id,
            sequence_id=i + 1,
        )

        spans.append(span)

    if not spans:
        logger.warning("Trajectory produced no LLM-call spans; the reward span will be orphaned.")

    reward_core = agl.emit_reward(trajectory.reward, propagate=False)
    reward_core.start_time = base_time + len(responses) + 1
    reward_core.end_time = reward_core.start_time + 0.5

    reward_span = Span.from_core_fields(
        core=reward_core,
        rollout_id=rollout.rollout_id,
        attempt_id=rollout.attempt.attempt_id,
        sequence_id=len(spans) + 1,
    )

    spans.append(reward_span)

    return spans

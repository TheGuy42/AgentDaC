from __future__ import annotations
from typing import Any

import agentlightning as agl
from agentlightning import Span, TraceStatus, Attributes


from src.aliases import Response
from src.trajectory import Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


def _extract_attributes(resp: Response) -> dict[str, Any]:

    response_token_ids: list[int]
    prompt_token_ids: list[int]

    if hasattr(resp.choices[0], "provider_specific_fields") and ("token_ids" in resp.choices[0].provider_specific_fields):  # type: ignore[attr-defined]
        # When manually upgrading to vllm==0.11.0 and verl==0.6.1 (latest stable versions)
        # then AGL patch stops working and the token_ids are placed here
        response_token_ids = resp.choices[0].provider_specific_fields["token_ids"]  # type: ignore[attr-defined]

    elif hasattr(resp, "response_token_ids"):
        # This attr appears in the patched response object by AGL
        # See `agentlightning.instrumentation.vllm.ChatCompletionResponsePatched`
        response_token_ids = getattr(resp, "response_token_ids")[0]

    elif hasattr(resp.choices[0], "token_ids"):
        # This is the original object structure from OpenAI
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
    """
    Converts native Trajectory to a list of Spans for AGL training.

    - *Note:* This function is explicitly designed to work with `custom.adapter.VerlAdapter` adapter, and the emitted spans are structured accordingly.
        The below implementation is not designed to work with other adapters and may fail or produce incorrect results if used with a different adapter.
    """
    if len(trajectory.histories) > 0:
        logger.warning("Trajectory has additional histories; They are not yet supported and will be ignored in the span conversion.")
        logger.info("To convert and train on additional histories, please pass them explicitly.")

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
        logger.debug("Trajectory produced no LLM-call spans; returning empty span list.")
        return []

    metrics = {"custom_metrics": trajectory.metrics.copy()}
    reward_core = agl.emit_reward(trajectory.reward, attributes=metrics, propagate=False)
    reward_core.start_time = base_time + len(spans) + 1
    reward_core.end_time = reward_core.start_time + 0.5

    reward_span = Span.from_core_fields(
        core=reward_core,
        rollout_id=rollout.rollout_id,
        attempt_id=rollout.attempt.attempt_id,
        sequence_id=len(spans) + 1,
    )

    spans.append(reward_span)

    return spans

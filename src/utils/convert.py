from __future__ import annotations
from typing import Any

import agentlightning as agl
from agentlightning import Span, TraceStatus

from src.aliases import Response
from src.trajectory import History, Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


def convert_trajectory(trajectory: Trajectory, rollout: agl.AttemptedRollout) -> list[Span]:

    spans: list[Span] = []
    base_time = trajectory.start_time.timestamp()
    responses = [r for r in trajectory.messages_and_responses if isinstance(r, Response)]

    for i, resp in enumerate(responses):
        
        try:
        
            attributes: dict[str, Any] = {
                "prompt_token_ids": resp.prompt_token_ids,  # type: ignore[attr-defined]
                "response_token_ids": resp.choices[0].token_ids,  # type: ignore[attr-defined]
                "gen_ai.response.id": resp.id,  # Helps the adapter dedup repeated spans for the same response.
            }
            
        except (Exception, BaseException) as e:
            raise Exception(
                f"Error for response index {i}:"
                f"Warning: Response object is missing expected token ID attributes: {e}"
                f"Available attributes on response: {resp.model_extra.keys() if resp.model_extra is not None else []}"
                f"Available attributes on choices[0]: {resp.choices[0].model_extra.keys() if resp.choices and resp.choices[0].model_extra is not None else []}"
            )

        core = agl.SpanCoreFields(
            name="openai.chat.completion",
            status=TraceStatus(status_code="OK"),
            attributes=attributes,
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

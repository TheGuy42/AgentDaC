from __future__ import annotations

from typing import Optional, Sequence, Any

import agentlightning as agl
from agentlightning.adapter.triplet import RewardMatchPolicy
from agentlightning.emitter.reward import get_reward_value
from agentlightning.utils.otel import filter_and_unflatten_attributes
from opentelemetry.sdk.trace import ReadableSpan


class VerlAdapter(agl.TracerTraceToTriplet):
    """
    Custom implementation of the `agentlightning.TracerTraceToTriplet` to support additional custom metrics.
    Searches for `custom_metrics` in the span attributes of the reward span, and places them in the metadata of the last triplet of the rollout.
    """

    def __init__(self, agent_match: Optional[str] = None):
        super().__init__(
            repair_hierarchy=False,  # NOTE: currently no need to repair anything since we emit traces manually
            llm_call_match=r"openai\.chat\.completion",
            agent_match=agent_match,
            exclude_llm_call_in_reward=True,
            reward_match=RewardMatchPolicy.FIRST_OCCURRENCE,
            _skip_empty_token_spans=True,
        )

    def adapt(self, source: Sequence[agl.Span] | Sequence[ReadableSpan]) -> list[agl.Triplet]:

        triplets = super().adapt(source)
        custom_metrics: dict[str, Any] = {}

        # Since we attach metrics to the reward span, search reward spans from the end.
        for span in reversed(source):
            if get_reward_value(span) is None:
                continue

            attrs = span.attributes or {}
            if not any(k.startswith("custom_metrics.") for k in attrs):
                continue

            recovered = filter_and_unflatten_attributes(attrs, "custom_metrics")  # type: ignore
            if isinstance(recovered, dict):
                custom_metrics = recovered
                break

        if not custom_metrics or not triplets:
            return triplets

        # attach to last triplet
        last = triplets[-1]
        metadata = dict(last.metadata)
        metadata["custom_metrics"] = custom_metrics
        triplets[-1] = last.model_copy(update={"metadata": metadata})
        return triplets

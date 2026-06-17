"""Live agent smoke tests (require a running vLLM/OpenAI-compatible server).

Skipped automatically unless ``--model`` is provided, e.g.::

    pytest tests/integration --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://0.0.0.0:8000/v1

These replace the four ``scripts/smoke_*.py`` runners: one parametrized suite over every agent,
sharing the configs/assertions in ``agent_specs``.
"""

from __future__ import annotations

import random

import pytest

from tests.integration.agent_specs import (
    AGENT_SPECS,
    SMOKE_PROMPTS,
    build_client,
    run_agent_on_prompt,
)
from tests.unit.factories import make_rollout

pytestmark = pytest.mark.integration

# (id, AgentSpec) pairs for readable parametrization output.
_AGENT_PARAMS = [pytest.param(spec, id=spec.name) for spec in AGENT_SPECS]


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_agent_single_prompt(spec, base_url, api_key, model, max_completion_tokens):
    """Each agent runs one prompt and the trajectory converts into trainable spans."""
    agent = spec.build_agent(build_client(base_url, api_key), model)
    _, prompt = SMOKE_PROMPTS[0]
    rollout = make_rollout(rollout_id=f"smoke-{spec.name}-0", attempt_id="1")

    await run_agent_on_prompt(
        agent=agent,
        prompt=prompt,
        rollout=rollout,
        reward=random.uniform(-1.0, 1.0),
        model=model,
        max_completion_tokens=max_completion_tokens,
    )


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_agent_multi_invocation(spec, base_url, api_key, model, max_completion_tokens):
    """Reuse one agent instance across several prompts (the multi-step interaction case) and
    confirm every invocation still converts cleanly and keeps the metric invariants."""
    agent = spec.build_agent(build_client(base_url, api_key), model)

    for index, (_, prompt) in enumerate(SMOKE_PROMPTS):
        rollout = make_rollout(rollout_id=f"smoke-{spec.name}-{index}", attempt_id=str(index))
        await run_agent_on_prompt(
            agent=agent,
            prompt=prompt,
            rollout=rollout,
            reward=random.uniform(-1.0, 1.0),
            model=model,
            max_completion_tokens=max_completion_tokens,
        )

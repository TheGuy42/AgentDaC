"""`AglTrainer.chat_kwargs` precedence + required-flag injection.

`chat_kwargs` merges the LLM's sampling parameters with the per-stage rollout kwargs and force-sets
the flags AGL/VERL require. Precedence (high -> low): rollout_config stage kwargs > rollout_config
base kwargs > llm.sampling_parameters.
"""

from __future__ import annotations

from typing import Any

import agentlightning as agl
import pytest
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.configs import DecompConfig, PromptConfig, RolloutConfig, TrainingConfig
from src.trainer import AglTrainer, RolloutStage
from src.trajectory import Trajectory


class _StubTrainer(AglTrainer):
    """Minimal concrete subclass so we can exercise the non-abstract `chat_kwargs`."""

    def create_agent(self, client: AsyncOpenAI, model: str, stage: RolloutStage) -> BaseAgent:  # pragma: no cover
        raise NotImplementedError

    def format_prompt(self, sample: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError

    async def score_trajectory(self, sample, trajectory, stage) -> Trajectory:  # pragma: no cover
        raise NotImplementedError


def make_trainer(*, rollout_config: RolloutConfig, verl_config: dict | None = None) -> _StubTrainer:
    return _StubTrainer(
        prompt_config=PromptConfig(mode="text"),
        decomp_config=DecompConfig(),
        rollout_config=rollout_config,
        train_config=TrainingConfig(),
        verl_config=verl_config or {},
    )


def make_llm(**sampling: Any) -> agl.LLM:
    return agl.LLM(endpoint="http://x/v1", model="m", sampling_parameters=sampling)


def test_required_flags_are_injected():
    trainer = make_trainer(rollout_config=RolloutConfig())
    kwargs = trainer.chat_kwargs(RolloutStage.VAL, make_llm())
    assert kwargs["extra_body"]["return_token_ids"] is True
    assert kwargs["logprobs"] is False


def test_logprobs_default_does_not_override_explicit_value():
    trainer = make_trainer(rollout_config=RolloutConfig(kwargs={"logprobs": True}))
    kwargs = trainer.chat_kwargs(RolloutStage.VAL, make_llm())
    assert kwargs["logprobs"] is True


def test_rollout_kwargs_override_llm_sampling_parameters():
    trainer = make_trainer(rollout_config=RolloutConfig(kwargs={"temperature": 0.2}))
    kwargs = trainer.chat_kwargs(RolloutStage.VAL, make_llm(temperature=1.5, top_p=0.9))
    assert kwargs["temperature"] == 0.2  # rollout config wins
    assert kwargs["top_p"] == 0.9  # llm-only param preserved


def test_stage_kwargs_applied():
    cfg = RolloutConfig(kwargs={"temperature": 0.7}, train_kwargs={"temperature": 1.0})
    trainer = make_trainer(
        rollout_config=cfg,
        verl_config={"actor_rollout_ref": {"rollout": {"temperature": 1.0}}},
    )
    assert trainer.chat_kwargs(RolloutStage.TRAIN, make_llm())["temperature"] == 1.0
    assert trainer.chat_kwargs(RolloutStage.VAL, make_llm())["temperature"] == 0.7


def test_train_stage_temperature_mismatch_does_not_raise():
    # Mismatched client vs verl temperature only logs an error; it must not crash the rollout.
    cfg = RolloutConfig(kwargs={"temperature": 0.2})
    trainer = make_trainer(
        rollout_config=cfg,
        verl_config={"actor_rollout_ref": {"rollout": {"temperature": 1.0}}},
    )
    kwargs = trainer.chat_kwargs(RolloutStage.TRAIN, make_llm())
    assert kwargs["temperature"] == 0.2

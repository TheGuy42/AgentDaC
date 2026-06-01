"""Base AgentLightning ``LitAgent`` for AgentDaC experiments.

This is the AGL analogue of the old ``ArtTrainer``: a generic rollout executor that each
experiment subclasses, implementing a few small methods. There are no callbacks — the
extension points are plain overridable methods, mirroring the previous design:

* ``create_agent(client, model, stage)`` — build the AgentDaC agent for one rollout.
* ``format_prompt(task)`` — turn a task dict into the user-prompt string.
* ``score_trajectory(task, trajectory, stage)`` — compute the scalar reward.

The base ``rollout_async`` wires these together: it reads the ``"main_llm"`` LLM resource
that VERL registers, builds an OpenAI client against it, runs the agent, scores the
trajectory, and converts ``(trajectory, reward)`` into AgentLightning spans (which VERL's
daemon turns into training triplets). Override ``forward_step`` for full control of the
prompt/chat step; most experiments only need the three methods above.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import agentlightning as agl
from agentlightning.algorithm.verl import VERL
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.aliases import UserMessage
from src.configs import DecompConfig, PromptConfig, RolloutConfig, TrainingConfig
from src.trajectory import Trajectory
from src.utils.convert import convert_trajectory
from src.utils.logging import create_logger

logger = create_logger(__name__)

# Resource key registered by VERL's daemon (agentlightning/verl/daemon.py).
MAIN_LLM_RESOURCE = "main_llm"


class RolloutStage(str, Enum):
    """Rollout stage. Maps onto AgentLightning's ``RolloutMode`` ("train"/"val")."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AglTrainer(agl.LitAgent, ABC):
    """Abstract rollout executor. Subclass per experiment.

    Args:
        prompt_config: Prompt configuration handed to the agent.
        decomp_config: Decomposition configuration handed to the agent.
        rollout_config: Inference kwargs (per stage) forwarded to ``agent.chat``.
        extra_config: Optional free-form experiment settings (e.g. decomposition
            randomization flags), available to subclasses as ``self.extra_config``.
    """

    def __init__(
        self,
        *,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        rollout_config: RolloutConfig,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config
        self.rollout_config = rollout_config
        self.extra_config = extra_config or {}

        # The actual trainer instance
        self._trainer: agl.Trainer | None = None

    # ------------------------------------------------------------------ #
    # Experiment extension points
    # ------------------------------------------------------------------ #
    @abstractmethod
    def create_agent(self, client: AsyncOpenAI, model: str, stage: RolloutStage) -> BaseAgent:
        """Build the AgentDaC agent used for a single rollout.

        Args:
            client: OpenAI client pointed at the VERL-managed inference endpoint.
            model: Served model name to request.
            stage: The rollout stage (e.g. enable decomposition randomization on TRAIN).
        """

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str:
        """Build the user-prompt string from a sample dict."""

    @abstractmethod
    async def score_trajectory(self, sample: dict[str, Any], trajectory: Trajectory, stage: RolloutStage) -> Trajectory:
        """Scores the trajectory and returns it (after updating its reward, metrics, and metadata as needed)."""

    async def forward_step(
        self,
        agent: BaseAgent,
        task: dict[str, Any],
        stage: RolloutStage,
        chat_kwargs: dict[str, Any],
    ) -> Trajectory:
        """Run the agent on the task and return its trajectory."""
        message = UserMessage(role="user", content=self.format_prompt(task))
        return await agent.chat(message, **chat_kwargs)

    def build_client(self, llm: agl.LLM, rollout: agl.AttemptedRollout) -> AsyncOpenAI:
        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        return AsyncOpenAI(base_url=base_url, api_key=llm.api_key or "EMPTY")

    def chat_kwargs(self, stage: RolloutStage, llm: agl.LLM) -> dict[str, Any]:
        # VERL-provided sampling parameters take precedence over experiment defaults.
        return {**self.rollout_config.get_kwargs(stage.value), **llm.sampling_parameters}

    async def rollout_async(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> agl.RolloutRawResult:
        llm = resources["main_llm"]
        if not isinstance(llm, agl.LLM):
            raise TypeError(f"Resource 'main_llm' must be an LLM, got {type(llm)!r}.")

        if not isinstance(rollout, agl.AttemptedRollout):
            raise TypeError(f"Expected rollout to be an AttemptedRollout, got {type(rollout)!r}.")

        stage = RolloutStage(rollout.mode or RolloutStage.TRAIN.value)
        client = self.build_client(llm, rollout)
        agent = self.create_agent(client=client, model=llm.model, stage=stage)

        kwargs = self.chat_kwargs(stage, llm)
        trajectory = await self.forward_step(agent, task, stage, kwargs)
        trajectory = await self.score_trajectory(task, trajectory, stage)
        return convert_trajectory(trajectory, rollout)

    def train(
        self,
        config: TrainingConfig,
        train_dataset: list[dict],
        val_dataset: list[dict] | None = None,
    ):
        trainer = agl.Trainer(
            n_runners=config.n_runners,
            algorithm=VERL(config=config.verl_config),
            adapter=agl.TracerTraceToTriplet(repair_hierarchy=False),  # TODO: currently no need to repair anything since we emit traces manually
        )

        trainer.fit(self, train_dataset=train_dataset, val_dataset=val_dataset)
        return trainer

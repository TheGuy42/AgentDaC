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
from src.utils.logging import create_logger
from src.custom import convert_trajectory, VerlTrainer, VerlDaemon, VerlTracer


logger = create_logger(__name__)


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
        train_config: TrainingConfig,
        verl_config: dict[str, Any],
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config
        self.rollout_config = rollout_config
        self.train_config = train_config
        self.verl_config = verl_config
        self.extra_config = extra_config or {}

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
            raise TypeError(f"Resource 'main_llm' must be an LLM, got {type(llm)}.")

        if not isinstance(rollout, agl.AttemptedRollout):
            raise TypeError(f"Expected rollout to be an AttemptedRollout, got {type(rollout)}.")

        stage = RolloutStage(rollout.mode or RolloutStage.TRAIN.value)
        client = self.build_client(llm, rollout)
        agent = self.create_agent(client=client, model=llm.model, stage=stage)

        kwargs = self.chat_kwargs(stage, llm)

        try:
            trajectory = await self.forward_step(agent, task, stage, kwargs)
            trajectory = await self.score_trajectory(task, trajectory, stage)
        except Exception as e:
            logger.error(f"Rollout failed with exception: {e}")
            return []

        return convert_trajectory(trajectory, rollout)

    def train(self, train_dataset: list[dict], val_dataset: list[dict] | None = None) -> agl.Trainer:

        # NOTE: consider using config.agentlightning.trace_aggregator.level="trajectory" (see agentlightning.algorithm.verl.VERL docs)
        # This combines multiple spans from the same rollout into a single span with the full trajectory and masking
        # Otherwise, GRPO groups contain all the spans of the trajectory simultaneously: https://github.com/microsoft/agent-lightning/issues/489
        # Read the blog about trajectory-level aggregation: https://agent-lightning.github.io/posts/trajectory_level_aggregation/
        # Another reason why we should enable it is this: https://github.com/microsoft/agent-lightning/pull/462
        assert self.verl_config["agentlightning"]["trace_aggregator"]["level"] == "trajectory", (
            "For proper training with AgentLightning, the trace aggregator level must be set to 'trajectory'. "
            "Please update verl_config.json accordingly."
        )

        if val_dataset is None:
            val_dataset = train_dataset

        trainer = agl.Trainer(
            n_runners=self.train_config.n_runners,
            algorithm=VERL(config=self.verl_config, trainer_cls=VerlTrainer, daemon_cls=VerlDaemon),
            # NOTE: currently no need to repair anything since we emit traces manually
            # NOTE: we explicitly need to use `agl.TracerTraceToTriplet` adapter since our conversion
            # function [Trajectories -> Spans] was designed around this adapter.
            adapter=VerlTracer(),
        )

        try:
            trainer.fit(self, train_dataset=train_dataset, val_dataset=val_dataset)

        except Exception as e:
            logger.error("Training failed with exception, performing cleanup...")
            trainer.kill_orphaned_processes()
            raise e

        return trainer

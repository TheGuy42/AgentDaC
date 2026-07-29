from abc import ABC, abstractmethod
from typing import Any

from src.running.stage import RolloutStage
from src.agents.base import BaseAgent
from src.aliases import UserMessage
from src.configs import DecompConfig, PromptConfig, RolloutConfig
from src.trajectory import Trajectory
from src.inference import InferenceClient
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


class RolloutError(Exception):
    """A rollout failed. Carries the partial trajectory on the exception, so the backend
    can emit a degenerate output without the task having to store it."""

    def __init__(self, message: str, trajectory: Trajectory) -> None:
        super().__init__(message)
        self.trajectory = trajectory


# TODO: WIP, design may change drastically
class RolloutTask(ABC):
    """One experiment's rollout: prompt formatting, agent construction, trajectory scoring.

    Framework-agnostic - never imports verl or art. A backend constructs it from the
    config dict it was given plus its tokenizer, then calls `rollout()` per sample.
    Subclasses implement only `format_prompt` and `score_trajectory`.
    """

    def __init__(self, configs: dict[str, Any]) -> None:
        self.configs = configs

        # TODO: why initialize() here? the agents initialize it anyways, no?
        self.prompt_config = PromptConfig.model_validate(configs["prompt_config"]).initialize()
        self.decomp_config = DecompConfig.model_validate(configs["decomp_config"])
        self.rollout_config = RolloutConfig.model_validate(configs["rollout_config"])
        self.extra_config: dict[str, Any] = configs.get("extra_config") or {}

        tw = configs.get("traj_writer") or {}
        self.trajectory_writer = TrajectoryWriter(
            tw.get("dir", "trajectories"),
            enabled=bool(tw.get("enabled", False)),
        )

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str: ...

    @abstractmethod
    async def score_trajectory(self, sample: dict[str, Any], trajectory: Trajectory, stage: RolloutStage, agent: BaseAgent) -> Trajectory: ...

    @abstractmethod
    def build_decomp_config(self, stage: RolloutStage) -> DecompConfig: ...

    @abstractmethod
    def create_agent(self, client: InferenceClient, stage: RolloutStage) -> BaseAgent: ...

    @abstractmethod
    def chat_kwargs(self, stage: RolloutStage, **kwargs) -> dict[str, Any]: ...

    async def forward_step(self, agent: BaseAgent, sample: dict[str, Any], stage: RolloutStage, chat_kwargs: dict[str, Any]) -> Trajectory:
        message = UserMessage(role="user", content=self.format_prompt(sample))
        return await agent.chat(message, **chat_kwargs)

    async def rollout(
        self,
        client: InferenceClient,
        sample: dict[str, Any],
        stage: RolloutStage,
        chat_kwargs: dict[str, Any],
        *,
        rollout_id: str,
        step: int | None = None,
    ) -> Trajectory:
        """Run + score one rollout, write it to disk, return it.

        On failure: writes under `degenerate/<rollout_id>` and raises `RolloutError`, so
        each backend handles it natively (verl -> degenerate_output to keep the batch
        shape; ART -> propagates to TrajectoryGroup, counted against max_exceptions).
        """
        agent = self.create_agent(client, stage)
        try:
            trajectory = await self.forward_step(agent, sample, stage, chat_kwargs)
            trajectory = await self.score_trajectory(sample, trajectory, stage, agent)
            self.trajectory_writer.write(trajectory, rollout_id=rollout_id, stage=stage, step=step)
            return trajectory

        except Exception as e:
            logger.error("Rollout %s failed: %s", rollout_id, e, exc_info=True)
            agent.trajectory.error(kind="critical", message=f"Rollout failed: {e}")
            trajectory = agent.trajectory.finish()
            self.trajectory_writer.write(trajectory, rollout_id=f"degenerate/{rollout_id}", stage=stage, step=step)
            raise RolloutError(str(e), trajectory) from e

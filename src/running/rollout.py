from abc import ABC, abstractmethod
from typing import Any

from src.running.stage import RolloutStage
from src.agents.base import BaseAgent
from src.aliases import UserMessage
from src.trajectory import Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


class RolloutError(Exception):
    """A rollout failed. Carries the partial trajectory on the exception."""

    def __init__(self, message: str, trajectory: Trajectory) -> None:
        super().__init__(message)
        self.trajectory = trajectory


class RolloutTask(ABC):
    """One experiment's rollout: prompt formatting, agent construction, trajectory scoring."""

    def __init__(self, configs: dict[str, Any]) -> None:
        self.configs = configs
        self.extra_config: dict[str, Any] = configs.get("extra_config") or {}

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str: ...

    """Format prompt text from raw sample data."""

    @abstractmethod
    async def score_trajectory(self, sample: dict[str, Any], trajectory: Trajectory, stage: RolloutStage, agent: BaseAgent) -> Trajectory: ...

    """Score the trajectory, updating its `reward` / `metrics` / `metadata`."""

    async def rollout(
        self,
        agent: BaseAgent,
        sample: dict[str, Any],
        stage: RolloutStage,
        chat_kwargs: dict[str, Any],
    ) -> Trajectory:
        """
        One rollout: build the prompt, create the agent, run the chat, score the trajectory.

        Args:
            agent (BaseAgent): The agent to use for the rollout.
            sample (dict[str, Any]): The sample from the dataset to use for the rollout.
            stage (RolloutStage): The rollout stage (train/val/test).
            chat_kwargs (dict[str, Any]): Additional keyword arguments to pass to the agent's chat method.

        Returns:
            Trajectory: The scored trajectory from the rollout.

        Raises:
            RolloutError: If the rollout fails, with the partial trajectory attached.
        """

        message = UserMessage(role="user", content=self.format_prompt(sample))

        try:
            trajectory = await agent.chat(message, **chat_kwargs)
            trajectory = await self.score_trajectory(sample, trajectory, stage, agent)
            return trajectory

        except Exception as e:
            agent.trajectory.error(kind="critical", message=f"Rollout failed: {e}")
            trajectory = agent.trajectory.finish()
            raise RolloutError(str(e), trajectory) from e

    async def aclose(self) -> None:
        pass

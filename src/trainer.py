from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from src.agents.base import BaseAgent
from src.aliases import UserMessage
from src.configs import DecompConfig, PromptConfig, RolloutConfig
from src.custom import VerlClient, convert_trajectory, degenerate_output
from src.trajectory import Trajectory
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


class RolloutStage(str, Enum):
    """
    Rollout stage, inferred from the dataset's ``training_stage`` column.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VerlTrainer(AgentLoopBase, ABC):
    """
    Abstract verl ``AgentLoop`` = a single rollout. Subclass per experiment.
    Subclasses implement ``create_agent`` / ``format_prompt`` / ``score_trajectory``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Rebuild the pydantic configs from the OmegaConf config, so we can use type validation and defaults.
        custom_configs = self.config.custom_configs
        self.prompt_config = PromptConfig.model_validate(OmegaConf.to_container(custom_configs.prompt_config, resolve=True))
        self.decomp_config = DecompConfig.model_validate(OmegaConf.to_container(custom_configs.decomp_config, resolve=True))
        self.rollout_kwargs = RolloutConfig.model_validate(OmegaConf.to_container(custom_configs.rollout_config, resolve=True))
        self.extra_config: dict[str, Any] = OmegaConf.to_container(custom_configs.extra_config, resolve=True)

        # Initialize the TrajectoryWriter for logging rollouts to disk.
        self.trajectory_writer = TrajectoryWriter(
            custom_configs.traj_writer.dir,
            enabled=custom_configs.traj_writer.enabled,
        )

    @abstractmethod
    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        """Build the agent for one rollout."""

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str:
        """Build the user-prompt string from the dataset row (``kwargs``)."""

    @abstractmethod
    async def score_trajectory(self, sample: dict[str, Any], trajectory: Trajectory, stage: RolloutStage) -> Trajectory:
        """Score the trajectory, updating its ``reward`` / ``metrics`` / ``metadata``."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        try:
            stage = self._stage(kwargs)
            client = VerlClient(self)
            agent = self.create_agent(client, stage)
            chat_kw = self.chat_kwargs(stage, sampling_params)
            trajectory = await self.forward_step(agent, kwargs, stage, chat_kw)
            trajectory = await self.score_trajectory(kwargs, trajectory, stage)

            # UUID assigned per prompt dispatch
            # session_id is rollout.n sample index: 0, 1, ..., n-1
            # index is dataset/batch sample index
            self.trajectory_writer.write(
                trajectory,
                rollout_id=f"{kwargs['uid']}-{kwargs['session_id']}-{kwargs['index']}",
                stage=stage.value,
                step=kwargs["global_steps"],
            )

            return convert_trajectory(
                trajectory,
                self.rollout_config.prompt_length,
                self.rollout_config.response_length,
            )

        except Exception as e:
            logger.error("Rollout failed; emitting degenerate AgentLoopOutput: %s", e, exc_info=True)
            return degenerate_output(self.tokenizer)

    async def forward_step(
        self,
        agent: BaseAgent,
        sample: dict[str, Any],
        stage: RolloutStage,
        chat_kwargs: dict[str, Any],
    ) -> Trajectory:
        """Run the agent on the formatted prompt and return its trajectory."""
        message = UserMessage(role="user", content=self.format_prompt(sample))
        return await agent.chat(message, **chat_kwargs)

    def _stage(self, kwargs: dict[str, Any]) -> RolloutStage:
        raw = kwargs.get("training_stage")
        return RolloutStage.TRAIN if raw is None else RolloutStage(str(raw))

    def chat_kwargs(self, stage: RolloutStage, sampling_params: dict[str, Any]) -> dict[str, Any]:
        extra = self.rollout_kwargs.get_kwargs(stage.value)
        if "n" in extra:
            raise ValueError("rollout_kwargs must not set 'n'; verl controls the number of rollouts per prompt.")

        kwargs = {**sampling_params, **extra}

        if stage == RolloutStage.TRAIN:
            verl_temp = self.rollout_config.temperature
            if kwargs.get("temperature") != verl_temp:
                raise ValueError(
                    f"TRAIN generation temperature {kwargs.get('temperature')!r} != verl "
                    f"rollout.temperature {verl_temp!r}. verl recomputes log-probs at the latter, "
                    f"so a mismatch corrupts on-policy GRPO. Remove the temperature override from "
                    f"rollout_config.train_kwargs."
                )
        return kwargs

from __future__ import annotations
from typing import Any
import random

from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.import_utils import load_class_from_fqn

from src.agents.registry import AgentContext, create_agent
from src.configs import DecompConfig, PromptConfig, RolloutConfig
from src.backends.verl.client import VerlClient
from src.backends.verl.convert import convert_trajectory, degenerate_output
from src.running.rollout import RolloutError, RolloutTask
from src.running.stage import RolloutStage
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


class VerlLoop(AgentLoopBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Rebuild the pydantic configs from the OmegaConf config, so we can use type validation and defaults.
        self.custom_config = self.config.custom_configs
        self.prompt_config = PromptConfig.model_validate(OmegaConf.to_container(self.custom_config.prompt_config, resolve=True))
        self.decomp_config = DecompConfig.model_validate(OmegaConf.to_container(self.custom_config.decomp_config, resolve=True))
        self.rollout_kwargs = RolloutConfig.model_validate(OmegaConf.to_container(self.custom_config.rollout_config, resolve=True))
        self.extra_config: dict[str, Any] = OmegaConf.to_container(self.custom_config.extra_config, resolve=True)  # type: ignore

        # Initialize the TrajectoryWriter for logging rollouts to disk.
        self.writer = TrajectoryWriter(
            self.custom_config.traj_writer.dir,
            enabled=self.custom_config.traj_writer.enabled,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:

        stage = self._stage(kwargs)
        client = VerlClient(self)
        chat_kw = self._backend_kwargs(stage, sampling_params)
        agent_ctx = self._agent_context(stage)
        rollout_task = self._create_task(stage)

        # UUID assigned per prompt dispatch
        # session_id is rollout.n sample index: 0, 1, ..., n-1
        # index is dataset/batch sample index
        rollout_id = f"{kwargs['uid']}-{kwargs['session_id']}-{kwargs['index']}"

        try:
            agent = create_agent(client=client, ctx=agent_ctx)
            trajectory = await rollout_task.rollout(agent=agent, sample=kwargs, stage=stage, chat_kwargs=chat_kw)

        except RolloutError as e:
            logger.error("Rollout %s failed; emitting degenerate AgentLoopOutput: %s", rollout_id, e, exc_info=True)
            self.writer.write(e.trajectory, rollout_id=f"degenerate/{rollout_id}", stage=stage, step=kwargs["global_steps"])
            return degenerate_output(e.trajectory, self.tokenizer)

        self.writer.write(trajectory, rollout_id=rollout_id, stage=stage, step=kwargs["global_steps"])

        return convert_trajectory(
            trajectory,
            self.rollout_config.prompt_length,
            self.rollout_config.response_length,
        )

    def _stage(self, kwargs: dict[str, Any]) -> RolloutStage:
        raw = kwargs.get("training_stage")
        return RolloutStage.TRAIN if raw is None else RolloutStage(str(raw))

    def build_decomp_config(self, stage: RolloutStage) -> DecompConfig:
        """The decomposition budget for one rollout, optionally randomized during training."""

        dc = self.decomp_config
        max_depth, max_tasks, max_rounds = dc.max_depth, dc.max_tasks, dc.max_rounds

        if stage == RolloutStage.TRAIN:
            if self.extra_config.get("randomize_decomp_depth", False):
                max_depth = random.randint(0, dc.max_depth)
            if self.extra_config.get("randomize_decomp_tasks", False):
                max_tasks = random.randint(0, dc.max_tasks)
            if self.extra_config.get("randomize_decomp_rounds", False):
                max_rounds = random.randint(0, dc.max_rounds)

        return DecompConfig(max_depth=max_depth, max_tasks=max_tasks, max_rounds=max_rounds)

    def _agent_context(self, stage: RolloutStage) -> AgentContext:
        return AgentContext(
            agent_key=str(self.custom_config.agent),
            prompt_config=self.prompt_config,
            decomp_config=self.build_decomp_config(stage=stage),
            extra_config=self.extra_config,
            tool_parser=self.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(self.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=self.tokenizer,
        )

    def _backend_kwargs(self, stage: RolloutStage, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Build the kwargs for the agent's `call` method, merging the sampling params with any stage-specific overrides from the verl config."""
        extra_kwargs = self.rollout_kwargs.get_kwargs(stage)
        if "n" in extra_kwargs:
            raise ValueError("rollout_kwargs must not set 'n'; verl controls the number of rollouts per prompt.")

        kwargs = {**sampling_params, **extra_kwargs}

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

    def _create_task(self, stage: RolloutStage) -> RolloutTask:
        """Rebuild the experiment's RolloutTask from its FQDN in the verl config."""
        task_cls = load_class_from_fqn(self.custom_config.task, description="RolloutTask")
        return task_cls(self.custom_config)

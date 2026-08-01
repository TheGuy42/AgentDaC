from __future__ import annotations

import asyncio
import random
from typing import Any
import statistics

import art
from art.local import LocalBackend
from art.utils import iterate_dataset
from wandb.sdk.wandb_run import Run as WandbRun

from src.agents.registry import AgentContext, create_agent
from src.backends.art.config import ArtConfig
from src.backends.art.convert import convert_trajectory
from src.configs import DecompConfig, PromptConfig, RolloutConfig
from src.inference import OAIClient
from src.running.rollout import RolloutError, RolloutTask
from src.running.stage import RolloutStage
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


def aggregate_metrics(groups: list[art.TrajectoryGroup]) -> dict[str, float]:
    """Mean of `reward` and of every trajectory metric across `groups`."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def add(key: str, value: float) -> None:
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1

    std_devs: list[float] = []
    for group in groups:
        for _ in group.exceptions:
            add("exception_rate", 1.0)

        rewards: list[float] = []
        for trajectory in group.trajectories:
            add("exception_rate", 0.0)
            add("reward", trajectory.reward)
            rewards.append(trajectory.reward)
            for key, value in trajectory.metrics.items():
                add(key, value)

        # Within-group reward spread: 0 means the group carries no GRPO signal.
        if len(rewards) > 1:
            std_devs.append(statistics.pstdev(rewards))

    metrics = {key: total / counts[key] for key, total in sums.items()}
    metrics["reward_std_dev"] = sum(std_devs) / len(std_devs) if std_devs else 0.0
    return metrics


class ArtTrainer:
    def __init__(
        self,
        model: art.TrainableModel,
        task: RolloutTask,
        config: ArtConfig,
        rollout_config: RolloutConfig,
        agent_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        extra_config: dict[str, Any] | None = None,
        writer: TrajectoryWriter | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.config = config
        self.rollout_config = rollout_config
        self.agent_name = agent_name
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config
        self.extra_config: dict[str, Any] = extra_config or {}
        self.writer = writer

    @property
    def wandb_run(self) -> WandbRun | None:
        try:
            return self.model._get_wandb_run()
        except Exception as e:
            logger.warning(f"Failed to get wandb run: {e}")
            return None

    @property
    def backend(self) -> LocalBackend:
        return self.model.backend()  # type: ignore[return-value]

    def log_hparams(self, d: dict) -> None:
        run = self.wandb_run
        if run is None:
            logger.warning("No wandb run found. Skipping hyperparameter logging.")
            return
        run.config.update(d, allow_val_change=True)

    async def close(self) -> None:
        try:
            run = self.wandb_run
            if run is not None:
                run.finish()
        except Exception as e:
            logger.error(f"Failed to finish wandb run: {e}")

        try:
            backend = self.model._backend
            if backend is not None:
                await backend.close()
        except Exception as e:
            logger.error(f"Failed to close model backend: {e}")

    def _decomp_config(self, stage: RolloutStage) -> DecompConfig:
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
            agent_key=self.agent_name,
            prompt_config=self.prompt_config,
            decomp_config=self._decomp_config(stage),
            extra_config=self.extra_config,
            tool_parser=self.config.model.tool_parser,
            reasoning_parser=self.config.model.reasoning_parser,
            # TODO: probably instantiate a tokenizer here and pass it instead?
            tokenizer=self.config.model.base_model,
        )

    def _chat_kwargs(self, stage: RolloutStage) -> dict[str, Any]:
        kwargs = self.rollout_config.get_kwargs(stage)
        return {**kwargs, "logprobs": True}

    async def rollout_one(self, sample: dict, stage: RolloutStage, step: int, rollout_id: str) -> art.Trajectory:
        """
        Perform a single rollout on a sample.

        Args:
            sample (dict): The input sample for the rollout.
            stage (RolloutStage): The current stage of the rollout (TRAIN, VAL, etc.).
            step (int): The current training step number.
            rollout_id (str): A unique identifier for this rollout.

        Returns:
            art.Trajectory: The trajectory resulting from the rollout.
        """
        client = OAIClient(model_name=self.model.get_inference_name(), client=self.model.openai_client())
        chat_kw = self._chat_kwargs(stage)
        agent_ctx = self._agent_context(stage)

        try:
            agent = create_agent(client=client, ctx=agent_ctx)
            trajectory = await self.task.rollout(agent=agent, sample=sample, stage=stage, chat_kwargs=chat_kw)

        except RolloutError as e:
            logger.error("Rollout %s failed: %s", rollout_id, e, exc_info=True)
            if self.writer:
                self.writer.write(e.trajectory, rollout_id=f"degenerate/{rollout_id}", stage=stage, step=step)
            raise

        trajectory.metrics["n_histories"] = len(trajectory.histories)

        if self.writer:
            self.writer.write(trajectory, rollout_id=rollout_id, stage=stage, step=step)

        return convert_trajectory(trajectory)

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        step: int,
        stage: RolloutStage = RolloutStage.TRAIN,
    ) -> list[art.TrajectoryGroup]:
        """
        Perform rollouts on a dataset using the model.
        Returns a list of trajectory groups, each group corresponding to a sample in the dataset.

        Args:
            dataset (list[dict]): List of samples to rollout.
            group_size (int): Number of trajectories to generate per sample.
            step (int): The current training step number.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (list[art.TrajectoryGroup]): List of trajectory groups for each sample in the dataset.
        """

        groups = []
        for sample_index, sample in enumerate(dataset):
            group = art.TrajectoryGroup(
                [
                    self.rollout_one(sample=sample, stage=stage, step=step, rollout_id=f"step-{step}-sample-{sample_index}-group-{group_index}")
                    for group_index in range(group_size)
                ]
            )
            groups.append(group)

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=f"rollout-{stage.value}",
            max_exceptions=self.config.train.max_exceptions,
            pbar_total_completion_tokens=False,
        )

        return trajectory_groups

    async def train(self, train_dataset: list[dict], val_dataset: list[dict] | None = None) -> art.TrainableModel:
        if not isinstance(self.model, art.TrainableModel):
            raise ValueError("Model must be an `art.TrainableModel` to train.")

        # Log hyperparameters
        self.log_hparams(
            {
                "agent": self.agent_name,
                "task": type(self.task).__name__,
                "model": self.model.model_dump(),
                "config": self.config.model_dump(),
                "prompt_config": self.prompt_config.model_dump(),
                "decomp_config": self.decomp_config.model_dump(),
                "rollout_config": self.rollout_config.model_dump(),
                "extra_config": self.extra_config,
                "task_config": self.task.configs,
            }
        )

        # Prepare datasets
        if val_dataset is None:
            val_dataset = train_dataset.copy()

        train_config = self.config.train

        train_iter = iterate_dataset(
            dataset=train_dataset,
            groups_per_step=train_config.num_groups,
            num_epochs=train_config.epochs,
            initial_step=await self.model.get_step(),
            use_tqdm=True,
        )

        for train_batch in train_iter:
            if train_batch.step % train_config.val_log_steps == 0:
                # Perform validation and training rollout
                val_groups, train_groups = await asyncio.gather(
                    self.rollout(
                        val_dataset,
                        group_size=1,
                        stage=RolloutStage.VAL,
                        step=train_batch.step,
                    ),
                    self.rollout(
                        train_batch.items,
                        group_size=train_config.group_size,
                        stage=RolloutStage.TRAIN,
                        step=train_batch.step,
                    ),
                )

                await self.model.log(
                    split=RolloutStage.VAL,
                    metrics=aggregate_metrics(val_groups),
                    step=train_batch.step,
                )

            else:
                # Perform training rollout
                train_groups = await self.rollout(
                    train_batch.items,
                    group_size=train_config.group_size,
                    stage=RolloutStage.TRAIN,
                    step=train_batch.step,
                )

            # Train step
            result = await self.backend.train(
                model=self.model,
                trajectory_groups=train_groups,
                **train_config.train_params.model_dump(),
                save_checkpoint=True,
                verbose=train_config.verbose,
            )

            await self.model.log(
                split=RolloutStage.TRAIN,
                metrics={**result.metrics, **aggregate_metrics(train_groups)},
                step=result.step,
            )

            # Update checkpoints
            if train_config.delete_checkpoints:
                metric_name = f"{RolloutStage.VAL}/{train_config.checkpoint_metric}"
                await self.model.delete_checkpoints(best_checkpoint_metric=metric_name)

        return self.model

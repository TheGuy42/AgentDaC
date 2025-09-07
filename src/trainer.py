from abc import abstractmethod
import random
from enum import Enum
from copy import deepcopy

from wandb.sdk.wandb_run import Run as WandbRun
import asyncio

import art
from art.utils import iterate_dataset
from art.local import LocalBackend

from src.agents import BaseAgent
from src.utils.logging import create_logger
from src.vllm_client import VllmRouter
from src.utils.replay import RewardBasedDoubleQuantileReplayBuffer
from src.utils.sample_buffer import SampleBuffer

from src.configs import (
    PromptConfig,
    PathConfig,
    RolloutConfig,
    DecompConfig,
    TrainingConfig,
    ReplayConfig,
    SampleBufferConfig,
)


logger = create_logger(__name__)


class RolloutStage(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    
    def __str__(self) -> str:
        return self.value


class Trainer:
    def __init__(
        self,
        model: art.Model,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        replay_config: ReplayConfig = ReplayConfig(),
        sample_buffer_config: SampleBufferConfig = SampleBufferConfig(),
        rollout_config: RolloutConfig | None = None,
        
        extra_config: dict | None = None,
        **kwargs,
    ):
        self.model = model
        self.path_config = path_config
        self.vllm_router = vllm_router
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config
        self.rollout_config = rollout_config or RolloutConfig()
        self.replay_config = replay_config
        self.sample_buffer_config = sample_buffer_config
        self.extra_config = extra_config or {}
        self.extra_config.update(kwargs)

        self._train_config: TrainingConfig | None = None

        self.replay_buffer = None
        if self.replay_config.use_replay:
            self.replay_buffer = RewardBasedDoubleQuantileReplayBuffer(
                directory=self.path_config.get_trajectories_path(split="train"),
                grouping_keys=self.replay_config.grouping_keys,
                buffer_size=self.replay_config.buffer_size,
                **self.replay_config.kwargs,
            )
        
        self.sample_buffer = None
        if self.sample_buffer_config.use_buffer:
            self.sample_buffer = SampleBuffer(
                max_size=self.sample_buffer_config.max_size,
            )

    @property
    def wandb_run(self) -> WandbRun | None:
        try:
            backend: LocalBackend = self.model.backend()  # type: ignore
            return backend._get_wandb_run(self.model)
        except Exception as e:
            logger.warning(f"Failed to get wandb run: {e}")
            return None

    def train_config(self) -> TrainingConfig:
        if self._train_config is None:
            raise ValueError("Training config is not set.")
        return self._train_config

    async def close(self):
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

    def log_hparams(self, d: dict):
        """
        Logs hyperparameters to wandb.

        Args:
            d (dict): Dictionary of hyperparameters to log.
        """
        run = self.wandb_run
        if run is None:
            logger.warning("No wandb run found. Skipping hyperparameter logging.")
            return
        run.config.update(d, allow_val_change=True)

    async def sync_lora(self, step: int | None = None):
        """
        Syncs the LoRA weights with the current model step.

        Args:
            step (int | None): The step to sync the LoRA weights with.
                If None, the current model step will be used.
        """
        if step is None:
            if not isinstance(self.model, art.TrainableModel):
                raise ValueError("Model must be a TrainableModel to get step.")
            step = await self.model.get_step()

        await self.vllm_router.unload_all_loras()
        curr_checkpoint_dir = self.path_config.get_step_checkpoint_dir(step)
        await self.vllm_router.load_lora(self.model.get_inference_name(), curr_checkpoint_dir)

    async def train(
        self,
        config: TrainingConfig,
        train_dataset: list[dict],
        val_dataset: list[dict] | None = None,
    ) -> art.TrainableModel:
        if not isinstance(self.model, art.TrainableModel):
            raise ValueError("Model must be an `art.TrainableModel` to train.")

        self._train_config = config

        # Prepare datasets
        if val_dataset is None:
            val_dataset = train_dataset.copy()

        if config.val_size is not None:
            val_dataset = val_dataset[: config.val_size]

        if config.train_size is not None:
            train_dataset = train_dataset[: config.train_size]

        # Log hyperparameters
        self.log_hparams(
            {
                "model": self.model.model_dump(),
                "path_config": self.path_config.model_dump(),
                "prompt_config": self.prompt_config.model_dump(),
                "decomp_config": self.decomp_config.model_dump(),
                "training_config": self.train_config().model_dump(),
                "rollout_config": self.rollout_config.model_dump(),
                "replay_config": self.replay_config.model_dump(),
                "sample_buffer_config": self.sample_buffer_config.model_dump(),
                "extra_config": self.extra_config,
            }
        )

        # Training set iterator
        train_iter = iterate_dataset(
            dataset=train_dataset,
            groups_per_step=config.num_groups,
            num_epochs=config.epochs,
            initial_step=await self.model.get_step(),
            use_tqdm=True,
        )

        await self.sync_lora()  # Sync all vLLM clients with model weights

        for train_batch in train_iter:
            # Add previous samples from buffer to the current batch
            max_prev_samples = int(config.num_groups * self.sample_buffer_config.added_ratio)
            prev_samples = []
            if self.sample_buffer and max_prev_samples > 0:
                prev_samples = self.sample_buffer.sample(k=max_prev_samples)
                logger.info(f"Max previous samples to add from buffer: {max_prev_samples}. Loaded {len(prev_samples)} samples.")

            if train_batch.step % config.val_log_steps == 0:
                # Perform validation and training rollout
                val_groups, train_groups = await asyncio.gather(
                    self.rollout(
                        dataset=val_dataset,
                        group_size=1,
                        stage=RolloutStage.VAL,
                        max_exceptions=config.max_exceptions,
                    ),
                    self.rollout(
                        dataset=train_batch.items + prev_samples,
                        group_size=config.group_size,
                        stage=RolloutStage.TRAIN,
                        max_exceptions=config.max_exceptions,
                    ),
                )
                await self.model.log(val_groups, split=RolloutStage.VAL.value)

            else:
                # Perform training rollout
                train_groups = await self.rollout(
                    dataset=train_batch.items + prev_samples,
                    group_size=config.group_size,
                    stage=RolloutStage.TRAIN,
                    max_exceptions=config.max_exceptions,
                )

            # update the sample buffer
            # sample small portion of the train groups to keep in the buffer
            if self.sample_buffer and max_prev_samples > 0:
                logger.info(f"Sampling {max_prev_samples} groups to keep in the sample buffer")
                sampled_groups = deepcopy(train_batch.items)[:max_prev_samples]
                self.sample_buffer.add(sampled_groups)

            # Train step
            await self.model.train(
                train_groups,
                config=config.art_config,
                _config=config.dev_art_config,
                verbose=config.verbose,
            )

            await self.sync_lora()  # Sync all vLLM clients with updated model

            # Update checkpoints
            if config.delete_checkpoints:
                metric_name = f"{RolloutStage.VAL.value}/{config.checkpoint_metric}"
                await self.model.delete_checkpoints(best_checkpoint_metric=metric_name)

        await self.sync_lora()
        return self.model

    async def predict(
        self,
        dataset: list[dict],
        stage: RolloutStage = RolloutStage.TEST,
        max_exceptions: int | float = 0,
    ) -> list[str | Exception]:
        """
        Perform predictions on a dataset using the model.
        Returns a list of predicted answers.

        Args:
            dataset (list[dict]): List of samples to predict.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (list[str | Exception]): List of predicted answers for each sample in the dataset.
        """

        agents = [self.create_agent(stage) for _ in dataset]
        tasks = [self.forward_step(agent, sample, stage) for agent, sample in zip(agents, dataset)]

        trajectories = await art.gather_trajectories(
            tasks,
            pbar_desc=f"predict-{stage.value}",
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        answers = []
        for tr, ag in zip(trajectories, agents):
            if not isinstance(tr, art.Trajectory):
                # Exception occurred during rollout
                answers.append(tr)
                continue

            last_message = tr.messages()[-1]
            answer = ag.parse_answer(last_message)
            answers.append(answer)

        return answers

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        stage: RolloutStage = RolloutStage.TRAIN,
        max_exceptions: int | float = 0,
    ) -> list[art.TrajectoryGroup]:
        async def sample_forward(sample: dict) -> art.Trajectory:
            agent = self.create_agent(stage)
            trajectory = await self.forward_step(agent, sample, stage)
            return await self.score_trajectory(sample, trajectory, stage)

        async def group_forward(group: art.TrajectoryGroup) -> art.TrajectoryGroup | None:
            return await self.score_group(group, stage)

        groups = []
        for sample in dataset:
            group = art.TrajectoryGroup([sample_forward(sample) for _ in range(group_size)])
            groups.append(group)

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=f"rollout-{stage.value}",
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
            after_each=group_forward,
        )

        if self.replay_buffer is not None and stage == RolloutStage.TRAIN:
            new_files = self.replay_buffer.update_trajectories()
            logger.info(f"Loaded {new_files} new trajectories into the replay buffer, total size now {self.replay_buffer.num_files_loaded}")

            for traj_group in trajectory_groups:
                for traj in traj_group.trajectories: # mark original as not replay
                    traj.metrics["replay"] = False
                group_key = self.replay_buffer._get_grouping_key(traj_group.trajectories[0])
                n = max(1, int(len(traj_group.trajectories) * self.replay_config.buffer_ratio))
                replay_buffer = self.replay_buffer.sample_group(group_key=group_key, n=n)
                for traj in replay_buffer: # mark added trajectories as replay
                    traj.metrics["replay"] = True
                traj_group.trajectories.extend(replay_buffer) # add replay to the group

        return trajectory_groups

    @abstractmethod
    def create_agent(
        self,
        stage: RolloutStage,
    ) -> BaseAgent:
        """
        Create an instance of an BaseAgent for the given stage.
        Args:
            stage (RolloutStage): The current stage of the rollout.
        Returns:
            (BaseAgent): An instance of an BaseAgent.
        """
        raise NotImplementedError("Subclasses must implement the create_agent method.")

    @abstractmethod
    async def forward_step(
        self,
        agent: BaseAgent,
        sample: dict,
        stage: RolloutStage,
    ) -> art.Trajectory:
        """
        Perform a raw forward step for a single sample.

        Args:
            agent (BaseAgent): The agent to use for the forward step.
            sample (dict): A single sample from the dataset.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (art.Trajectory): The trajectory generated by the model for the sample.
        """
        raise NotImplementedError("Subclasses must implement the forward_step method.")

    @abstractmethod
    async def score_trajectory(
        self,
        sample: dict,
        trajectory: art.Trajectory,
        stage: RolloutStage,
    ) -> art.Trajectory:
        """
        Score a single trajectory based on the sample and the trajectory itself.
        This function populates all the relevant fields in the trajectory, including reward, metrics, and metadata.

        Args:
            sample (dict): The original sample from the dataset.
            trajectory (art.Trajectory): The trajectory to score.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (art.Trajectory): The scored trajectory.
        """
        raise NotImplementedError("Subclasses must implement the score_trajectory method.")

    async def score_group(
        self,
        group: art.TrajectoryGroup,
        stage: RolloutStage,
    ) -> art.TrajectoryGroup | None:
        """
        Score a group of trajectories.
        This function is used to score a group of trajectories after they have been rolled out,
        and each group trajectory has been scored individually.

        Args:
            group (art.TrajectoryGroup): The group of trajectories to score.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (art.TrajectoryGroup | None): The scored group of trajectories or None if scoring failed.
        """
        if stage != RolloutStage.TRAIN:
            return group

        ruler_config = self.train_config().ruler_config
        if ruler_config is not None and ruler_config.judge_model is not None:
            logger.warning(
                "Skipping RULER scoring. To enable RULER scoring, please override the `score_group` "
                "method in your Trainer subclass and implement the desired behavior there."
            )
        return group

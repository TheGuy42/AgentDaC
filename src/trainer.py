from abc import abstractmethod
import random
import tqdm
import tqdm.asyncio

from wandb.sdk.wandb_run import Run as WandbRun
import numpy as np
import asyncio

import art
from art.utils import iterate_dataset
from art.local import LocalBackend
from art.rewards import ruler_score_group

from src.utils.logging import create_logger
from src.vllm_client import VllmRouter

from src.configs import (
    PromptConfig,
    PathConfig,
    RolloutConfig,
    RolloutStage,
    StopCriteria,
    TrainingConfig,
    RulerConfig,
)


logger = create_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: art.Model,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        rollout_config: RolloutConfig,
    ):
        self.model = model
        self.path_config = path_config
        self.vllm_router = vllm_router
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria
        self.rollout_config = rollout_config

    @property
    def wandb_run(self) -> WandbRun | None:
        try:
            backend: LocalBackend = self.model.backend()  # type: ignore
            return backend._get_wandb_run(self.model)
        except Exception as e:
            logger.warning(f"Failed to get wandb run: {e}")
            return None

    def close(self):
        try:
            run = self.wandb_run
            if run is not None:
                run.finish()
        except Exception as e:
            logger.error(f"Failed to finish wandb run: {e}")

        try:
            backend: None | art.Backend = self.model._backend  # type: ignore
            if backend is not None:
                backend.close()  # type: ignore
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

        # Prepare datasets
        if val_dataset is None:
            val_dataset = train_dataset.copy()

        random.Random(0).shuffle(val_dataset)
        random.Random(1).shuffle(train_dataset)

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
                "stop_criteria": self.stop_criteria.model_dump(),
                "training_config": config.model_dump(),
                "rollout_config": self.rollout_config.model_dump(),
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
            if train_batch.step % config.val_log_steps == 0:
                # Perform validation and training rollout
                val_groups, train_groups = await asyncio.gather(
                    self.rollout(
                        dataset=val_dataset,
                        group_size=1,
                        stage=RolloutStage.Val,
                        max_exceptions=config.max_exceptions,
                    ),
                    self.rollout(
                        dataset=train_batch.items,
                        group_size=config.group_size,
                        stage=RolloutStage.Train,
                        max_exceptions=config.max_exceptions,
                        ruler_config=config.ruler_config,
                    ),
                )
                await self.model.log(val_groups, split=RolloutStage.Val.value)

            else:
                # Perform training rollout
                train_groups = await self.rollout(
                    dataset=train_batch.items,
                    group_size=config.group_size,
                    stage=RolloutStage.Train,
                    max_exceptions=config.max_exceptions,
                    ruler_config=config.ruler_config,
                )

            # Log training trajectories
            if train_batch.step % config.train_log_steps == 0:
                await self.model.log(train_groups, split=RolloutStage.Train.value)

            # Filter groups with low reward standard deviation
            filtered_groups = []
            for group in train_groups:
                rewards = [tr.reward for tr in group.trajectories]
                if config.min_reward_stdev is None or np.std(rewards) >= config.min_reward_stdev:
                    filtered_groups.append(group)

            train_groups = filtered_groups

            if not train_groups:
                logger.warning(f"No trajectories left to train on at step {train_batch.step}. Skipping this step.")
                continue

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
                metric_name = f"{RolloutStage.Val.value}/{config.checkpoint_metric}"
                await self.model.delete_checkpoints(best_checkpoint_metric=metric_name)

        await self.sync_lora()
        return self.model

    async def predict(
        self,
        dataset: list[dict],
        stage: RolloutStage = RolloutStage.Val,
    ) -> list[str] | BaseException:
        """
        Perform predictions on a dataset using the model.
        Returns a list of predicted answers.

        Args:
            dataset (list[dict]): List of samples to predict.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (list[str]): List of predicted answers for each sample in the dataset.
        """

        tasks = [self.predict_step(sample, stage) for sample in dataset]

        return await tqdm.asyncio.tqdm.gather(
            *tasks,
            desc=f"predict-{stage.value}",
            total=len(tasks),
            leave=False,
        )

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        stage: RolloutStage = RolloutStage.Train,
        max_exceptions: int | float = 0,
        ruler_config: RulerConfig | None = None,
    ) -> list[art.TrajectoryGroup]:
        async def sample_forward(sample: dict) -> art.Trajectory:
            trajectory = await self.forward_step(sample, stage)
            return await self.score_trajectory(sample, trajectory, stage)

        async def group_forward(group: art.TrajectoryGroup) -> art.TrajectoryGroup | None:
            if ruler_config and ruler_config.judge_model:
                group = await ruler_score_group(
                    group=group,
                    **ruler_config.model_dump(exclude_none=True, exclude_unset=True),
                )  # type: ignore
                if group is None:
                    return None
            try:
                return await self.score_group(group, stage)
            except Exception as e:
                logger.warning(f"Failed to score group: {e}")
                return None

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

        return trajectory_groups

    @abstractmethod
    async def predict_step(
        self,
        sample: dict,
        stage: RolloutStage,
    ) -> str:
        """
        Perform a prediction step for a single sample.
        This function is used to get the final answer from the model.

        Args:
            sample (dict): A single sample from the dataset.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (str): The predicted answer for the sample.
        """
        raise NotImplementedError("Subclasses must implement the predict_step method.")

    @abstractmethod
    async def forward_step(
        self,
        sample: dict,
        stage: RolloutStage,
    ) -> art.Trajectory:
        """
        Perform a raw forward step for a single sample.

        Args:
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

    @abstractmethod
    async def score_group(
        self,
        group: art.TrajectoryGroup,
        stage: RolloutStage,
    ) -> art.TrajectoryGroup:
        """
        Score a group of trajectories.
        This function is used to score a group of trajectories after they have been rolled out,
        and each group trajectory has been scored individually.

        Note: If RULER is used, this method will be called after the group has been scored by the RULER.

        Args:
            group (art.TrajectoryGroup): The group of trajectories to score.
            stage (RolloutStage): The current stage of the rollout.

        Returns:
            (art.TrajectoryGroup): The scored group of trajectories.
        """
        raise NotImplementedError("Subclasses must implement the score_group method.")

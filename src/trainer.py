from abc import abstractmethod
import random

from wandb.sdk.wandb_run import Run as WandbRun
from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path
import asyncio

import art
from art.utils import iterate_dataset
from art.local import LocalBackend
from art.rewards import ruler_score_group

from src.vllm_client import VllmRouter
from src.models import PathConfig
from src.dac_agent import ChatMessage, PromptConfig, StopCriteria
from src.utils import text as text_utils
from src.utils.logging import create_logger
from src.utils.io import save_base_model


logger = create_logger(__name__)


class RulerConfig(BaseModel):
    judge_model: str | None = None
    extra_litellm_params: dict | None = None
    rubric: str | None = None
    swallow_exceptions: bool = True
    debug: bool = False


class TrainingConfig(BaseModel, extra="allow"):
    epochs: int = 10
    num_groups: int = 12
    group_size: int = 10
    min_reward_stdev: float | None = None

    train_log_steps: int = 1
    train_size: int | None = None
    val_log_steps: int = 5
    val_size: int | None = None
    delete_checkpoints: bool = False
    checkpoint_metric: str = "reward"

    rollout_kwargs: dict = Field(default_factory=dict)
    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None
    ruler_config: RulerConfig | None = Field(default_factory=RulerConfig)

    verbose: bool = False
    max_exceptions: int | float = 0

    def save(self, dir_name: str, file_name: str = "train_config.json") -> None:
        """
        Save the training configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)


class Trainer:
    def __init__(
        self,
        model: art.Model,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        default_kwargs: dict | None = None,
    ):
        if default_kwargs is None:
            default_kwargs = {}

        self.model = model
        self.path_config = path_config
        self.vllm_router = vllm_router
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria
        self.default_kwargs = default_kwargs

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
                "default_kwargs": self.default_kwargs,
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
                        max_exceptions=config.max_exceptions,
                        pbar_desc="rollout-val",
                        **config.rollout_kwargs,
                    ),
                    self.rollout(
                        dataset=train_batch.items,
                        group_size=config.group_size,
                        max_exceptions=config.max_exceptions,
                        pbar_desc="rollout-train",
                        ruler_config=config.ruler_config,
                        **config.rollout_kwargs,
                    ),
                )
                await self.model.log(val_groups, split="val")

            else:
                # Perform training rollout
                train_groups = await self.rollout(
                    dataset=train_batch.items,
                    group_size=config.group_size,
                    max_exceptions=config.max_exceptions,
                    pbar_desc="rollout-train",
                    ruler_config=config.ruler_config,
                    **config.rollout_kwargs,
                )

            # Log training trajectories
            if train_batch.step % config.train_log_steps == 0:
                await self.model.log(train_groups, split="train")

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
                metric_name = f"val/{config.checkpoint_metric}"
                await self.model.delete_checkpoints(best_checkpoint_metric=metric_name)

        await self.sync_lora()
        return self.model

    async def predict(
        self,
        dataset: list[dict],
        max_exceptions: int | float = 0,
        pbar_desc: str = "predict",
        **kwargs,
    ) -> list[str] | BaseException:
        """
        Perform predictions on a dataset using the model.
        Returns a list of predicted answers.

        Args:
            dataset (list[dict]): List of samples to predict.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            pbar_desc (str): Description for the progress bar.
            **kwargs: Additional keyword arguments for the prediction step.

        Returns:
            (list[str]): List of predicted answers for each sample in the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        forward_tasks = [self.forward_step(sample, **kwargs) for sample in dataset]

        trajectories = await art.gather_trajectories(
            forward_tasks,
            pbar_desc=pbar_desc,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        answers = []
        for tr in trajectories:
            if not isinstance(tr, art.Trajectory):
                answers.append(tr)
                continue

            answer = self.extract_answer(tr)
            answers.append(answer)

        return answers

    def extract_answer(self, trajectory: art.Trajectory) -> str:
        """
        Extract a final answer from a trajectory.

        Args:
            trajectory (art.Trajectory): The trajectory from which to extract the answer.

        Returns:
            (str): The extracted answer from the trajectory.
        """
        last_messages = trajectory.messages()[-1]
        last_chat = ChatMessage.model_validate(last_messages, from_attributes=True)

        if last_chat.role != "assistant":
            logger.warning(
                f"Expected the last message to be from the assistant, but got {last_chat.role}. "
                "Returning an empty answer."
            )
            return ""

        return text_utils.extract_answer(last_chat.content)

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        max_exceptions: int | float = 0,
        ruler_config: RulerConfig | None = None,
        pbar_desc: str = "rollout",
        **kwargs,
    ) -> list[art.TrajectoryGroup]:
        if not kwargs:
            kwargs = self.default_kwargs

        async def sample_forward(sample: dict) -> art.Trajectory:
            trajectory = await self.forward_step(sample, **kwargs)
            return await self.score_trajectory(sample, trajectory)

        async def group_forward(group: art.TrajectoryGroup) -> art.TrajectoryGroup | None:
            if ruler_config and ruler_config.judge_model:
                group = await ruler_score_group(group, **ruler_config.model_dump(exclude_none=True, exclude_unset=True))  # type: ignore
                if group is None:
                    return None
            try:
                return await self.score_group(group)
            except Exception as e:
                logger.warning(f"Failed to score group: {e}")
                return None

        groups = []
        for sample in dataset:
            group = art.TrajectoryGroup([sample_forward(sample) for _ in range(group_size)])
            groups.append(group)

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=pbar_desc,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
            after_each=group_forward,
        )

        return trajectory_groups

    @abstractmethod
    async def forward_step(
        self,
        sample: dict,
        **kwargs,
    ) -> art.Trajectory:
        """
        Perform a raw forward step for a single sample.

        Args:
            sample (dict): A single sample from the dataset.
            **kwargs: Additional keyword arguments for the forward step.

        Returns:
            (art.Trajectory): The trajectory generated by the model for the sample.
        """
        raise NotImplementedError("Subclasses must implement the forward_step method.")

    @abstractmethod
    async def score_trajectory(
        self,
        sample: dict,
        trajectory: art.Trajectory,
    ) -> art.Trajectory:
        """
        Score a single trajectory based on the sample and the trajectory itself.
        This function populates all the relevant fields in the trajectory, including reward, metrics, and metadata.

        Args:
            sample (dict): The original sample from the dataset.
            trajectory (art.Trajectory): The trajectory to score.

        Returns:
            (art.Trajectory): The scored trajectory.
        """
        raise NotImplementedError("Subclasses must implement the score_trajectory method.")

    @abstractmethod
    async def score_group(
        self,
        group: art.TrajectoryGroup,
    ) -> art.TrajectoryGroup:
        """
        Score a group of trajectories.
        This function is used to score a group of trajectories after they have been rolled out,
        and each group trajectory has been scored individually.

        Note: If RULER is used, this method will be called after the group has been scored by the RULER.

        Args:
            group (art.TrajectoryGroup): The group of trajectories to score.

        Returns:
            (art.TrajectoryGroup): The scored group of trajectories.
        """
        raise NotImplementedError("Subclasses must implement the score_group method.")

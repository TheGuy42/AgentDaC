from abc import abstractmethod
import sys

import wandb
from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path

import art
from art.utils import iterate_dataset
from art.local import LocalBackend

from src.vllm_client import VllmRouter
from src.models import PathConfig
from src.dac_agent import ChatMessage, PromptConfig, StopCriteria
from src.utils import text as text_utils
from src.utils.logging import create_logger
from src.utils.io import save_base_model


logger = create_logger(__name__)


class TrainingConfig(BaseModel, extra="allow"):
    epochs: int = 10
    num_groups: int = 5
    group_size: int = 10
    min_reward_stdev: float | None = None

    train_log_steps: int | None = 1
    eval_log_steps: int | None = None
    eval_size: int | None = None
    checkpoint_metric: str = "reward"

    rollout_kwargs: dict = Field(default_factory=dict)
    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None

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
    def wandb_run(self) -> wandb.Run | None:
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
        eval_dataset: list[dict] | None = None,
    ) -> art.TrainableModel:
        if not isinstance(self.model, art.TrainableModel):
            raise ValueError("Model must be an `art.TrainableModel` to train.")

        if eval_dataset is None:
            eval_dataset = train_dataset

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
        train_batch_size = config.num_groups * config.group_size
        train_step = await self.model.get_step()
        train_iter = iterate_dataset(
            dataset=train_dataset,
            groups_per_step=train_batch_size,
            num_epochs=config.epochs,
            initial_step=train_step,
            use_tqdm=True,
        )

        # Evaluation set iterator
        eval_batch_size = config.eval_size if config.eval_size else len(eval_dataset)
        eval_step = train_step // config.eval_log_steps if config.eval_log_steps else 0
        eval_iter = iterate_dataset(
            dataset=eval_dataset,
            groups_per_step=eval_batch_size,
            num_epochs=sys.maxsize,
            initial_step=eval_step,
            use_tqdm=False,
        )

        await self.sync_lora()  # Sync all vLLM clients with model weights

        for train_batch in train_iter:
            step_data = train_batch.items
            global_step = train_batch.step

            # Evaluate and log
            if (config.eval_log_steps is not None) and (global_step % config.eval_log_steps == 0):
                trajectories = await self.rollout(
                    dataset=next(eval_iter).items,
                    max_exceptions=config.max_exceptions,
                    **config.rollout_kwargs,
                )
                await self.model.log(trajectories, split="eval")

            # Perform training rollout
            trajectory_groups = await self.rollout(
                step_data,
                group_size=config.group_size,
                max_exceptions=config.max_exceptions,
                **config.rollout_kwargs,
            )

            # Log training trajectories
            if (config.train_log_steps is not None) and (global_step % config.train_log_steps == 0):
                await self.model.log(trajectory_groups, split="train")

            # Filter groups with low reward standard deviation
            filtered_groups = []
            for group in trajectory_groups:
                rewards = [tr.reward for tr in group.trajectories]
                if config.min_reward_stdev is None or np.std(rewards) >= config.min_reward_stdev:
                    filtered_groups.append(group)

            trajectory_groups = filtered_groups

            if not trajectory_groups:
                logger.warning(f"No trajectories left to train on at step {global_step}. Skipping this step.")
                continue

            # Train step
            await self.model.train(
                trajectory_groups,
                config=config.art_config,
                _config=config.dev_art_config,
                verbose=config.verbose,
            )

            await self.sync_lora()  # Sync all vLLM clients with updated model

            # Update checkpoints
            split_name = "eval" if config.eval_log_steps is not None else "train"
            metric_name = f"{split_name}/{config.checkpoint_metric}"
            await self.model.delete_checkpoints(best_checkpoint_metric=metric_name)

        await self.sync_lora()
        return self.model

    async def predict(
        self,
        dataset: list[dict],
        max_exceptions: int | float = 0,
        **kwargs,
    ) -> list[str]:
        """
        Perform predictions on a dataset using the model.
        Returns a list of predicted answers.

        Args:
            dataset (list[dict]): List of samples to predict.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            **kwargs: Additional keyword arguments for the prediction step.

        Returns:
            list[str]: List of predicted answers for each sample in the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        trajectories = await self.forward(dataset, max_exceptions=max_exceptions, **kwargs)

        answers = []
        for tr in trajectories:
            last_messages = tr.messages()[-1]
            last_chat = ChatMessage.model_validate(last_messages, from_attributes=True)
            if last_chat.role != "assistant":
                logger.warning(
                    f"Expected the last message to be from the assistant, but got {last_chat.role}. "
                    "Returning an empty answer."
                )
                answers.append("")
                continue

            answer = text_utils.extract_answer(last_chat.content)
            answers.append(answer)
        return answers

    async def forward(
        self,
        dataset: list[dict],
        max_exceptions: int | float = 0,
        **kwargs,
    ) -> list[art.Trajectory]:
        """
        Performs a raw forward pass through the model for a given dataset.

        Args:
            dataset (list[dict]): List of samples to process.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            **kwargs: Additional keyword arguments for the forward step.

        Returns:
            list[art.Trajectory]: List of trajectories generated by the model for the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        forward_tasks = [self.forward_step(sample, **kwargs) for sample in dataset]

        trajectories = await art.gather_trajectories(
            forward_tasks,
            pbar_desc="forward",
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        filtered = [tr for tr in trajectories if isinstance(tr, art.Trajectory)]

        if len(filtered) != len(trajectories):
            logger.warning(
                f"Some tasks in the forward pass did not return a valid trajectory. "
                f"Expected {len(trajectories)} but got {len(filtered)}."
            )

        return filtered

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
            art.Trajectory: The trajectory generated by the model for the sample.
        """
        raise NotImplementedError("Subclasses must implement the forward_step method.")

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int | None = None,
        max_exceptions: int | float = 0,
        **kwargs,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            group_size (int | None): Size of each group to split the dataset into.
                If None, the entire dataset is treated as a single group.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            **kwargs: Additional keyword arguments for the rollout step.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        if group_size is None:
            group_size = len(dataset)

        groups = [
            art.TrajectoryGroup([self.rollout_step(sample, **kwargs) for sample in data])
            for data in self.split_into_groups(dataset, group_size)
        ]

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        return trajectory_groups

    def split_into_groups(self, batch: list[dict], group_size: int) -> list[list[dict]]:
        return [batch[i : i + group_size] for i in range(0, len(batch), group_size)]

    @abstractmethod
    async def rollout_step(
        self,
        sample: dict,
        **kwargs,
    ) -> art.Trajectory:
        """
        Performs a single training rollout step for a single sample.
        Crucially, the returned trajectory must have the `reward` field set to the expected reward for the sample.
        It should also contain updated metadata and metrics.

        Args:
            sample (dict): A single sample from the training dataset.
            **kwargs: Additional keyword arguments for the rollout step.

        Returns:
            art.Trajectory: The trajectory generated by the model for the sample, which includes the reward.
        """
        raise NotImplementedError("Subclasses must implement the train_step method.")

from abc import abstractmethod
import os
import sys

from wandb.sdk.wandb_run import Run
from pydantic import BaseModel, Field
import numpy as np

import art
from art.dev import InternalModelConfig
from art.utils import iterate_dataset, retry
from art.local import LocalBackend

from src.vllm_client import VllmClient, VllmRouter
from src.models import PathConfig
from src.dac_agent import AgentNode, ChatMessage, PromptConfig, StopCriteria
from src.utils import text as text_utils
from src.utils.logging import create_logger


logger = create_logger(__name__)


class TrainingConfig(BaseModel, extra="allow", strict=True):
    epochs: int = 10
    num_groups: int = 5
    group_size: int = 10
    min_reward_stdev: float | None = None
    verbose: bool = False

    log_every: int | None = None  # log training results every `log_every` steps
    eval_every: int | None = None  # evaluate on eval dataset every `eval_every` steps
    eval_size: int | None = None  # if None then use the entire eval dataset

    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None


# TODO: should add a standardized way of handling exceptions
# we may encounter exceptions every time we predict with the agent
# need to figure out how to deal with them gracefully


class Trainer:
    def __init__(
        self,
        model: art.TrainableModel,
        inference_clients: list[VllmClient],
        path_config: PathConfig,
        train_config: TrainingConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
    ):
        self.model = model
        self.path_config = path_config
        self.training_config = train_config
        self.vllm_router = VllmRouter(vllm_clients=inference_clients)
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria

    @property
    def model_config(self) -> InternalModelConfig:
        config = self.model._internal_config
        if config is None:
            raise ValueError("Model configuration is not set.")
        return config

    @property
    def logger_run(self) -> Run:
        backend: LocalBackend = self.model.backend()  # type: ignore
        return backend._get_wandb_run(self.model)  # type: ignore

    def close(self):
        try:
            self.logger_run.finish()
        except Exception as e:
            logger.error(f"Failed to finish logger run: {e}")
        try:
            # NOTE: for the base class Backend, close() method is async
            # but as a result of a bug, for LocalBackend it is not async
            backend: LocalBackend = self.model.backend()  # type: ignore
            backend.close()
        except Exception as e:
            logger.error(f"Failed to close model backend: {e}")

    def log_hparams(self):
        run = self.logger_run
        if run is None:
            logger.warning("No wandb run found. Skipping hyperparameter logging.")
            return

        run.config.update(
            {
                "model": self.model.model_dump(),
                "path_config": self.path_config.model_dump(),
                "training_config": self.training_config.model_dump(),
                "prompt_config": self.prompt_config.model_dump(),
                "stop_criteria": self.stop_criteria.model_dump(),
            },
            allow_val_change=True,
        )

    def get_client(self) -> VllmClient:
        return next(self.vllm_router)

    def create_agent(self) -> AgentNode:
        """
        Create a new instance of the AgentNode with the current model and configuration.
        """
        client = self.get_client()
        return AgentNode(
            model_name=client.get_inference_name(),
            openai_client=client.client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    @retry(max_attempts=3, delay=1)
    def update_lora(self, step: int):
        prev_checkpoint_dir = self.path_config.get_step_checkpoint_dir(step - 1)
        if os.path.exists(prev_checkpoint_dir):
            success = self.vllm_router.unload_lora(self.path_config.run_name)
            if not success:
                raise RuntimeError(f"Failed to unload LoRA for step {step - 1}.")

        curr_checkpoint_dir = self.path_config.get_step_checkpoint_dir(step)
        if os.path.exists(curr_checkpoint_dir):
            success = self.vllm_router.load_lora(self.path_config.run_name, curr_checkpoint_dir)
            if not success:
                raise RuntimeError(f"Failed to load LoRA for step {step}.")

    async def train(self, train_dataset: list[dict], eval_dataset: list[dict]) -> art.TrainableModel:
        model = self.model
        vllm_router = self.vllm_router
        train_config = self.training_config

        vllm_router.unload_all_loras()
        self.log_hparams()

        # Training set iterator
        train_batch_size = train_config.num_groups * train_config.group_size
        train_step = await model.get_step()
        train_iter = iterate_dataset(
            dataset=train_dataset,
            groups_per_step=train_batch_size,
            num_epochs=train_config.epochs,
            initial_step=train_step,
            use_tqdm=True,
        )

        # Evaluation set iterator
        eval_batch_size = train_config.eval_size if train_config.eval_size else len(eval_dataset)
        eval_step = train_step // train_config.eval_every if train_config.eval_every else 0
        eval_iter = iterate_dataset(
            dataset=eval_dataset,
            groups_per_step=eval_batch_size,
            num_epochs=sys.maxsize,
            initial_step=eval_step,
            use_tqdm=False,
        )

        for train_batch in train_iter:
            step_data = train_batch.items
            global_step = train_batch.step

            self.update_lora(global_step)

            # Evaluate model
            if (train_config.eval_every is not None) and (global_step % train_config.eval_every == 0):
                eval_batch = next(eval_iter)
                await self.rollout(eval_batch.items, split="eval", log=True)

            # Perform Rollout
            log_step = (train_config.log_every is not None) and (global_step % train_config.log_every == 0)
            trajectory_groups = await self.rollout(step_data, split="train", log=log_step)

            # Filter groups with low reward standard deviation
            filtered_groups = []
            for group in trajectory_groups:
                rewards = [tr.reward for tr in group.trajectories]
                if train_config.min_reward_stdev is None or np.std(rewards) >= train_config.min_reward_stdev:
                    filtered_groups.append(group)

            trajectory_groups = filtered_groups

            if not trajectory_groups:
                logger.warning(f"No trajectories left to train on at step {global_step}. Skipping this step.")
                continue

            # Update checkpoints
            metric_name = "eval/reward" if train_config.eval_every is not None else "train/reward"
            await model.delete_checkpoints(metric_name)

            # Train step
            await model.train(
                trajectory_groups,
                config=train_config.art_config,
                _config=train_config.dev_art_config,
                verbose=train_config.verbose,
            )

        # Final evaluation after training
        eval_batch = next(eval_iter)
        print("Evaluating final mode...")
        await self.rollout(eval_batch.items, split="eval", log=True)

        return self.model

    async def predict(self, dataset: list[dict], **kwargs) -> list[str]:
        """
        Perform predictions on a dataset using the model.
        Returns a list of predicted answers.

        Args:
            dataset (list[dict]): List of samples to predict.
            **kwargs: Additional keyword arguments for the prediction step.

        Returns:
            list[str]: List of predicted answers for each sample in the dataset.
        """

        trajectories = await self.forward(dataset, **kwargs)

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

    async def forward(self, dataset: list[dict], **kwargs) -> list[art.Trajectory]:
        """
        Performs a raw forward pass through the model for a given dataset.

        Args:
            dataset (list[dict]): List of samples to process.
            **kwargs: Additional keyword arguments for the forward step.

        Returns:
            list[art.Trajectory]: List of trajectories generated by the model for the dataset.
        """
        forward_tasks = [self.forward_step(sample, **kwargs) for sample in dataset]
        trajectories = await art.gather_trajectories(forward_tasks, pbar_desc="forward", max_exceptions=0)
        return trajectories

    @abstractmethod
    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
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
        split: str = "train",
        log: bool = False,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            split (str): The split name for logging purposes (e.g., "train", "eval").
            log (bool): Whether to log the rollout results.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        groups = [
            art.TrajectoryGroup([self.rollout_step(sample) for sample in data])
            for data in self.split_into_groups(dataset, self.training_config.group_size)
        ]

        trajectory_groups = await art.gather_trajectory_groups(groups, pbar_desc=split, max_exceptions=0)

        if log:
            await self.model.log(trajectory_groups, split=split)

        return trajectory_groups

    def split_into_groups(self, batch: list[dict], group_size: int) -> list[list[dict]]:
        return [batch[i : i + group_size] for i in range(0, len(batch), group_size)]

    @abstractmethod
    async def rollout_step(self, sample: dict) -> art.Trajectory:
        """
        Performs a single training rollout step for a single sample.
        Crucially, the returned trajectory must have the `reward` field set to the expected reward for the sample.
        It should also contain updated metadata and metrics.

        Args:
            sample (dict): A single sample from the training dataset.

        Returns:
            art.Trajectory: The trajectory generated by the model for the sample, which includes the reward.
        """
        raise NotImplementedError("Subclasses must implement the train_step method.")

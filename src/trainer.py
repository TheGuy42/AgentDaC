from abc import abstractmethod
import sys

from wandb.sdk.wandb_run import Run
from pydantic import BaseModel, Field
import numpy as np

import art
from art.dev import InternalModelConfig
from art.utils import iterate_dataset
from art.local import LocalBackend

from src.vllm_client import VllmRouter
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

    log_every: int | None = None  # log training results every `log_every` steps
    eval_every: int | None = None  # evaluate on eval dataset every `eval_every` steps
    eval_size: int | None = None  # if None then use the entire eval dataset

    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None

    verbose: bool = False
    max_exceptions: int | float = 0


class Trainer:
    def __init__(
        self,
        model: art.TrainableModel,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        train_config: TrainingConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
    ):
        self.model = model
        self.path_config = path_config
        self.training_config = train_config
        self.vllm_router = vllm_router
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

    @property
    def inference_name(self) -> str:
        """
        Returns the inference name of the model.
        This is used to identify the model in the vLLM router.
        """
        return self.model.get_inference_name()

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
        """
        Logs hyperparameters to wandb.

        Note: Inheriting classes should override this method to log additional hyperparameters.
        """
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

    def create_agent(self) -> AgentNode:
        """
        Create a new instance of the AgentNode with the current model and configuration.
        """
        client = self.vllm_router.next()
        return AgentNode(
            model_name=self.inference_name,
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    async def sync_lora(self, step: int | None = None):
        """
        Syncs the LoRA weights with the current model step.

        Args:
            step (int | None): The step to sync the LoRA weights with.
                If None, the current model step will be used.
        """
        if step is None:
            step = await self.model.get_step()

        await self.vllm_router.unload_all_loras()
        curr_checkpoint_dir = self.path_config.get_step_checkpoint_dir(step)
        await self.vllm_router.load_lora(self.inference_name, curr_checkpoint_dir)

    async def train(self, train_dataset: list[dict], eval_dataset: list[dict]) -> art.TrainableModel:
        config = self.training_config
        self.log_hparams()

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
        eval_step = train_step // config.eval_every if config.eval_every else 0
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

            # Evaluate model
            if (config.eval_every is not None) and (global_step % config.eval_every == 0):
                await self.rollout(
                    dataset=next(eval_iter).items,
                    split="eval",
                    max_exceptions=config.max_exceptions,
                    log=True,
                )

            # Perform Rollout
            log_step = (config.log_every is not None) and (global_step % config.log_every == 0)
            trajectory_groups = await self.rollout(
                dataset=step_data,
                split="train",
                max_exceptions=config.max_exceptions,
                log=log_step,
            )

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
            metric_name = "eval/reward" if config.eval_every is not None else "train/reward"
            await self.model.delete_checkpoints(metric_name)

        # Final evaluation after training
        await self.sync_lora()
        eval_batch = next(eval_iter)
        await self.rollout(eval_batch.items, split="eval", log=True)

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
        forward_tasks = [self.forward_step(sample, **kwargs) for sample in dataset]

        trajectories = await art.gather_trajectories(
            forward_tasks,
            pbar_desc="forward",
            max_exceptions=max_exceptions,
        )

        filtered = [tr for tr in trajectories if isinstance(tr, art.Trajectory)]

        if len(filtered) != len(trajectories):
            logger.warning(
                f"Some tasks in the forward pass did not return a valid trajectory. "
                f"Expected {len(trajectories)} but got {len(filtered)}."
            )

        return filtered

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
        max_exceptions: int | float = 0,
        log: bool = False,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            split (str): The split name for logging purposes (e.g., "train", "eval").
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            log (bool): Whether to log the rollout results.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        groups = [
            art.TrajectoryGroup([self.rollout_step(sample) for sample in data])
            for data in self.split_into_groups(dataset, self.training_config.group_size)
        ]

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=f"{split}-rollout",
            max_exceptions=max_exceptions,
        )

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

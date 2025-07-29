from abc import abstractmethod
import os
import sys
import logging

from pydantic import BaseModel, Field
import art
from art.dev import InternalModelConfig
from art.utils import iterate_dataset, retry
import numpy as np

from src.vllm_client import VllmClient, VllmRouter
from src.models import PathConfig
from src.dac_agent import AgentNode, ChatMessage, PromptConfig, StopCriteria
from src.utils import text as text_utils


logger = logging.getLogger(__name__)


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
        client_list: list[VllmClient],
        path_config: PathConfig,
        train_config: TrainingConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
    ):
        self.model = model
        self.path_config = path_config
        self.training_config = train_config
        self.vllm_router = VllmRouter(vllm_clients=client_list)
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria

    @property
    def model_config(self) -> InternalModelConfig:
        config = self.model._internal_config
        if config is None:
            raise ValueError("Model configuration is not set.")
        return config

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

    async def evaluate(self, dataset: list[dict]) -> dict[str, float]:
        """
        Evaluate the model on a given dataset.

        Args:
            dataset (list[dict]): List of samples to evaluate.

        Returns:
            dict[str, float]: Dictionary containing average evaluation metrics.
        """

        if len(dataset) == 0:
            return {}

        # Rollout and log each trajectory
        eval_tasks = [self.rollout_step(sample) for sample in dataset]
        eval_trajectories = await art.gather_trajectories(eval_tasks, pbar_desc="eval", max_exceptions=0)
        await self.model.log(eval_trajectories, split="eval")

        # Aggregate metrics from all trajectories
        metrics = [tr.metrics for tr in eval_trajectories]
        n = len(metrics)
        keys = metrics[0].keys() if n > 0 else []
        eval_results = {key: sum(m[key] for m in metrics) / n for key in keys}
        return eval_results

    async def train(self, train_dataset: list[dict], eval_dataset: list[dict]) -> art.TrainableModel:
        model = self.model
        vllm_router = self.vllm_router
        train_config = self.training_config

        # Unload all lora adapters before starting the training
        # including adapters loaded in previous training sessions
        vllm_router.unload_all_loras()

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
            if train_config.eval_every and global_step % train_config.eval_every == 0:
                eval_batch = next(eval_iter)
                print(f"Evaluating model at train step {global_step} and eval step {eval_batch.step}...")
                eval_results = await self.evaluate(eval_batch.items)
                print(f"Evaluation results: {eval_results}")

            # Construct GRPO training groups
            train_groups = [
                art.TrajectoryGroup([self.rollout_step(sample) for sample in data])
                for data in self.split_into_groups(step_data, train_config.group_size)
            ]

            # Perform rollout
            trajectory_groups = await art.gather_trajectory_groups(
                train_groups,
                pbar_desc="gather",
                max_exceptions=0,
            )

            # Log rollout results
            if train_config.log_every and global_step % train_config.log_every == 0:
                await model.log(trajectory_groups, split="train")

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

            await model.delete_checkpoints("eval/reward")

            await model.train(
                trajectory_groups,
                config=train_config.art_config,
                _config=train_config.dev_art_config,
                verbose=train_config.verbose,
            )

        # Final evaluation after training
        eval_batch = next(eval_iter)
        print(f"Evaluating model at eval step {eval_batch.step}...")
        eval_results = await self.evaluate(eval_batch.items)
        print(f"Final evaluation results: {eval_results}")

        return self.model

    def split_into_groups(self, batch: list[dict], group_size: int) -> list[list[dict]]:
        return [batch[i : i + group_size] for i in range(0, len(batch), group_size)]

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

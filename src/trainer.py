from abc import abstractmethod
import warnings

from pydantic import BaseModel, Field
import art
from art.dev import InternalModelConfig
from art.utils import iterate_dataset
import numpy as np

from src.vllm_client import VllmClient, VllmRouter
from src.models import DirConfig
from src.dac_agent import AgentNode, ChatMessage, PromptConfig, StopCriteria
from src.utils import text as text_utils


class TrainingConfig(BaseModel, extra="allow", strict=True):
    epochs: int = 10
    num_groups: int = 5
    group_size: int = 10
    min_reward_stdev: float | None = None
    eval_steps: int | None = None
    verbose: bool = False

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
        dir_config: DirConfig,
        train_config: TrainingConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
    ):
        self.model = model
        self.dir_config = dir_config
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
        return self.vllm_router.__next__()

    def create_agent(self) -> AgentNode:
        """
        Create a new instance of the AgentNode with the current model and configuration.
        """
        client = self.get_client()
        return AgentNode(
            model_name=client.get_inference_name(),
            client=client.client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    def reload_lora(self, step: int) -> bool:
        prev_checkpoint_dir = None
        curr_checkpoint_dir = None

        if step > 0:
            curr_checkpoint_dir = self.dir_config.get_step_checkpoint_dir(step)

        if step - 1 > 0:
            prev_checkpoint_dir = self.dir_config.get_step_checkpoint_dir(step - 1)

        if prev_checkpoint_dir is not None:
            print("Unloading lora")
            if not self.vllm_router.unload_lora(self.dir_config.run_name):
                print(f"Failed to unload lora from {prev_checkpoint_dir}")
                return False

        if curr_checkpoint_dir is not None:
            print(f"Loading lora from {curr_checkpoint_dir}")
            if not self.vllm_router.load_lora(self.dir_config.run_name, curr_checkpoint_dir):
                print(f"Failed to load lora from {curr_checkpoint_dir}")
                return False

        return True

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
                warnings.warn(
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
            dict[str, float]: Dictionary containing evaluation metrics.
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
        vllm_router.unload_all_loras()

        batch_size = train_config.num_groups * train_config.group_size

        train_iter = iterate_dataset(
            dataset=train_dataset,
            groups_per_step=batch_size,
            num_epochs=train_config.epochs,
            initial_step=await model.get_step(),
            use_tqdm=True,
        )

        for step_data, epoch, global_step, epoch_step in train_iter:
            self.reload_lora(global_step)

            # Evaluate model
            if train_config.eval_steps and global_step % train_config.eval_steps == 0:
                eval_results = await self.evaluate(eval_dataset)
                print(f"Evaluation results at step {global_step}: {eval_results}")

            group_data = [
                step_data[i : i + train_config.group_size] for i in range(0, len(step_data), train_config.group_size)
            ]

            train_groups = [art.TrajectoryGroup([self.rollout_step(sample) for sample in data]) for data in group_data]

            trajectory_groups = await art.gather_trajectory_groups(train_groups, pbar_desc="gather", max_exceptions=0)

            filtered_groups = []
            for group in trajectory_groups:
                rewards = [tr.reward for tr in group.trajectories]
                if train_config.min_reward_stdev is None or np.std(rewards) >= train_config.min_reward_stdev:
                    filtered_groups.append(group)

            trajectory_groups = filtered_groups

            if not trajectory_groups:
                warnings.warn(f"No trajectories left to train on at step {global_step}. Skipping this step.")
                continue

            await model.delete_checkpoints("eval/reward")

            await model.train(
                trajectory_groups,
                config=train_config.art_config,
                _config=train_config.dev_art_config,
                verbose=train_config.verbose,
            )

        # Final evaluation after training
        eval_results = await self.evaluate(eval_dataset)
        print(f"Final evaluation results: {eval_results}")

        return self.model

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

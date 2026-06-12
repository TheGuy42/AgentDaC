import sys
import os
import torch
import pathlib
import argparse
import logging
from typing import Any, Tuple
from abc import ABC, abstractmethod
import random

import json
import pydantic
from datasets import Dataset
from datetime import datetime

from src.utils.env import prepare_environment, set_seed
from src.utils.logging import create_logger, setup_logging
from src.utils.io import load_object
from src.utils.dicts import get_dict_value, set_dict_value
from src.configs import TrainingConfig, PromptConfig, DecompConfig, RolloutConfig
from src.trainer import AglTrainer
from src.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


class ExperimentRunner(ABC):
    def __init__(self) -> None:
        self._parser_args = None

    def args(self) -> argparse.Namespace:
        if self._parser_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parser_args

    @abstractmethod
    def default_project_name(self) -> str:
        """Override to specify default project name."""
        pass

    @abstractmethod
    def default_config_dir(self) -> str:
        """Override to specify default config directory."""
        pass

    @abstractmethod
    def load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and return (train_dataset, val_dataset, test_data)."""
        pass

    @abstractmethod
    def create_trainer(self, **kwargs) -> AglTrainer:
        """Return the trainer class to use."""
        pass

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""
        pass

    def _create_trajectory_writer(self, configs: dict[str, Any]) -> TrajectoryWriter:
        """Build the trajectory writer."""
        args = self.args()
        exp_name, _ = get_dict_value(configs["verl_config"], "trainer", "experiment_name", raise_missing=True)
        output_dir = pathlib.Path(args.traj_dir or "trajectories") / args.project / exp_name

        if args.traj_dir is not None:
            logger.info(f"Logging full rollout trajectories to '{output_dir}'.")

        return TrajectoryWriter(output_dir, enabled=args.traj_dir is not None)

    def _generate_run_name(self, base_model: str) -> str:
        """
        Generate a run name based on the model name and current date.
        """
        base_model = base_model.split("/")[-1]
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        return f"{base_model}_{date_str}"

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--project",
            type=str,
            default=self.default_project_name(),
            help="The name of the project for saving results.",
        )

        parser.add_argument(
            "--run",
            type=str,
            default="",
            help="The name of the experiment run.",
        )

        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Whether to resume from a previous checkpoint. Provide the checkpoint path.",
        )

        parser.add_argument(
            "--gpus",
            type=int,
            nargs="+",
            default=[0],
            help=f"The ID of the GPU(s) to use (e.g., 0 or 0 1). Available GPUs: {list(range(torch.cuda.device_count()))}",
        )

        parser.add_argument(
            "--config_dir",
            type=str,
            default=self.default_config_dir(),
            help="Directory containing experiment configuration files.",
        )

        parser.add_argument(
            "--traj_dir",
            type=str,
            default=None,
            help=(
                "Base directory for logging full rollout trajectories to disk "
                "(one JSON file per rollout, grouped by training step). Disabled when not provided."
            ),
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility (default: random).",
        )

        parser.add_argument(
            "--silent",
            action="store_true",
            help="Disable verbose outputs.",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="Perform a quick test run with minimal training for debugging purposes.",
        )

        self.add_arguments(parser)
        self._parser_args = parser.parse_args()
        args = self.args()

        # verify valid GPU IDs
        if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpus):
            raise ValueError(f"Invalid GPU IDs provided: {args.gpus}. Available GPUs: {list(range(torch.cuda.device_count()))}")

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()

        return args

    def _load_configs(self, dir: str | pathlib.Path) -> dict[str, Any]:
        """Load all configuration files."""
        if isinstance(dir, str):
            dir = pathlib.Path(dir)

        return {
            "train_config": TrainingConfig.load_from_path(dir / "train_config.json", do_raise=True),
            "prompt_config": PromptConfig.load_from_path(dir / "prompt_config.json", do_raise=True),
            "decomp_config": DecompConfig.load_from_path(dir / "decomp_config.json", do_raise=True),
            "rollout_config": RolloutConfig.load_from_path(dir / "rollout_config.json", do_raise=True),
            "verl_config": load_object(dir / "verl_config.json", do_raise=True),
            "extra_config": load_object(dir / "extra_config.json", do_raise=False),
        }

    def _update_configs_test(self, configs: dict[str, Any]):
        train_config = configs["train_config"]
        train_config.n_runners = 1
        train_config.train_size = 20
        train_config.val_size = 10

        verl_config = configs["verl_config"]
        set_dict_value(verl_config, "trainer", "logger", value=["console"])
        set_dict_value(verl_config, "trainer", "nnodes", value=1)
        set_dict_value(verl_config, "trainer", "n_gpus_per_node", value=1)
        set_dict_value(verl_config, "trainer", "test_freq", value=1)
        set_dict_value(verl_config, "trainer", "save_freq", value=-1)
        set_dict_value(verl_config, "trainer", "total_epochs", value=1)
        set_dict_value(verl_config, "trainer", "total_training_steps", value=2)
        set_dict_value(verl_config, "data", "train_batch_size", value=6)
        set_dict_value(verl_config, "actor_rollout_ref", "rollout", "n", value=2)
        set_dict_value(verl_config, "actor_rollout_ref", "actor", "ppo_mini_batch_size", value=2)
        set_dict_value(verl_config, "actor_rollout_ref", "actor", "ppo_micro_batch_size_per_gpu", value=2)
        set_dict_value(verl_config, "actor_rollout_ref", "ref", "log_prob_micro_batch_size_per_gpu", value=2)

    def _patch_lengths(self, configs: dict[str, Any]) -> None:
        """
        Patch the prompt and response lengths in the VERL config to ensure they fit within the model's context window.
        Utilizes `DecompConfig` to compute a conservative upper bound on the maximum cumulative response length of a trajectory.
        """

        # NOTE: Index:
        # - agentlightning.trace_aggregator.trajectory_max_prompt_length is the length of the initial prompt fed to the model, which includes the user question and the system instructions
        # - agentlightning.trace_aggregator.trajectory_max_response_length is the cumulative length of the rest of the conversation without the initial prompt over the entire trajectory rollout, which includes all model responses and tool responses.
        # - data.max_response_length is the length of single model response. Controls vllm max_new_tokens in SamplingParams
        # - data.max_prompt_length is the maximum length of the prompt fed to the model during any stage of the trajectory rollout, which includes the initial prompt and the conversation history.
        # - actor_rollout_ref.rollout.max_model_len is the maximum overall length of the entire trajectory

        verl_config = configs["verl_config"]
        decomp_config: DecompConfig = configs["decomp_config"]

        # Compute the maximum possible cumulative response length of a trajectory based on the number of rounds and tasks,
        # assuming the worst case where every task is decomposed until the max rounds, and every model response is as long as the max_response_length.
        # This is a conservative upper bound to ensure we never exceed the model context window, even in edge cases.
        sing_resp_len, _ = get_dict_value(verl_config, "data", "max_response_length", raise_missing=True)
        traj_resp_len = decomp_config.max_tasks * (2 * sing_resp_len) + (decomp_config.max_rounds - decomp_config.max_tasks) * sing_resp_len
        traj_resp_len += 32 * decomp_config.max_rounds  # Add extra buffer chat template

        inp_len = 1024  # Max length of the input first system + user prompt
        if get_dict_value(verl_config, "agentlightning", "trace_aggregator", "level") == "trajectory":
            inp_len, _ = get_dict_value(verl_config, "agentlightning", "trace_aggregator", "trajectory_max_prompt_length", raise_missing=True)

        model_len = inp_len + traj_resp_len # Overall model context length needed to fit the entire trajectory
        prompt_len = model_len - sing_resp_len

        set_dict_value(verl_config, "agentlightning", "trace_aggregator", "trajectory_max_response_length", value=traj_resp_len)
        set_dict_value(verl_config, "data", "max_prompt_length", value=prompt_len)
        set_dict_value(verl_config, "actor_rollout_ref", "rollout", "max_model_len", value=model_len)

        logger.info(f"Setting `agentlightning.trace_aggregator.trajectory_max_response_length` to {traj_resp_len}")
        logger.info(f"Setting `data.max_prompt_length` to {prompt_len}")
        logger.info(f"Setting `actor_rollout_ref.rollout.max_model_len` to {model_len}")

    def _patch_configs(self, configs: dict[str, Any]) -> dict[str, Any]:
        verl_config = configs["verl_config"]
        
        model_name, _ = get_dict_value(verl_config, "actor_rollout_ref", "model", "path", raise_missing=True)
        exp_name = self.args().run or self._generate_run_name(model_name)
        
        logger.info(f"Experiment name set to '{exp_name}'")
        set_dict_value(verl_config, "trainer", "project_name", value=self.args().project)
        set_dict_value(verl_config, "trainer", "experiment_name", value=exp_name)

        logger.info(f"Setting VERL seed to {self.args().seed}")
        set_dict_value(verl_config, "data", "seed", value=self.args().seed)

        self._patch_lengths(configs)
            
        if resume_path := self.args().resume:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            set_dict_value(verl_config, "trainer", "resume_mode", value="resume_path")
            set_dict_value(verl_config, "trainer", "resume_from_path", value=resume_path)

        if self.args().test_run:
            logger.info("Test run enabled: Overriding configs for a quick test run.")
            self._update_configs_test(configs)

        return configs

    def _print_configs(self, configs: dict[str, Any]) -> None:
        for cfg_name, cfg_obj in configs.items():
            try:
                if isinstance(cfg_obj, (pydantic.BaseModel)):
                    cfg_str = cfg_obj.model_dump_json(indent=2)
                else:
                    cfg_str = json.dumps(cfg_obj, indent=2)
            except Exception:
                cfg_str = str(cfg_obj)

            logger.info(f"Configuration for {cfg_name} ({type(cfg_obj).__name__}): {cfg_str}")

    def _main(self) -> None:
        """Main experiment execution logic."""

        args = self.args()
        logger.info(f"Current working directory: {os.getcwd()}")

        # Set random seed
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

        # Set the GPU environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

        # Load configurations
        configs = self._load_configs(args.config_dir)
        configs = self._patch_configs(configs)
        train_config: TrainingConfig = configs["train_config"]

        # Print all configurations
        if not args.silent:
            self._print_configs(configs)

        # Load dataset
        logger.info("Loading data...")
        train_dataset, val_dataset, test_dataset = self.load_data()
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        train_dataset = train_dataset.shuffle(0).to_list()
        val_dataset = val_dataset.shuffle(1).to_list()
        test_dataset = test_dataset.shuffle(2).to_list()

        if train_config.train_size is not None:
            train_dataset = train_dataset[: train_config.train_size]
            logger.info(f"Truncated train dataset to size: {len(train_dataset)}")

        if train_config.val_size is not None:
            val_dataset = val_dataset[: train_config.val_size]
            logger.info(f"Truncated val dataset to size: {len(val_dataset)}")

        if train_config.test_size is not None:
            test_dataset = test_dataset[: train_config.test_size]
            logger.info(f"Truncated test dataset to size: {len(test_dataset)}")

        # Create and configure the trainer
        writer = self._create_trajectory_writer(configs)
        trainer = self.create_trainer(**configs, trajectory_writer=writer)

        # Start training
        logger.info("Starting training...")
        trainer.train(train_dataset=train_dataset, val_dataset=val_dataset)

    def run(self) -> None:
        """Entry point to run the experiment."""
        try:
            self._parse_args()
            prepare_environment()
            setup_logging(level=logging.WARNING if self.args().silent else logging.INFO)
            self._main()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)

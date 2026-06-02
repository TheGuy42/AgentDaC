import sys
import os
import torch
import pathlib
import asyncio
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
from src.configs import TrainingConfig, PromptConfig, DecompConfig, RolloutConfig
from src.trainer import AglTrainer


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
            "--dev",
            action="store_true",
            help="Perform a quick development run.",
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
            "extra_config": load_object(dir / "extra_config.json", do_raise=False),
        }

    def _patch_configs(self, configs: dict[str, Any]) -> dict[str, Any]:
        rollout_config: RolloutConfig = configs["rollout_config"]
        train_config: TrainingConfig = configs["train_config"]
        verl_config = train_config.verl_config

        model_name = verl_config["actor_rollout_ref"]["model"]["path"]
        exp_name = self.args().run or self._generate_run_name(model_name)

        verl_config.setdefault("trainer", {})["project_name"] = self.args().project
        verl_config.setdefault("trainer", {})["experiment_name"] = exp_name

        if "Qwen3" in model_name:
            # disable "thinking" for Qwen3 models
            logger.info("Disabling 'thinking' for Qwen3 model.")
            rollout_config.kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
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
        train_config: TrainingConfig = configs.pop("train_config")

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

        if train_config.val_size is not None:
            test_dataset = test_dataset[: train_config.val_size]  # TODO: create separate config entry test_size
            logger.info(f"Truncated test dataset to size: {len(test_dataset)}")

        # Create and configure the trainer
        trainer = self.create_trainer(**configs)

        if not args.dev:
            # Start training
            logger.info("Starting training...")
            trainer.train(
                config=train_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

        else:
            # Perform a development run
            logger.info("Starting development run...")
            trainer.dev(
                config=train_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

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

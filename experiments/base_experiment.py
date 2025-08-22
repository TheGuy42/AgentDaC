import sys
import os
import torch
import pathlib
import asyncio
import argparse
import logging
from typing import Any, Tuple
from abc import ABC, abstractmethod

from datasets import Dataset
from art import TrainableModel

from src.utils.env import prepare_environment
from src.utils.logging import create_logger, setup_logging
from src.utils.io import load_base_model
from src.utils.loaders import load_art_model
from src.vllm_client import VllmClient, ArtClient, VllmRouter
from src.configs.models.art import available_configs, ArtConfig
from src.configs import PathConfig, TrainingConfig, PromptConfig, StopCriteria, RolloutConfig
from src.trainer import Trainer


logger = create_logger(__name__)


class BaseExperiment(ABC):
    def __init__(self, verbose: bool = True) -> None:
        self.model_name = ""
        self.verbose = verbose

    @abstractmethod
    def get_default_project_name(self) -> str:
        """Override to specify default project name."""
        pass

    @abstractmethod
    def get_default_config_dir(self) -> str:
        """Override to specify default config directory."""
        pass

    @abstractmethod
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Load and return (train_dataset, val_dataset)."""
        pass

    @abstractmethod
    def create_trainer(self, model: TrainableModel, vllm_router: VllmRouter, **cfgs) -> Trainer:
        """Return the trainer class to use."""
        pass

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help=f"The name or path of the model to serve. Available models are: {available_configs()}",
        )

        parser.add_argument(
            "--project",
            type=str,
            default=self.get_default_project_name(),
            help="The name of the project for saving results.",
        )

        parser.add_argument(
            "--run",
            type=str,
            default="",
            help="The name of the run to continue (if exists).",
        )

        parser.add_argument(
            "--gpu",
            type=int,
            nargs="+",
            default=[0],
            help=f"The ID of the GPU(s) to use (e.g., 0 or 0 1). Available GPUs: {list(range(torch.cuda.device_count()))}",
        )

        parser.add_argument(
            "--vllm_ports",
            type=int,
            nargs="*",
            default=[],
            help="List of endpoint ports for vLLM servers.",
        )

        parser.add_argument(
            "--config_dir",
            type=str,
            default=self.get_default_config_dir(),
            help="Directory containing experiment configuration files.",
        )

        parser.add_argument(
            "--silent",
            action="store_true",
            help="Disable verbose outputs.",
        )

        args = parser.parse_args()

        # store state params for later use
        self.model_name = args.model
        self.verbose = not args.silent if hasattr(args, "silent") else True

        # verify valid GPU IDs
        if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpu):
            raise ValueError(
                f"Invalid GPU IDs provided: {args.gpu}. Available GPUs: {list(range(torch.cuda.device_count()))}"
            )

        # print the parsed arguments
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

        return args

    def load_configs(self, config_dir: str | pathlib.Path) -> dict[str, Any]:
        """Load all configuration files."""
        if isinstance(config_dir, str):
            config_dir = pathlib.Path(config_dir)

        configs = {
            "art_config": load_base_model(ArtConfig, config_dir / "art_config.json", do_raise=False),
            "train_config": load_base_model(TrainingConfig, config_dir / "train_config.json", do_raise=True),
            "prompt_config": load_base_model(PromptConfig, config_dir / "prompt_config.json", do_raise=True),
            "stop_criteria": load_base_model(StopCriteria, config_dir / "stop_criteria.json", do_raise=True),
            "rollout_config": load_base_model(RolloutConfig, config_dir / "rollout_config.json", do_raise=True),
        }

        if "Qwen3" in self.model_name:
            rollout_config = configs["rollout_config"]
            rollout_config.kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        return configs

    def create_inference_clients(self, model: TrainableModel, vllm_ports: list[int]) -> VllmRouter:
        """Create and configure inference clients."""
        inference_clients = [ArtClient.from_art_model(model)]
        for port in vllm_ports:
            inference_clients.append(
                VllmClient.from_connection(
                    port=port,
                    base_model=model.base_model,
                    model_name=model.get_inference_name(),
                )
            )
        return VllmRouter(inference_clients)

    async def main(self, args: argparse.Namespace) -> None:
        """Main experiment execution logic."""
        # Set the GPU environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))

        print()
        print(f"Current working directory: {os.getcwd()}")
        print()

        path_config = PathConfig(
            base_model=args.model,
            project_name=args.project,
            run_name=args.run,
        )

        # Load configurations
        config_dict = self.load_configs(args.config_dir)
        art_config: ArtConfig | None = config_dict["art_config"]
        train_config: TrainingConfig = config_dict["train_config"]

        # Print all configurations
        if self.verbose:
            print("\nLoaded configurations:")
            for key, config in config_dict.items():
                print(f"\n{key}:")
                print(config)

        # Load model
        model = await load_art_model(
            path_config=path_config,
            art_config=art_config,
            print_full=self.verbose,
        )

        # Load dataset
        train_dataset, val_dataset = self.load_data()

        # Create inference clients
        vllm_router = self.create_inference_clients(model, args.vllm_ports)

        # Create and configure the trainer
        trainer = self.create_trainer(
            model=model,
            vllm_router=vllm_router,
            **config_dict,
        )

        # Log code files
        if trainer.wandb_run is not None:
            trainer.wandb_run.log_code(root="src", name="src")
            trainer.wandb_run.log_code(root="scripts", name="scripts")
            trainer.wandb_run.log_code(root="experiments", name="experiments")

        # Start training
        try:
            await trainer.train(
                config=train_config,
                train_dataset=train_dataset.to_list(),
                val_dataset=val_dataset.to_list(),
            )
        finally:
            trainer.close()

    def run(self) -> None:
        """Entry point to run the experiment."""
        try:
            prepare_environment()
            setup_logging(level=logging.INFO if self.verbose else logging.WARNING)
            args = self.parse_args()
            asyncio.run(self.main(args))
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)

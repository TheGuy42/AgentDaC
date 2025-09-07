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
import art

from src.utils.rng import set_seed
from src.utils.env import prepare_environment
from src.utils.logging import create_logger, setup_logging
from src.utils.io import load_base_model, load_object
from src.utils.loaders import load_art_model
from src.vllm_client import VllmClient, ArtClient, VllmRouter
from src.configs.models.art import available_configs, ArtConfig
from src.configs import PathConfig, TrainingConfig, PromptConfig, DecompConfig, RolloutConfig
from src.trainer import Trainer, RolloutStage


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
    def create_trainer(
        self,
        model: art.Model,
        **kwargs,
    ) -> Trainer:
        """Return the trainer class to use."""
        pass

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""
        pass

    def _parse_args(self) -> argparse.Namespace:
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
            default=self.default_project_name(),
            help="The name of the project for saving results.",
        )

        parser.add_argument(
            "--run",
            type=str,
            default="",
            help="The name of the run to continue (if exists).",
        )

        parser.add_argument(
            "--gpus",
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
            "--eval",
            action="store_true",
            help="Run evaluation only of base model (skip training).",
        )

        self.add_arguments(parser)
        self._parser_args = parser.parse_args()
        args = self.args()

        # verify valid GPU IDs
        if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpus):
            raise ValueError(
                f"Invalid GPU IDs provided: {args.gpus}. Available GPUs: {list(range(torch.cuda.device_count()))}"
            )

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
            "art_config": load_base_model(ArtConfig, dir / "art_config.json", do_raise=False),
            "train_config": load_base_model(TrainingConfig, dir / "train_config.json", do_raise=True),
            "prompt_config": load_base_model(PromptConfig, dir / "prompt_config.json", do_raise=True),
            "decomp_config": load_base_model(DecompConfig, dir / "decomp_config.json", do_raise=True),
            "rollout_config": load_base_model(RolloutConfig, dir / "rollout_config.json", do_raise=True),
            "extra_config": load_object(dir / "extra_config.json", do_raise=False),
        }

    def _patch_configs(self, configs: dict[str, Any]) -> dict[str, Any]:
        rollout_config = configs.get("rollout_config")
        if ("Qwen3" in self.args().model) and rollout_config:
            # disable "thinking" for Qwen3 models
            logger.info("Disabling 'thinking' for Qwen3 model.")
            rollout_config.kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return configs

    def _create_inference_clients(self, model: art.Model, vllm_ports: list[int]) -> VllmRouter:
        """Create and configure inference clients."""
        art_client = ArtClient.from_art_model(model)
        inference_clients = [art_client]
        for port in vllm_ports:
            inference_clients.append(
                VllmClient.from_connection(
                    port=port,
                    api_key=art_client.api_key,
                    base_model=art_client.base_model,
                    model_name=art_client.model_name,
                )
            )
        return VllmRouter(inference_clients)

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

    async def _main(self) -> None:
        """Main experiment execution logic."""

        args = self.args()
        logger.info(f"Current working directory: {os.getcwd()}")

        # Set random seed
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

        # Set the GPU environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

        path_config = PathConfig(
            base_model=args.model,
            project_name=args.project,
            run_name=args.run,
        )

        # Load configurations
        configs = self._load_configs(args.config_dir)
        configs["path_config"] = path_config
        configs = self._patch_configs(configs)

        art_config: ArtConfig | None = configs["art_config"]
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

        if train_config.val_size is not None:
            test_dataset = test_dataset[: train_config.val_size] # TODO: create separate config entry test_size
            logger.info(f"Truncated test dataset to size: {len(test_dataset)}")

        # Load model
        model = await load_art_model(
            path_config=path_config,
            art_config=art_config,
            seed=args.seed,
        )

        # If eval only, create a non-trainable version of the model
        if args.eval:
            eval_model = art.Model(
                name=f"{model.name}_eval",
                project=model.project,
                inference_api_key=model.inference_api_key,
                inference_base_url=model.inference_base_url,
                inference_model_name=model.base_model,
            )
            await eval_model.register(model.backend())
            model = eval_model

        # Create inference clients
        vllm_router = self._create_inference_clients(model, args.vllm_ports)

        # Create and configure the trainer
        trainer = self.create_trainer(
            model=model,
            vllm_router=vllm_router,
            **configs,
        )

        # Log code files
        if trainer.wandb_run is not None:
            trainer.wandb_run.log_code(root="src", name="src")
            trainer.wandb_run.log_code(root="scripts", name="scripts")
            trainer.wandb_run.log_code(root="experiments", name="experiments")

        try:
            if not args.eval:
                # Run training if applicable
                logger.info("Starting training...")
                await trainer.train(
                    config=train_config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                )

            # Run evaluation
            logger.info("Starting val-set evaluation...")
            groups = await trainer.rollout(
                dataset=val_dataset,
                group_size=1,
                stage=RolloutStage.VAL,
                max_exceptions=train_config.max_exceptions,
            )
            await trainer.model.log(groups, split=RolloutStage.VAL.value)

            logger.info("Starting test-set evaluation...")
            groups = await trainer.rollout(
                dataset=test_dataset,
                group_size=1,
                stage=RolloutStage.TEST,
                max_exceptions=train_config.max_exceptions,
            )
            await trainer.model.log(groups, split=RolloutStage.TEST.value)

        finally:
            await trainer.close()
            await vllm_router.close()

    def run(self) -> None:
        """Entry point to run the experiment."""
        try:
            self._parse_args()
            prepare_environment()
            setup_logging(level=logging.WARNING if self.args().silent else logging.INFO)
            asyncio.run(self._main())
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)

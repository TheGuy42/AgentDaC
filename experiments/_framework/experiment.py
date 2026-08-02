from __future__ import annotations
import argparse
import logging
import os
import pathlib
import random
import sys
from typing import Any, Sequence

import torch
from src.running.dataset import TaskDataset
from src.running.rollout import RolloutTask
from src.utils.env import prepare_environment, set_seed
from src.utils.logging import create_logger, setup_logging, parse_log_level
from experiments._framework.backends.backend import BackendArgs
from experiments._framework.backends.registry import BackendName, create_backend

logger = create_logger(__name__)


class Experiment:
    """Parses the CLI, assembles the configs, and dispatches to a backend."""

    def __init__(
        self,
        task_name: str,
        supported_agents: Sequence[str],
        task_cls: type[RolloutTask],
        dataset_cls: type[TaskDataset],
    ) -> None:
        self.task_name = task_name
        self.supported_agents = list(supported_agents)
        self.task_cls = task_cls
        self.dataset_cls = dataset_cls

    def task_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        """Load task-specific configs. Override this in subclasses to add more configs."""
        return {}

    def default_project_name(self, agent: str) -> str:
        return f"{self.task_name}_{agent}"

    def default_config_dir(self, agent: str) -> str:
        return f"experiments/{self.task_name}/configs/{agent}"

    def _parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--agent",
            type=str,
            required=True,
            help="The agent kind to run",
            choices=self.supported_agents,
        )

        parser.add_argument(
            "--backend",
            type=str,
            default=None,
            choices=[k.value for k in BackendName],
            help="Training backend. If not provided, detected from what is installed.",
        )

        parser.add_argument(
            "--project",
            type=str,
            default=None,
            help="Project name. If not provided, defaults to `<experiment>_<agent>`.",
        )

        parser.add_argument(
            "--run",
            type=str,
            default=None,
            help="Experiment run name. If not provided, the backend derives it from the base model.",
        )

        parser.add_argument(
            "--gpus",
            type=int,
            nargs="+",
            default=[0],
            help="GPU IDs to use.",
        )

        parser.add_argument(
            "--config_dir",
            type=str,
            default=None,
            help="Config directory. If not provided, defaults to `experiments/<experiment>/configs/<agent>`.",
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
            help="Random seed.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=logging.getLevelNamesMapping().keys(),
            default="INFO",
            help="Logging level.",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="Quick minimal run for debugging.",
        )

        args = parser.parse_args(argv)

        if args.project is None:
            args.project = self.default_project_name(args.agent)

        if args.config_dir is None:
            args.config_dir = self.default_config_dir(args.agent)

        if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpus):
            raise ValueError(f"Invalid GPU IDs {args.gpus}. Available: {list(range(torch.cuda.device_count()))}")

        print("\nParsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()
        return args

    def run(self, argv: list[str] | None = None) -> None:
        try:
            args = self._parse_args(argv)

            # Prepare the environment
            prepare_environment()
            setup_logging(level=parse_log_level(args.log_level))
            set_seed(args.seed)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

            config_dir = pathlib.Path(args.config_dir)
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Config directory: {config_dir.resolve()}")
            logger.info(f"Backend selection: {args.backend or 'auto-detect'}")

            backend = create_backend(
                BackendArgs(
                    experiment=self.task_name,
                    agent_name=args.agent,
                    task_cls=self.task_cls,
                    dataset_cls=self.dataset_cls,
                    args=args,
                    config_dir=config_dir,
                ),
                kind=args.backend,
            )

            # Prepare configs: backend + task-specific
            configs = {
                **backend.load_configs(config_dir),
                **self.task_configs(config_dir),
            }

            # Launch backend
            backend.launch(configs)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)

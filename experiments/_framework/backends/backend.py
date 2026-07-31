from __future__ import annotations

import argparse
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.running.dataset import TaskDataset
from src.running.rollout import RolloutTask


@dataclass(frozen=True)
class BackendArgs:
    """Everything the runner hands to a backend."""

    experiment: str
    agent_name: str
    task_cls: type[RolloutTask]
    dataset_cls: type[TaskDataset]
    args: argparse.Namespace
    config_dir: pathlib.Path


class Backend(ABC):
    def __init__(self, backend_args: BackendArgs) -> None:
        self.args = backend_args

    @property
    @abstractmethod
    def name(self) -> str:
        """The backend's name, used for logging and config paths."""

    @abstractmethod
    def load_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        """This backend's own config files, read from `self.backend_config.config_dir`."""

    @abstractmethod
    def launch(self, configs: dict[str, Any]) -> None:
        """Run the experiment against the fully-assembled configs."""

    def default_run_name(self, base_model: str) -> str:
        """`<model>_<MM_DD_HH_MM>`, so run names are uniform across backends."""
        base_model = base_model.split("/")[-1]
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        return f"{base_model}_{date_str}"

    def traj_writer_config(self, exp_name: str) -> dict[str, Any]:
        """Where this run's trajectories go. Kept as a dict because verl embeds it in its config
        and constructs the writer inside the Ray worker."""
        args = self.args.args
        return {
            "dir": (pathlib.Path(args.traj_dir or "trajectories") / self.name / args.project / exp_name).as_posix(),
            "enabled": args.traj_dir is not None,
        }

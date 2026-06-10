from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

from datasets import Dataset

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.trainer import AglTrainer
from src.utils.logging import create_logger
from experiments.experiment_runner import ExperimentRunner
from experiments.chess_perst.trainer import ChessTrainer
from experiments.chess_perst.chess_engine import EngineConfig
from experiments.chess_perst.data import load_dataset


logger = create_logger(__name__)


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "chess_perst_dac"

    def default_config_dir(self) -> str:
        return "experiments/chess_perst/defaults"

    def _load_configs(self, dir: str | pathlib.Path) -> dict[str, Any]:
        """Load the shared configs plus the chess-specific engine config."""
        configs = super()._load_configs(dir)
        configs["engine_config"] = EngineConfig.load_from_path(pathlib.Path(dir) / "engine_config.json", do_raise=True)
        return configs

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--position_source", choices=["random", "engine"], default="random", help="How positions are generated.")
        parser.add_argument("--num_train", type=int, default=2000, help="Number of training positions to generate.")
        parser.add_argument("--num_val", type=int, default=200, help="Number of validation positions to generate.")
        parser.add_argument("--min_ply", type=int, default=8, help="Minimum plies played before sampling a position.")
        parser.add_argument("--max_ply", type=int, default=40, help="Maximum plies played before sampling a position.")
        parser.add_argument("--data_seed", type=int, default=1234, help="Seed for reproducible position generation.")

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        args = self.args()
        engine_config = EngineConfig.load_from_path(pathlib.Path(args.config_dir) / "engine_config.json", do_raise=True)
        common = dict(source=args.position_source, min_ply=args.min_ply, max_ply=args.max_ply, engine_config=engine_config)

        train = load_dataset("train", num=args.num_train, seed=args.data_seed, **common)
        val = load_dataset("val", num=args.num_val, seed=args.data_seed + 1, **common)
        return train, val, val

    def create_trainer(self, **kwargs) -> AglTrainer:
        return ChessTrainer(**kwargs)


if __name__ == "__main__":
    Runner().run()

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils.logging import create_logger
from experiments.experiment_runner import ExperimentRunner
from experiments.chess_perst.dataset import ChessPerstDataset
from experiments.chess_perst.trainer import ChessTrainer
from experiments.chess_perst.chess_engine import EngineConfig
from experiments.chess_perst.data import ChessDataset, SUPPORTED_DATASETS


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
        parser.add_argument(
            "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=[ChessDataset.PUZZLES.value],
            help="Which chess dataset(s) to load positions from (pooled when more than one).",
        )
        parser.add_argument("--num_train", type=int, default=2000, help="Number of training positions.")
        parser.add_argument("--num_val", type=int, default=200, help="Number of validation positions.")
        parser.add_argument("--data_seed", type=int, default=1234, help="Seed for reproducible data loading/shuffling.")
        parser.add_argument("--min_rating", type=int, default=None, help="Minimum puzzle rating (lichess-puzzles only).")
        parser.add_argument("--max_rating", type=int, default=None, help="Maximum puzzle rating (lichess-puzzles only).")

    def dataset_class(self) -> type:
        return ChessPerstDataset

    def trainer_class(self) -> type:
        return ChessTrainer

    def dataset_args(self) -> dict[str, Any]:
        args = self.args()
        return {
            "datasets": args.datasets,
            "num_train": args.num_train,
            "num_val": args.num_val,
            "data_seed": args.data_seed,
            "min_rating": args.min_rating,
            "max_rating": args.max_rating,
        }


if __name__ == "__main__":
    Runner().run()

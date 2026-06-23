from argparse import ArgumentParser
import sys
import pathlib
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments.experiment_runner import ExperimentRunner
from experiments.easy2hard.dataset import Easy2HardDataset
from experiments.easy2hard.trainer import Easy2HardTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "easy2hard_dac"

    def default_config_dir(self) -> str:
        return "experiments/easy2hard/defaults"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--min_difficulty", type=int, default=0)
        parser.add_argument("--max_difficulty", type=int, default=100)

    def dataset_class(self) -> type:
        return Easy2HardDataset

    def trainer_class(self) -> type:
        return Easy2HardTrainer

    def dataset_args(self) -> dict[str, Any]:
        return {"min_difficulty": self.args().min_difficulty, "max_difficulty": self.args().max_difficulty}


if __name__ == "__main__":
    Runner().run()

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.math.dataset import MathDataset
from experiments.math.trainer import MathTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "math_dac"

    def default_config_dir(self) -> str:
        return "experiments/math/defaults"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--min_level", type=int, default=1)
        parser.add_argument("--max_level", type=int, default=5)

    def dataset_class(self) -> type:
        return MathDataset

    def trainer_class(self) -> type:
        return MathTrainer

    def dataset_args(self) -> dict[str, Any]:
        return {"min_level": self.args().min_level, "max_level": self.args().max_level}


if __name__ == "__main__":
    Runner().run()

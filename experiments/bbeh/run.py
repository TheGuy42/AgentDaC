from argparse import ArgumentParser
import sys
import pathlib
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.bbeh.dataset import BbehDataset
from experiments.bbeh.trainer import BbehTrainer
from experiments.bbeh.tasks import SupportedTasks


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "bbeh_dac"

    def default_config_dir(self) -> str:
        return "experiments/bbeh/defaults"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--tasks",
            type=str,
            nargs="+",
            default=None,
            required=True,
            help=f"Which tasks of the BBEH dataset to use. Available tasks: {SupportedTasks.list_values()}",
        )

    def dataset_class(self) -> type:
        return BbehDataset

    def trainer_class(self) -> type:
        return BbehTrainer

    def dataset_args(self) -> dict[str, Any]:
        return {"tasks": self.args().tasks}


if __name__ == "__main__":
    Runner().run()

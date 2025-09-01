from argparse import ArgumentParser
import sys
import pathlib

import art
from datasets import Dataset, load_dataset, DatasetDict

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.bbeh.trainer import BbehTrainer
from experiments.bbeh.tasks import SupportedTasks
from src.trainer import Trainer


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

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        data: Dataset = load_dataset(
            path="BBEH/bbeh",
            split="train",
        )  # type: ignore

        # Filter by tasks
        data = data.map(lambda sample: {"task": sample["task"].replace(" ", "_")})
        data = data.filter(lambda sample: sample["task"] in self.args().tasks)

        split_dict = data.train_test_split(test_size=0.25, seed=0)
        ds_train = split_dict["train"]
        ds_eval = split_dict["test"]
        return ds_train, ds_eval, ds_eval

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer:
        return BbehTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

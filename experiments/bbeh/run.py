from argparse import ArgumentParser
import sys
import pathlib

from art import TrainableModel
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
            choices=SupportedTasks.list_values(),
            help=f"Which tasks of the BBEH dataset to use. Available tasks: {SupportedTasks.list_values()}",
        )

    def load_data(self) -> tuple[Dataset, Dataset]:
        data: Dataset = load_dataset(
            path="BBEH/bbeh",
            split="train",
        )  # type: ignore

        split_dict = data.train_test_split(test_size=0.25, seed=0)
        ds_train = split_dict["train"]
        ds_eval = split_dict["test"]

        # format task-names to match dataset entries
        tasks: list[str] = self.args().tasks
        tasks = [t.replace("_", " ") for t in tasks]

        ds_train = ds_train.filter(lambda sample: sample["task"] in tasks)
        ds_eval = ds_eval.filter(lambda sample: sample["task"] in tasks)
        return ds_train, ds_eval

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer:
        return BbehTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

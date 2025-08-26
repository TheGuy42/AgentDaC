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
from experiments.easy2hard.trainer import Easy2HardTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "easy2hard_dac"

    def default_config_dir(self) -> str:
        return "experiments/easy2hard/defaults"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--min_difficulty",
            type=int,
            default=0,
        )

    def load_data(self) -> tuple[Dataset, Dataset]:
        dataset_dict: DatasetDict = load_dataset(
            path="furonghuang-lab/Easy2Hard-Bench",
            name="E2H-AMC",
            split=None,
        )  # type: ignore

        # NOTE: validation set is larger, so use it for training
        ds_train: Dataset = dataset_dict["eval"]
        ds_val: Dataset = dataset_dict["train"]

        # filter by difficulty
        difficulty = self.args().min_difficulty
        ds_train = ds_train.filter(lambda sample: sample["item_difficulty"] >= difficulty)
        ds_val = ds_val.filter(lambda sample: sample["item_difficulty"] >= difficulty)

        return ds_train, ds_val

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer:
        return Easy2HardTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

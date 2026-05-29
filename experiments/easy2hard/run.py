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
from experiments.easy2hard.trainer import Easy2HardTrainer
from src.trainer import ArtTrainer


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
        
        parser.add_argument(
            "--max_difficulty",
            type=int,
            default=100,
        )

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset_dict: DatasetDict = load_dataset(
            path="furonghuang-lab/Easy2Hard-Bench",
            name="E2H-AMC",
            split=None,
        )  # type: ignore

        # NOTE: validation set is larger, so use it for training
        ds_train: Dataset = dataset_dict["eval"]
        ds_val: Dataset = dataset_dict["train"]

        # filter by difficulty
        min_dif = self.args().min_difficulty
        max_dif = self.args().max_difficulty
        ds_train = ds_train.filter(lambda sample: max_dif >= sample["item_difficulty"] >= min_dif)
        ds_val = ds_val.filter(lambda sample: max_dif >= sample["item_difficulty"] >= min_dif)
        return ds_train, ds_val, ds_val

    def create_trainer(self, model: art.Model, **kwargs) -> ArtTrainer:
        return Easy2HardTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

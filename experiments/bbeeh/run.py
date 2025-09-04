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
from experiments.bbeeh.create_dataset import load_dataset_as_datasetdict
from experiments.bbeeh.trainer import BbeehTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "bbeeh_dac"

    def default_config_dir(self) -> str:
        return "experiments/bbeeh/defaults"

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        
        dataset_dict: DatasetDict = load_dataset_as_datasetdict(
            dataset_dir="experiments/bbeeh/data",
            dataset_name="bbeh_boolean_expressions"
        )  # type: ignore

        ds_train = dataset_dict["train"]
        ds_val = dataset_dict["val"]
        ds_test = dataset_dict["test"]
        return ds_train, ds_val, ds_test

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer:
        return BbeehTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

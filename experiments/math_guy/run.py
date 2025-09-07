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
from experiments.math_guy.trainer import MathTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "math_dac"

    def default_config_dir(self) -> str:
        return "experiments/math_guy/defaults"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--min_level",
            type=int,
            default=1,
        )
        
        parser.add_argument(
            "--max_level",
            type=int,
            default=5,
        )

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset_dict: DatasetDict = load_dataset(
            path="nlile/hendrycks-MATH-benchmark",
            split=None,
        )  # type: ignore

        ds_train: Dataset = dataset_dict["train"]
        ds_val: Dataset = dataset_dict["test"]
        
        # filter by difficulty
        min_level = self.args().min_level
        max_level = self.args().max_level
        ds_train = ds_train.filter(lambda sample: max_level >= sample["level"] >= min_level)
        ds_val = ds_val.filter(lambda sample: max_level >= sample["level"] >= min_level)
        
        ds_train = ds_train.add_column("index", list(range(len(ds_train))), new_fingerprint=ds_train._fingerprint+'_index')
        ds_val = ds_val.add_column("index", list(range(len(ds_val))), new_fingerprint=ds_val._fingerprint+'_index')

        return ds_train, ds_val, ds_val

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer: 
        return MathTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

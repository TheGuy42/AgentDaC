import sys
import pathlib

import art
from datasets import Dataset, load_dataset, DatasetDict

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments.experiment_runner import ExperimentRunner
from experiments.big_code_bench.trainer import BigCodeBenchTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "big_code_bench_dac"

    def default_config_dir(self) -> str:
        return "experiments/big_code_bench/defaults"

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        data: Dataset = load_dataset(
            "bigcode/bigcodebench",
            split="v0.1.4",
        )  # type: ignore

        split_dict = data.train_test_split(test_size=0.2, seed=0)
        train_data: Dataset = split_dict["train"]
        test_data: Dataset = split_dict["test"]
        return train_data, test_data, test_data

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer:
        return BigCodeBenchTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

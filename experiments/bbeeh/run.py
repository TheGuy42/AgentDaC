import sys
import pathlib

import art
from datasets import Dataset

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.bbeeh.trainer import BbeehTrainer
from src.trainer import ArtTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "bbeeh_dac"

    def default_config_dir(self) -> str:
        return "experiments/bbeeh/defaults"

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        data = Dataset.load_from_disk("experiments/bbeeh/data", keep_in_memory=True)
        
        train_split = data.train_test_split(train_size=0.3, seed=0)
        train_data, other_data = train_split["train"], train_split["test"]

        other_split = other_data.train_test_split(train_size=0.25, seed=0)
        val_data, test_data = other_split["train"], other_split["test"]

        return train_data, val_data, test_data

    def create_trainer(self, model: art.Model, **kwargs) -> ArtTrainer:
        return BbeehTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

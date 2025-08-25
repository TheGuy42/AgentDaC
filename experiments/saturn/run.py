import sys
import pathlib

from art import TrainableModel
from datasets import Dataset

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.runner import ExperimentRunner
from experiments.saturn.trainer import SaturnTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "saturn_dac"

    def default_config_dir(self) -> str:
        return "experiments/saturn/defaults"

    def load_data(self) -> tuple[Dataset, Dataset]:
        data = Dataset.load_from_disk("experiments/saturn/dataset", keep_in_memory=True)
        split_dict = data.train_test_split(test_size=0.3, seed=0)
        train_data = split_dict["train"]
        test_data = split_dict["test"]
        return train_data, test_data

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer:
        return SaturnTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

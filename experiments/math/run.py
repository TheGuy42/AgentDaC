import sys
import pathlib

from art import TrainableModel
from datasets import Dataset, load_dataset, DatasetDict

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.runner import ExperimentRunner
from experiments.math.trainer import MathTrainer
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "math_dac"

    def default_config_dir(self) -> str:
        return "experiments/math/defaults"

    def load_data(self) -> tuple[Dataset, Dataset]:
        dataset_dict: DatasetDict = load_dataset(
            path="ankner/math-500",
            name="E2H-AMC",
            split=None,
        )  # type: ignore

        ds_train: Dataset = dataset_dict["train"]
        ds_val: Dataset = dataset_dict["test"]
        return ds_train, ds_val

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer: 
        return MathTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

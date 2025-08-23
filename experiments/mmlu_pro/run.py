import sys
import pathlib

from art import TrainableModel
from datasets import Dataset, load_dataset, DatasetDict

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.runner import ExperimentRunner
from experiments.mmlu_pro.trainer import MmluProTrainer
from src.vllm_client import VllmRouter
from src.trainer import Trainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "mmlu_pro_dac"

    def default_config_dir(self) -> str:
        return "experiments/mmlu_pro/defaults"

    def load_data(self) -> tuple[Dataset, Dataset]:
        data: Dataset = load_dataset(
            path="TIGER-Lab/MMLU-Pro",
            split="test",
        )  # type: ignore

        split_dict = data.train_test_split(test_size=0.3, seed=0)
        train_data = split_dict["train"]
        test_data = split_dict["test"]
        return train_data, test_data

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer:
        return MmluProTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

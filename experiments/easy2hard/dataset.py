from __future__ import annotations
from datasets import DatasetDict, load_dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class Easy2HardDataset(TaskDataset):
    """Easy2Hard-Bench (E2H-AMC), filtered by difficulty. Params from `self.params`."""

    def load(self):
        min_difficulty = self.params["min_difficulty"]
        max_difficulty = self.params["max_difficulty"]

        def in_range(sample) -> bool:
            return max_difficulty >= sample["item_difficulty"] >= min_difficulty

        dataset_dict: DatasetDict = load_dataset(
            path="furonghuang-lab/Easy2Hard-Bench", name="E2H-AMC", split=None
        )  # type: ignore

        # NOTE: the dataset's `eval` split is the larger one, so it is used for training
        # (matching the original `load_data`); the `train` split is used for validation.
        return {
            RolloutStage.TRAIN: dataset_dict["eval"].filter(in_range),
            RolloutStage.VAL: dataset_dict["train"].filter(in_range),
        }

from __future__ import annotations
from datasets import DatasetDict, load_dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class MathDataset(TaskDataset):
    """Hendrycks MATH benchmark, filtered by difficulty level."""

    def load(self):
        min_level = self.params["min_level"]
        max_level = self.params["max_level"]

        def in_range(sample) -> bool:
            return max_level >= sample["level"] >= min_level

        dataset_dict: DatasetDict = load_dataset(path="nlile/hendrycks-MATH-benchmark", split=None)  # type: ignore
        return {
            RolloutStage.TRAIN: dataset_dict["train"].filter(in_range),
            RolloutStage.VAL: dataset_dict["test"].filter(in_range),  # the HF "test" split is our val
        }

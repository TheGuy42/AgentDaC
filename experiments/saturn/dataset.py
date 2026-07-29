from __future__ import annotations

from datasets import Dataset

from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class SaturnDataset(TaskDataset):
    """Saturn (local on-disk dataset), split 70% train / 30% eval (seed 0)."""

    def load(self):
        data = Dataset.load_from_disk("experiments/saturn/data", keep_in_memory=True)
        split_dict = data.train_test_split(test_size=0.3, seed=0)
        return {RolloutStage.TRAIN: split_dict["train"], RolloutStage.VAL: split_dict["test"]}

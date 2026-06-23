from __future__ import annotations

from datasets import Dataset

from src.custom.dataset import DynamicDataset


class SaturnDataset(DynamicDataset):
    """Saturn (local on-disk dataset), split 70% train / 30% eval (seed 0)."""

    def load_split(self, split: str):
        data = Dataset.load_from_disk("experiments/saturn/data", keep_in_memory=True)
        split_dict = data.train_test_split(test_size=0.3, seed=0)
        return split_dict["train"] if split == "train" else split_dict["test"]

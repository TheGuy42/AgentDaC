from __future__ import annotations

from datasets import load_dataset, Dataset

from src.custom.dataset import DynamicDataset


class BigCodeBenchDataset(DynamicDataset):
    """BigCodeBench v0.1.4, split 80% train / 20% eval (seed 0)."""

    def load_split(self, split: str):
        data: Dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")  # type: ignore
        split_dict = data.train_test_split(test_size=0.2, seed=0)
        return split_dict["train"] if split == "train" else split_dict["test"]

from __future__ import annotations

from datasets import load_dataset, Dataset

from src.custom.dataset import DynamicDataset


class MmluProDataset(DynamicDataset):
    """MMLU-Pro (test split), re-split 70% train / 30% eval (seed 0)."""

    def load_split(self, split: str):
        data: Dataset = load_dataset(path="TIGER-Lab/MMLU-Pro", split="test")  # type: ignore
        split_dict = data.train_test_split(test_size=0.3, seed=0)
        return split_dict["train"] if split == "train" else split_dict["test"]

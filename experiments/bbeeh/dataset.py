from __future__ import annotations

from datasets import Dataset

from src.custom.dataset import DynamicDataset


class BbeehDataset(DynamicDataset):
    """BBEEH (local on-disk dataset), split 30% train / 17.5% val / 52.5% test (seed 0)."""

    def load_split(self, split: str):
        data = Dataset.load_from_disk("experiments/bbeeh/data", keep_in_memory=True)

        train_split = data.train_test_split(train_size=0.3, seed=0)
        train_data, other_data = train_split["train"], train_split["test"]

        other_split = other_data.train_test_split(train_size=0.25, seed=0)
        val_data, test_data = other_split["train"], other_split["test"]

        return {"train": train_data, "val": val_data, "test": test_data}[split]

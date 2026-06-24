from __future__ import annotations

from datasets import load_dataset, Dataset

from src.custom.dataset import DynamicDataset


class BbehDataset(DynamicDataset):
    """BBEH, filtered to the selected tasks. Params from `config.data.custom_dataset`."""

    def load_split(self, split: str):
        cfg = self.config.custom_dataset

        data: Dataset = load_dataset(path="BBEH/bbeh", split="train")  # type: ignore
        data = data.map(lambda sample: {"task": sample["task"].replace(" ", "_")})
        data = data.filter(lambda sample: sample["task"] in list(cfg.tasks))

        split_dict = data.train_test_split(test_size=0.25, seed=0)
        return split_dict["train"] if split == "train" else split_dict["test"]

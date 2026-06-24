from __future__ import annotations

from datasets import DatasetDict, load_dataset

from src.custom.dataset import DynamicDataset


class Easy2HardDataset(DynamicDataset):
    """Easy2Hard-Bench (E2H-AMC), filtered by difficulty. Params from `config.data.custom_dataset`.

    NOTE: the dataset's `eval` split is larger, so it is used for training (matching the
    original `load_data`); the `train` split is used for validation.
    """

    def load_split(self, split: str):
        cfg = self.config.custom_dataset

        dataset_dict: DatasetDict = load_dataset(
            path="furonghuang-lab/Easy2Hard-Bench", name="E2H-AMC", split=None
        )  # type: ignore
        ds = dataset_dict["eval"] if split == "train" else dataset_dict["train"]

        return ds.filter(lambda s: cfg.max_difficulty >= s["item_difficulty"] >= cfg.min_difficulty)

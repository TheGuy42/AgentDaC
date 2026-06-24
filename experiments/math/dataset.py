from __future__ import annotations

from datasets import DatasetDict, load_dataset

from src.custom.dataset import DynamicDataset


class MathDataset(DynamicDataset):
    """Hendrycks MATH benchmark, filtered by difficulty level.

    Load params are read from `config.data.custom_dataset` (embedded by
    `ExperimentRunner._build_verl_config` from the experiment's CLI args).
    """

    def load_split(self, split: str):
        cfg = self.config.custom_dataset

        dataset_dict: DatasetDict = load_dataset(path="nlile/hendrycks-MATH-benchmark", split=None)  # type: ignore
        ds = dataset_dict["train"] if split == "train" else dataset_dict["test"]  # "val" -> HF "test"

        return ds.filter(lambda sample: cfg.max_level >= sample["level"] >= cfg.min_level)

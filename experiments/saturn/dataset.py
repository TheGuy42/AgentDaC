from __future__ import annotations
from datasets import Dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class SaturnDataset(TaskDataset):
    """Saturn (local on-disk dataset), split 70% train / 30% eval"""

    def load(self):
        data = Dataset.load_from_disk("experiments/saturn/data", keep_in_memory=True)
        train_ds, val_ds = self._split_data(data, ratios=[0.7, 0.3], seed=self.params["seed"])
        return {RolloutStage.TRAIN: train_ds, RolloutStage.VAL: val_ds}

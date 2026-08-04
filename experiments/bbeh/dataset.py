from __future__ import annotations
from datasets import load_dataset, Dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class BbehDataset(TaskDataset):
    """BBEH, filtered to the selected tasks. Params from `self.params`."""

    def load(self):
        data: Dataset = load_dataset(path="BBEH/bbeh", split="train")  # type: ignore
        data = data.map(lambda sample: {"task": sample["task"].replace(" ", "_")})
        data = data.filter(lambda sample: sample["task"] in self.params["tasks"])
        train_ds, val_ds = self._split_data(data, ratios=[0.75, 0.25], seed=self.params["seed"])
        return {RolloutStage.TRAIN: train_ds, RolloutStage.VAL: val_ds}

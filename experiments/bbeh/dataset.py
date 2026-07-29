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

        split_dict = data.train_test_split(test_size=0.25, seed=0)
        return {RolloutStage.TRAIN: split_dict["train"], RolloutStage.VAL: split_dict["test"]}

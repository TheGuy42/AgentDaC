from __future__ import annotations
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage
from experiments.chess.data import load_dataset


class ChessPerstDataset(TaskDataset):
    """Lichess positions, pulled once and split into disjoint train/val sets."""

    def load(self):
        train_size = self.params["train_size"]
        val_size = self.params["val_size"]

        if train_size is None or val_size is None:
            raise ValueError(
                "ChessPerstDataset requires concrete train_size and val_size "
                "(set them in train_config.json); the chess loader needs a finite "
                "budget to bound the streaming pull."
            )

        train, val = load_dataset(
            list(self.params["datasets"]),
            num_train=train_size,
            num_val=val_size,
            seed=self.params["data_seed"],
            min_rating=self.params["min_rating"],
            max_rating=self.params["max_rating"],
        )
        return {RolloutStage.TRAIN: train, RolloutStage.VAL: val}

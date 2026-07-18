from __future__ import annotations
from src.custom.dataset import DynamicDataset
from experiments.chess.data import load_dataset


class ChessPerstDataset(DynamicDataset):
    def load_split(self, split: str):
        cfg = self.config.custom_dataset
        if cfg.train_size is None or cfg.val_size is None:
            raise ValueError(
                "ChessPerstDataset requires concrete train_size and val_size "
                "(set them in train_config.json); the chess loader needs a finite "
                "budget to bound the streaming pull."
            )
        train, val = load_dataset(
            list(cfg.datasets),
            num_train=cfg.train_size,
            num_val=cfg.val_size,
            seed=cfg.data_seed,
            min_rating=cfg.min_rating,
            max_rating=cfg.max_rating,
        )
        return train if split == "train" else val

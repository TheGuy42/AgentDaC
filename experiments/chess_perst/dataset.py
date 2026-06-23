from __future__ import annotations

from src.custom.dataset import DynamicDataset

from experiments.chess_perst.data import load_dataset


class ChessPerstDataset(DynamicDataset):
    """Chess positions (FEN + ply) from one or more sources, pooled into train/val splits.

    The chess loader produces disjoint, pre-sized splits (``num_train``/``num_val``); the
    base ``DynamicDataset`` then applies the usual ``train_size``/``val_size`` slice. Params
    come from ``config.data.custom_dataset``.
    """

    def load_split(self, split: str):
        cfg = self.config.custom_dataset
        train, val = load_dataset(
            list(cfg.datasets),
            num_train=cfg.num_train,
            num_val=cfg.num_val,
            seed=cfg.data_seed,
            min_rating=cfg.min_rating,
            max_rating=cfg.max_rating,
        )
        return train if split == "train" else val

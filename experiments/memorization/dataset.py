from __future__ import annotations

import random

from datasets import Dataset

from src.custom.dataset import DynamicDataset

LABEL_KINDS = ["random", "const"]


class MemorizationDataset(DynamicDataset):
    """Synthetic memorization data: ``num_samples`` rows with fixed labels.

    All splits return the same deterministic set (the task is to memorize), seeded by
    ``config.data.custom_dataset.seed`` so train/val are identical across instantiations.
    """

    def load_split(self, split: str):
        cfg = self.config.custom_dataset
        labels: list[str] = list(cfg.labels)
        num_samples = int(cfg.num_samples)
        kind = cfg.label_kind
        rng = random.Random(cfg.seed)

        if kind == "random":
            sample_labels = (labels * (num_samples // len(labels) + 1))[:num_samples]
            rng.shuffle(sample_labels)
        elif kind == "const":
            sample_labels = [rng.choice(labels)] * num_samples
        else:
            raise ValueError(f"Invalid label_kind: {kind}. Must be one of {LABEL_KINDS}.")

        samples = [{"id": i, "answer": sample_labels[i], "labels": labels} for i in range(num_samples)]
        return Dataset.from_list(samples)

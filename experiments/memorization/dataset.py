from __future__ import annotations
import random
from datasets import Dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


LABEL_KINDS = ["random", "const"]


class MemorizationDataset(TaskDataset):
    """Synthetic memorization data: `num_samples` rows with fixed labels.

    Seeded by `self.params["seed"]`, so the rows are identical across instantiations.
    """

    def load(self):
        labels: list[str] = list(self.params["labels"])
        num_samples = int(self.params["num_samples"])
        kind = self.params["label_kind"]
        rng = random.Random(self.params["seed"])

        if kind not in LABEL_KINDS:
            raise ValueError(f"Invalid label_kind: {kind}. Must be one of {LABEL_KINDS}.")

        if kind == "random":
            sample_labels = (labels * (num_samples // len(labels) + 1))[:num_samples]
            rng.shuffle(sample_labels)
        elif kind == "const":
            sample_labels = [rng.choice(labels)] * num_samples
        else:
            raise ValueError(f"Invalid label_kind: {kind}. Must be one of {LABEL_KINDS}.")

        samples = [{"id": i, "answer": sample_labels[i], "labels": labels} for i in range(num_samples)]
        data = Dataset.from_list(samples)

        # The task is to memorize, so every stage is deliberately the same set.
        return {stage: data for stage in RolloutStage}

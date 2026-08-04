from __future__ import annotations
from datasets import Dataset, load_dataset
from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class DapoDataset(TaskDataset):
    """
    - Train: `DAPO-Math-17k` dataset
    - Val: Mix of `AIME-2025`, `Beyond-AIME`, `AIME-2024`, `HMMT-2025`
    """

    def load(self):
        train_ds: Dataset = load_dataset(path="BytedTsinghua-SIA/DAPO-Math-17k", split="train")  # type: ignore

        # Source: https://github.com/MasterVito/DAC-RL/blob/main/data/dac-rl-benchmarks.jsonl
        val_ds = Dataset.load_from_disk("experiments/dapo_math/data", keep_in_memory=True)

        val_ds = val_ds.map(
            lambda sample: {
                "prompt": [{"role": "user", "content": sample["question"]}],
                "reward_model": {"ground_truth": sample["ref_answer"]},
            },
            keep_in_memory=True,
        )

        return {RolloutStage.TRAIN: train_ds, RolloutStage.VAL: val_ds}

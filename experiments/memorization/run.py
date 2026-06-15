from __future__ import annotations
import argparse
import pathlib
import sys
import random
import copy

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.trainer import AglTrainer
from experiments.experiment_runner import Dataset, ExperimentRunner
from experiments.memorization.trainer import MemorizationTrainer


LABEL_KINDS = ["random", "uniform"]


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "memorization"

    def default_config_dir(self) -> str:
        return "experiments/memorization/defaults"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num_samples", type=int, default=20)
        parser.add_argument("--labels", type=str, nargs="+", default=["A", "B"])
        parser.add_argument("--label_kind", type=str, default="random", choices=LABEL_KINDS)

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        num_samples = self.args().num_samples
        labels: list[str] = self.args().labels

        kind = self.args().label_kind
        if kind not in LABEL_KINDS:
            raise ValueError(f"Invalid label_kind: {kind}. Must be one of {LABEL_KINDS}.")

        if kind == "random":
            random_labels = (labels * (num_samples // len(labels) + 1))[:num_samples]
            random.shuffle(random_labels)
        elif kind == "uniform":
            lbl = random.choice(labels)
            random_labels = [lbl] * num_samples
        else:
            raise ValueError(f"Invalid label_kind: {kind}")

        samples = [{"id": i, "answer": random_labels[i], "labels": labels} for i in range(num_samples)]
        ds_train = Dataset.from_list(samples)
        return ds_train, ds_train, ds_train

    def create_trainer(self, **kwargs) -> AglTrainer:
        return MemorizationTrainer(**kwargs)


if __name__ == "__main__":
    Runner().run()

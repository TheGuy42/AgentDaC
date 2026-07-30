from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.agents.registry import AgentKey
from experiments._framework import ExperimentRunner
from experiments.memorization.dataset import LABEL_KINDS, MemorizationDataset
from experiments.memorization.task import MemorizationTask


class Runner(ExperimentRunner):
    def task_name(self) -> str:
        return "memorization"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER, AgentKey.DUMMY, AgentKey.PERST, AgentKey.TOOL_SUBMIT]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--num_samples", type=int, default=10)
        parser.add_argument("--labels", type=str, nargs="+", default=["A", "B"])
        parser.add_argument("--label_kind", type=str, default="random", choices=LABEL_KINDS)

    def dataset_class(self) -> type:
        return MemorizationDataset

    def task_class(self) -> type:
        return MemorizationTask

    def dataset_args(self) -> dict[str, Any]:
        return {
            "num_samples": self.args().num_samples,
            "labels": self.args().labels,
            "label_kind": self.args().label_kind,
        }


if __name__ == "__main__":
    Runner().run()

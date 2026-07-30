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
from experiments.bbeh.dataset import BbehDataset
from experiments.bbeh.task import BbehTask
from experiments.bbeh.tasks import SupportedTasks


class Runner(ExperimentRunner):
    def task_name(self) -> str:
        return "bbeh"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--tasks",
            type=str,
            nargs="+",
            default=None,
            required=True,
            help=f"Which tasks of the BBEH dataset to use. Available tasks: {SupportedTasks.list_values()}",
        )

    def dataset_class(self) -> type:
        return BbehDataset

    def task_class(self) -> type:
        return BbehTask

    def dataset_args(self) -> dict[str, Any]:
        return {"tasks": self.args().tasks}


if __name__ == "__main__":
    Runner().run()

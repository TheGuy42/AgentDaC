from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.agents.registry import AgentKey
from experiments._framework import VerlRunner
from experiments.saturn.dataset import SaturnDataset
from experiments.saturn.task import SaturnTask


class Runner(VerlRunner):
    def task_name(self) -> str:
        return "saturn"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER]

    def dataset_class(self) -> type:
        return SaturnDataset

    def task_class(self) -> type:
        return SaturnTask


if __name__ == "__main__":
    Runner().run()

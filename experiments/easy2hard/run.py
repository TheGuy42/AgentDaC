from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.agents.registry import AgentKey
from experiments._framework import VerlRunner
from experiments.easy2hard.dataset import Easy2HardDataset
from experiments.easy2hard.task import Easy2HardTask


class Runner(VerlRunner):
    def task_name(self) -> str:
        return "easy2hard"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER, AgentKey.REGEX]

    def dataset_class(self) -> type:
        return Easy2HardDataset

    def task_class(self) -> type:
        return Easy2HardTask


if __name__ == "__main__":
    Runner().run()

from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.agents.registry import AgentKey
from experiments._framework import VerlRunner
from experiments.math.dataset import MathDataset
from experiments.math.task import MathTask


class Runner(VerlRunner):
    def task_name(self) -> str:
        return "math"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER, AgentKey.DUMMY, AgentKey.JSON, AgentKey.REGEX, AgentKey.PERST]

    def dataset_class(self) -> type:
        return MathDataset

    def task_class(self) -> type:
        return MathTask


if __name__ == "__main__":
    Runner().run()

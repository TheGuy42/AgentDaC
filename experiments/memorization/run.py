from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.agents.registry import AgentKey
from experiments._framework import VerlRunner
from experiments.memorization.dataset import MemorizationDataset
from experiments.memorization.task import MemorizationTask


class Runner(VerlRunner):
    def task_name(self) -> str:
        return "memorization"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER, AgentKey.DUMMY, AgentKey.PERST, AgentKey.TOOL_SUBMIT]

    def dataset_class(self) -> type:
        return MemorizationDataset

    def task_class(self) -> type:
        return MemorizationTask


if __name__ == "__main__":
    Runner().run()

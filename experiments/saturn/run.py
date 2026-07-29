from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.agents.registry import AgentKey
from experiments._framework import ExperimentRunner
from experiments.saturn.dataset import SaturnDataset
from experiments.saturn.trainer import SaturnTrainer


class Runner(ExperimentRunner):
    def task_name(self) -> str:
        return "saturn"

    def supported_agents(self) -> list[str]:
        return [AgentKey.MARKER]

    def dataset_class(self) -> type:
        return SaturnDataset

    def trainer_class(self) -> type:
        return SaturnTrainer


if __name__ == "__main__":
    Runner().run()

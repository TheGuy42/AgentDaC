from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.agents.registry import AgentKey
from experiments._framework.experiment import Experiment
from experiments.saturn.dataset import SaturnDataset
from experiments.saturn.task import SaturnTask


if __name__ == "__main__":
    Experiment(
        task_name="saturn",
        supported_agents=[AgentKey.MARKER],
        task_cls=SaturnTask,
        dataset_cls=SaturnDataset,
    ).run()

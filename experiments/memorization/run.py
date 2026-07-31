from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.agents.registry import AgentKey
from experiments._framework.experiment import Experiment
from experiments.memorization.dataset import MemorizationDataset
from experiments.memorization.task import MemorizationTask


if __name__ == "__main__":
    Experiment(
        task_name="memorization",
        supported_agents=[
            AgentKey.MARKER,
            AgentKey.DUMMY,
            AgentKey.PERST,
            AgentKey.TOOL_SUBMIT,
        ],
        task_cls=MemorizationTask,
        dataset_cls=MemorizationDataset,
    ).run()

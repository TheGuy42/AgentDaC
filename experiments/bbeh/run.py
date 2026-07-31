from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.agents.registry import AgentKey
from experiments._framework.experiment import Experiment
from experiments.bbeh.dataset import BbehDataset
from experiments.bbeh.task import BbehTask


if __name__ == "__main__":
    Experiment(
        task_name="bbeh",
        supported_agents=[AgentKey.MARKER],
        task_cls=BbehTask,
        dataset_cls=BbehDataset,
    ).run()

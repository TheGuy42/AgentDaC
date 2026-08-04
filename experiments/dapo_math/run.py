from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments._framework.experiment import Experiment
from experiments.dapo_math.dataset import DapoDataset
from experiments.dapo_math.task import DapoTask


if __name__ == "__main__":
    Experiment(
        task_name="dapo_math",
        task_cls=DapoTask,
        dataset_cls=DapoDataset,
    ).run()

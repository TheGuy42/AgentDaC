from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments._framework.experiment import Experiment
from experiments.easy2hard.dataset import Easy2HardDataset
from experiments.easy2hard.task import Easy2HardTask


if __name__ == "__main__":
    Experiment(
        task_name="easy2hard",
        task_cls=Easy2HardTask,
        dataset_cls=Easy2HardDataset,
    ).run()

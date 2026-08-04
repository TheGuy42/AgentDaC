from __future__ import annotations

import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments._framework.experiment import Experiment
from experiments.math.dataset import MathDataset
from experiments.math.task import MathTask


if __name__ == "__main__":
    Experiment(
        task_name="math",
        task_cls=MathTask,
        dataset_cls=MathDataset,
    ).run()

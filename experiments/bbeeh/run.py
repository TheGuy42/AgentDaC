import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.bbeeh.dataset import BbeehDataset
from experiments.bbeeh.trainer import BbeehTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "bbeeh_dac"

    def default_config_dir(self) -> str:
        return "experiments/bbeeh/defaults"

    def dataset_class(self) -> type:
        return BbeehDataset

    def trainer_class(self) -> type:
        return BbeehTrainer


if __name__ == "__main__":
    Runner().run()

import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.saturn.dataset import SaturnDataset
from experiments.saturn.trainer import SaturnTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "saturn_dac"

    def default_config_dir(self) -> str:
        return "experiments/saturn/defaults"

    def dataset_class(self) -> type:
        return SaturnDataset

    def trainer_class(self) -> type:
        return SaturnTrainer


if __name__ == "__main__":
    Runner().run()

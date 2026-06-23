import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments.experiment_runner import ExperimentRunner
from experiments.big_code_bench.dataset import BigCodeBenchDataset
from experiments.big_code_bench.trainer import BigCodeBenchTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "big_code_bench_dac"

    def default_config_dir(self) -> str:
        return "experiments/big_code_bench/defaults"

    def dataset_class(self) -> type:
        return BigCodeBenchDataset

    def trainer_class(self) -> type:
        return BigCodeBenchTrainer


if __name__ == "__main__":
    Runner().run()

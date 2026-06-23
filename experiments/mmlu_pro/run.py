import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.experiment_runner import ExperimentRunner
from experiments.mmlu_pro.dataset import MmluProDataset
from experiments.mmlu_pro.trainer import MmluProTrainer


class Runner(ExperimentRunner):
    def default_project_name(self) -> str:
        return "mmlu_pro_dac"

    def default_config_dir(self) -> str:
        return "experiments/mmlu_pro/defaults"

    def dataset_class(self) -> type:
        return MmluProDataset

    def trainer_class(self) -> type:
        return MmluProTrainer


if __name__ == "__main__":
    Runner().run()

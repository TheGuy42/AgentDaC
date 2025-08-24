import sys
import pathlib
from art import TrainableModel

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.easy2hard.run import Runner as RunnerEasy2Hard
from experiments.easy2hard_tools.trainer import Easy2HardToolTrainer
from src.trainer import Trainer


class Runner(RunnerEasy2Hard):
    def default_project_name(self) -> str:
        return "easy2hard_dac_tools"

    def default_config_dir(self) -> str:
        return "experiments/easy2hard_tools/defaults"

    def create_trainer(self, model: TrainableModel, **kwargs) -> Trainer:
        return Easy2HardToolTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

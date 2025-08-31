import sys
import pathlib
import art


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.easy2hard.run import Runner as RunnerEasy2hard
from experiments.easy2hard_ruler.trainer import Easy2HardRulerTrainer
from src.trainer import Trainer


class Runner(RunnerEasy2hard):
    def default_project_name(self) -> str:
        return "easy2hard_dac_ruler"

    def default_config_dir(self) -> str:
        return "experiments/easy2hard_ruler/defaults"

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer:
        return Easy2HardRulerTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

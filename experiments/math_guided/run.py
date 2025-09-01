import sys
import pathlib

import art

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.math_guided.trainer import MathGuidedTrainer
from experiments.math.run import Runner as MathRunner
from src.trainer import Trainer


class Runner(MathRunner):
    def default_project_name(self) -> str:
        return "math_guided_dac"

    def default_config_dir(self) -> str:
        return "experiments/math_guided/defaults"

    def create_trainer(self, model: art.Model, **kwargs) -> Trainer: 
        return MathGuidedTrainer(model=model, **kwargs)


if __name__ == "__main__":
    Runner().run()

import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.math.run import Runner as MathRunner
from experiments.math_perst.trainer import MathPerstTrainer


class Runner(MathRunner):
    def default_project_name(self) -> str:
        return "math_perst_dac"

    def default_config_dir(self) -> str:
        return "experiments/math_perst/defaults"

    def trainer_class(self) -> type:
        return MathPerstTrainer



if __name__ == "__main__":
    Runner().run()

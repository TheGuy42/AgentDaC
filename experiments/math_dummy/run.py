import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.math_dummy.trainer import MathDummyTrainer
from experiments.math.run import Runner as MathRunner


class Runner(MathRunner):
    def default_project_name(self) -> str:
        return "math_dummy"

    def default_config_dir(self) -> str:
        return "experiments/math_dummy/defaults"

    def create_trainer(self, **kwargs) -> MathDummyTrainer:
        return MathDummyTrainer(**kwargs)


if __name__ == "__main__":
    Runner().run()

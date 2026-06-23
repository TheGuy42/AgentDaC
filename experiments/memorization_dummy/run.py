import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from experiments.memorization.run import Runner as MemorizationRunner
from experiments.memorization_dummy.trainer import MemorizationDummyTrainer


class Runner(MemorizationRunner):
    def default_project_name(self) -> str:
        return "memorization_dummy"

    def default_config_dir(self) -> str:
        return "experiments/memorization_dummy/defaults"

    def trainer_class(self) -> type:
        return MemorizationDummyTrainer



if __name__ == "__main__":
    Runner().run()

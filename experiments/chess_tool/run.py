from __future__ import annotations
import pathlib
import sys

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils.logging import create_logger
from experiments.chess_perst.run import Runner as ChessRunner
from experiments.chess_tool.trainer import ChessToolTrainer


logger = create_logger(__name__)


class Runner(ChessRunner):
    def default_project_name(self) -> str:
        return "chess_tool_dac"

    def default_config_dir(self) -> str:
        return "experiments/chess_tool/defaults"

    def trainer_class(self) -> type:
        return ChessToolTrainer


if __name__ == "__main__":
    Runner().run()

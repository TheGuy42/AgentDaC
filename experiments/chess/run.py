from __future__ import annotations

import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from experiments._framework.experiment import Experiment
from experiments.chess.dataset import ChessPerstDataset
from experiments.chess.task import ChessTask
from experiments.chess.chess_engine import ChessConfig, EngineConfig


class ChessExperiment(Experiment):
    def task_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        """The chess-specific engine configs."""
        return {
            "engine_config": EngineConfig.load_from_path(config_dir / "engine_config.yaml", do_raise=True),
            "chess_config": ChessConfig.load_from_path(config_dir / "chess_config.yaml", do_raise=True),
        }


if __name__ == "__main__":
    ChessExperiment(
        task_name="chess",
        task_cls=ChessTask,
        dataset_cls=ChessPerstDataset,
    ).run()

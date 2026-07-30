from __future__ import annotations

import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils.logging import create_logger
from src.agents.registry import AgentKey
from experiments._framework import VerlRunner
from experiments.chess.dataset import ChessPerstDataset
from experiments.chess.task import ChessTask
from experiments.chess.chess_engine import ChessConfig, EngineConfig


logger = create_logger(__name__)


class Runner(VerlRunner):
    def task_name(self) -> str:
        return "chess"

    def supported_agents(self) -> list[str]:
        return [
            AgentKey.PERST,
            AgentKey.NATIVE_PERST,
            AgentKey.TOOL_STATELESS,
            AgentKey.TOOL_PERSISTENT,
            AgentKey.TOOL_SUBMIT,
        ]

    def _load_configs(self, dir: str | pathlib.Path) -> dict[str, Any]:
        """Load the shared configs plus the chess-specific engine config."""
        configs = super()._load_configs(dir)
        configs["engine_config"] = EngineConfig.load_from_path(pathlib.Path(dir) / "engine_config.json", do_raise=True)
        configs["chess_config"] = ChessConfig.load_from_path(pathlib.Path(dir) / "chess_config.json", do_raise=True)
        return configs

    def dataset_class(self) -> type:
        return ChessPerstDataset

    def task_class(self) -> type:
        return ChessTask


if __name__ == "__main__":
    Runner().run()

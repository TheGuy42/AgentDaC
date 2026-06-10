from __future__ import annotations
from typing import Literal
from src.configs.base_config import BaseConfig


class EngineConfig(BaseConfig):
    """Settings for the local UCI engine and the move reward it produces."""

    engine_path: str = "stockfish"
    depth: int = 12
    threads: int = 1
    hash_mb: int = 64
    reward: Literal["win_prob", "centipawns"] = "win_prob"
    illegal_score: float = 0.0

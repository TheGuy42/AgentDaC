from __future__ import annotations
from typing import Literal
from chess.engine import Limit
from src.configs.base_config import BaseConfig


class EngineConfig(BaseConfig):
    """Settings for the local UCI engine and the move reward it produces."""

    limit: Limit
    engine_path: str = "stockfish"
    threads: int = 1
    hash_mb: int = 64
    reward: Literal["win_prob", "centipawns"] = "win_prob"
    illegal_score: float = 0.0
    mate_score: int = 10_000
    """Centipawn magnitude a forced mate maps to (scaled down by distance to mate)."""

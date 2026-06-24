from __future__ import annotations
from typing import Literal
from chess.engine import Limit
from src.configs.base_config import BaseConfig


class EngineConfig(BaseConfig):
    """Settings for the local UCI engine and the move reward it produces."""

    limit: Limit
    """Search budget per analysis, e.g. `{"nodes": 1000000}` or `{"time": 0.5}`. A nodes (or
    depth) limit is reproducible; a time limit varies with machine load and concurrency."""

    engine_path: str = "stockfish"
    """Path to the UCI engine binary — a name on PATH or an absolute path. Any UCI engine works."""

    threads: int = 2
    """UCI search threads per engine process. Threads=1 is deterministic; multi-threaded Lazy-SMP
    is not. One engine process runs per rollout worker, so keep (workers * threads) within the
    core count."""

    hash_mb: int = 64
    """Engine transposition-table size in MB (the UCI `Hash` option), per engine process."""

    eval_mode: Literal["child", "root_multipv"] = "child"
    """How a move is evaluated: 'child' = one search per move;
    'root_multipv'= one MultiPV search of the root, cached and shared across the GRPO group."""

    reward: Literal["win_prob", "centipawns"] = "win_prob"
    """The type of reward to use for evaluating moves."""

    mate_score: int = 10_000
    """Centipawn magnitude a forced mate maps to (scaled down by distance to mate)."""

    win_scale: float = 600.0
    """Logistic scale for the win_prob reward; larger = less saturation (400 ~ Elo expectancy)."""

    mate_margin: float = 0.05
    """Width of the reserved end-bands that keep mate rewards distinct from any centipawn eval."""

    mate_decay: float = 0.005
    """Per-ply reward step within a mate band, so a faster mate scores higher."""

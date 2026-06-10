from __future__ import annotations

import asyncio

import chess
import chess.engine

from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine.config import EngineConfig


logger = create_logger(__name__)


# Magnitude assigned to a forced mate (which has no centipawn value); 
# python-chess scales it down by distance to mate, so quicker mates score higher.
MATE_CP = 10_000


class LocalEngine:
    """A single persistent local UCI engine process (e.g. Stockfish).

    Wraps python-chess's synchronous ``SimpleEngine``: the blocking search runs in a
    worker thread (``asyncio.to_thread``) and an ``asyncio.Lock`` serializes access, so one
    engine process can be shared safely across concurrent rollouts within a runner.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(config.engine_path)
        self._engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        self._limit = chess.engine.Limit(depth=config.depth)
        self._lock = asyncio.Lock()

    async def evaluate(self, board: chess.Board) -> int:
        """Return the engine's score for ``board`` in centipawns, from White's POV."""
        async with self._lock:
            info = await asyncio.to_thread(self._engine.analyse, board, self._limit)
        return info["score"].white().score(mate_score=MATE_CP)

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception as exc:  # noqa: BLE001 - best-effort shutdown
            logger.warning(f"Failed to quit engine cleanly: {exc}")

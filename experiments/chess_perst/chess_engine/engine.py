from __future__ import annotations
import asyncio
import chess

from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine.config import EngineConfig
from chess.engine import InfoDict, SimpleEngine

logger = create_logger(__name__)


class LocalEngine:
    """A single persistent local UCI engine process (e.g. Stockfish).

    A thin wrapper over python-chess's synchronous ``SimpleEngine`` that exposes one async
    method, :meth:`evaluate`, returning the raw analysis. All scoring/reward logic lives in
    :class:`MoveEvaluator`. The engine can be shared across concurrent rollouts within a
    runner: access is serialized and the blocking search runs off the event loop.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.engine: SimpleEngine = SimpleEngine.popen_uci(config.engine_path)
        self.engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        self._lock = asyncio.Lock()

    async def evaluate(self, board: chess.Board) -> InfoDict:
        """Return the engine's raw analysis of ``board`` (a single, serialized search)."""

        # Lock: a UCI engine is one serial pipe — a second analyse() would cancel the
        # running search, so only one is ever in flight.
        # to_thread: analyse() blocks until the search finishes; run it off-loop so other
        # rollouts keep progressing. (SimpleEngine is affinity-free, so any pool thread is fine.)
        async with self._lock:
            return await asyncio.to_thread(self.engine.analyse, board, limit=self.config.limit)

    def close(self) -> None:
        if self.engine is None:
            return
        try:
            self.engine.quit()
            self.engine = None  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"Failed to quit engine cleanly: {e}")

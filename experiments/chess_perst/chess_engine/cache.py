from __future__ import annotations
import asyncio
from collections import OrderedDict

import chess
from chess.engine import PovScore
from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine.engine import LocalEngine


logger = create_logger(__name__)


class RootCache:
    """Stampede-safe per-FEN cache of ``{move_uci -> PovScore}`` from one MultiPV search.

    The first rollout to ask for a FEN launches a single MultiPV search of the root (scoring every
    legal move on one footing); concurrent rollouts of that FEN ``await`` the same task instead of
    issuing their own search. The cache `_lock` guards the task map; the engine's own lock
    serializes the UCI pipe. Bounded by an LRU so memory stays flat over a run.
    """

    def __init__(self, engine: LocalEngine, cache_size: int) -> None:
        self.engine = engine
        self.cache_size = cache_size
        self._tasks: OrderedDict[str, asyncio.Task[dict[str, PovScore]]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def build(self, fen: str) -> dict[str, PovScore]:
        """
        Build a cache of analysis for the given FEN.
        """
        async with self._lock:
            task = self._tasks.get(fen)

            if task is None:
                # first rollout to ask for this FEN
                task = asyncio.create_task(self._search(fen))
                self._tasks[fen] = task
            else:
                # update LRU order
                self._tasks.move_to_end(fen)

            while len(self._tasks) > self.cache_size:
                self._tasks.popitem(last=False)

        try:
            return await task  # awaited outside the lock; a finished task returns instantly
        except Exception:
            # Don't let a failed search poison the FEN — drop it so the next call retries.
            async with self._lock:
                if self._tasks.get(fen) is task:
                    del self._tasks[fen]
            raise

    async def _search(self, fen: str) -> dict[str, PovScore]:
        board = chess.Board(fen)
        infos = await self.engine.analyse(board, multipv=board.legal_moves.count())
        return {info["pv"][0].uci(): info["score"] for info in infos}  # type: ignore[typeddict-item]

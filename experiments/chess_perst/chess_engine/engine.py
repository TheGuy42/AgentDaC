from __future__ import annotations
import asyncio
from typing import Iterable, overload

import chess
from chess.engine import InfoDict, SimpleEngine
from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine.config import EngineConfig, ChessConfig


logger = create_logger(__name__)


class LocalEngine:
    """A single persistent local UCI engine process (e.g. Stockfish).

    A thin wrapper over python-chess's synchronous `SimpleEngine` that exposes one async
    method, :meth:`analyse`, returning the raw analysis. All scoring/reward logic lives in
    :class:`MoveEvaluator`. The engine can be shared across concurrent rollouts within a
    runner: access is serialized and the blocking search runs off the event loop.
    """

    def __init__(self, engine_config: EngineConfig) -> None:
        self.engine_config = engine_config
        self.engine: SimpleEngine = SimpleEngine.popen_uci(engine_config.engine_path)
        self.engine.configure({"Threads": engine_config.threads, "Hash": engine_config.hash_mb})
        self._lock = asyncio.Lock()

    @overload
    async def analyse(
        self,
        board: chess.Board,
        config: ChessConfig,
        *,
        multipv: int,
        root_moves: Iterable[chess.Move] | None = None,
    ) -> list[InfoDict]: ...

    @overload
    async def analyse(
        self,
        board: chess.Board,
        config: ChessConfig,
        *,
        multipv: None = None,
        root_moves: Iterable[chess.Move] | None = None,
    ) -> InfoDict: ...

    @overload
    async def analyse(
        self,
        board: chess.Board,
        config: ChessConfig,
        *,
        multipv: int | None,
        root_moves: Iterable[chess.Move] | None = None,
    ) -> InfoDict | list[InfoDict]: ...

    async def analyse(
        self,
        board: chess.Board,
        config: ChessConfig,
        *,
        multipv: int | None = None,
        root_moves: Iterable[chess.Move] | None = None,
    ) -> InfoDict | list[InfoDict]:
        """Return the engine's raw analysis of `board` (a single, serialized search).

        Returns one :class:`InfoDict`, or a ``list[InfoDict]`` (one per line) when `multipv`
        is set. `root_moves` restricts the candidate *first* moves at the root (UCI
        `searchmoves`); e.g. `root_moves=[m]` evaluates only move ``m`` without pushing it.
        """

        # Lock: a UCI engine is one serial pipe — a second analyse() would cancel the
        # running search, so only one is ever in flight.
        # to_thread: analyse() blocks until the search finishes; run it off-loop so other
        # rollouts keep progressing. (SimpleEngine is affinity-free, so any pool thread is fine.)
        async with self._lock:
            kwargs = config.kwargs.copy()
            if config.deterministic:
                kwargs["game"] = object()

            return await asyncio.to_thread(
                self.engine.analyse,
                board,
                limit=config.limit,
                root_moves=root_moves,
                multipv=multipv,
                **kwargs,
            )

    def close(self) -> None:
        if self.engine is None:
            return
        try:
            self.engine.quit()
            self.engine = None  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"Failed to quit engine cleanly: {e}")

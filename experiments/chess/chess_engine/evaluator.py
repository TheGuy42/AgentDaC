from __future__ import annotations
import atexit
from dataclasses import dataclass
from typing import ClassVar

import chess
from chess.engine import Score, PovScore
from src.utils.logging import create_logger
from experiments.chess.chess_engine.config import EngineConfig, ChessConfig
from experiments.chess.chess_engine.engine import LocalEngine
from experiments.chess.chess_engine.cache import RootCache


logger = create_logger(__name__)


@dataclass(frozen=True)
class MoveResult:
    fen: str
    """Starting position in FEN notation."""

    parse_success: bool
    """Whether the move was legal and parseable."""

    agent_move: str | None = None
    """The move in UCI notation, or None if illegal/unparseable."""

    score: Score | None = None
    """The engine evaluation of the resulting position, from the mover's POV (higher = better). None if illegal."""

    best_score: Score | None = None
    """The score of the best move in the position, from the mover's POV (higher = better). None if illegal."""

    cp: int | None = None
    """The centipawn score of the resulting position, from the mover's POV."""


class MoveEvaluator:
    """Scores a proposed move from a FEN using a persistent local engine.

    Owns a :class:`LocalEngine` and the reward policy from :class:`EngineConfig`. One
    evaluator — hence one engine process — is shared per runner subprocess via
    :meth:`for_process`.
    """

    _process_instance: ClassVar[MoveEvaluator | None] = None
    _process_config: ClassVar[EngineConfig | None] = None

    def __init__(self, engine_config: EngineConfig, engine: LocalEngine | None = None) -> None:
        self.engine: LocalEngine = engine if engine is not None else LocalEngine(engine_config)
        self.root_cache = RootCache(self.engine, cache_size=4096)

    @property
    def config(self) -> EngineConfig:
        return self.engine.engine_config

    @classmethod
    def for_process(cls, engine_config: EngineConfig) -> MoveEvaluator:
        """
        Return the per-process shared evaluator, starting its engine on first use.
        The engine is started lazily and closed at interpreter exit. The first call's
        `config` wins for the lifetime of the process.
        """
        if cls._process_instance is None:
            logger.debug("Starting process-local MoveEvaluator engine...")
            cls._process_instance = cls(engine_config)
            cls._process_config = engine_config
            atexit.register(cls._process_instance.close)
        elif engine_config != cls._process_config:
            raise RuntimeError(
                "MoveEvaluator.for_process() was called with a different EngineConfig after the process-local evaluator was already initialized."
            )

        return cls._process_instance

    @classmethod
    def close_process(cls) -> None:
        """Close the process-local evaluator and its engine."""
        if cls._process_instance is not None:
            cls._process_instance.close()
            cls._process_instance = None
            cls._process_config = None

    async def score(self, board: chess.Board, move: chess.Move | None, config: ChessConfig) -> MoveResult:
        """
        Score a proposed move from a FEN using the persistent engine.

        Args:
            board (chess.Board): The current board position.
            move (chess.Move | None): The proposed move to evaluate.

        Returns:
            MoveResult: the outcome of the evaluation.
            The returned `score` is at opponents turn but from our POV (higher = better).
            `score` and `cp` are None if the move was illegal or unparseable.
        """
        fen = board.fen()

        # First, find the best move in the position, if requested.

        pov_best: PovScore | None = None
        if config.relative:
            score_cache = await self.root_cache.build(fen, config)
            pov_best = max(score_cache.values(), key=lambda ps: ps.pov(board.turn))

        if move is None:
            return MoveResult(
                fen=fen,
                parse_success=False,
                best_score=pov_best.pov(board.turn) if pov_best is not None else None,
            )

        # If the move is legal, get its score from the engine.
        # First check the root multipv cache, then fall back to a single-move search if not found.

        pov_move: PovScore | None = None
        if config.mode == "root_multipv":
            score_cache = await self.root_cache.build(fen, config)
            pov_move = score_cache.get(move.uci())

        if pov_move is None:
            info = await self.engine.analyse(board, config, root_moves=[move])
            pov_move = info.get("score")

        if pov_move is None:
            raise RuntimeError("Engine analysis did not return a score for the move.")

        score_move = pov_move.pov(board.turn)
        score_best = pov_best.pov(board.turn) if pov_best is not None else None

        return MoveResult(
            fen=fen,
            parse_success=True,
            agent_move=move.uci(),
            score=score_move,
            best_score=score_best,
            cp=score_move.score(mate_score=config.mate_score),
        )

    def close(self) -> None:
        if self.engine is None:
            return

        self.engine.close()
        self.engine = None  # type: ignore[assignment]

from __future__ import annotations
import atexit
import re
from dataclasses import dataclass
from typing import ClassVar

import chess
from chess.engine import Score
from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine.config import EngineConfig
from experiments.chess_perst.chess_engine.engine import LocalEngine


logger = create_logger(__name__)


@dataclass(frozen=True)
class MoveResult:
    """Outcome of evaluating one proposed move from a FEN.
    ``fen`` is the starting position, ``parse_success`` indicates whether the move was legal and parseable,
    ``score`` is the engine eval of the resulting position from the mover's POV, 
    and ``cp`` is the centipawn score (or None if illegal).
    """

    fen: str
    parse_success: bool
    agent_move: str | None = None
    score: Score | None = None
    cp: int | None = None


def parse_move(text: str, board: chess.Board) -> chess.Move | None:
    """Extract a single legal move from free-form model text.

    Parsing and legality are delegated to python-chess (``parse_uci``/``parse_san``). We
    only clean up the model output: strip an optional ``\\boxed{...}`` wrapper, then try the
    whole answer and each token (in case the move is embedded in a sentence). Returns the
    legal :class:`chess.Move`, or ``None`` if none can be recovered.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    raw = text.strip()
    boxed = re.search(r"\\boxed\{([^{}]*)\}", raw)
    if boxed:
        raw = boxed.group(1).strip()

    for token in [raw, *raw.replace(",", " ").split()]:
        token = token.strip().strip(".")
        if not token:
            continue
        for parse in (board.parse_uci, board.parse_san):
            try:
                return parse(token)
            except ValueError:
                continue

    return None


class MoveEvaluator:
    """Scores a proposed move from a FEN using a persistent local engine.

    Owns a :class:`LocalEngine` and the reward policy from :class:`EngineConfig`. One
    evaluator — hence one engine process — is shared per runner subprocess via
    :meth:`for_process`.
    """

    _process_instance: ClassVar[MoveEvaluator | None] = None
    _process_config: ClassVar[EngineConfig | None] = None

    def __init__(self, config: EngineConfig, engine: LocalEngine | None = None) -> None:
        self.engine: LocalEngine = engine if engine is not None else LocalEngine(config)
        

    @property
    def config(self) -> EngineConfig:
        return self.engine.config

    @classmethod
    def for_process(cls, config: EngineConfig) -> MoveEvaluator:
        """
        Return the per-process shared evaluator, starting its engine on first use.
        The engine is started lazily and closed at interpreter exit. The first call's
        ``config`` wins for the lifetime of the process.
        """
        if cls._process_instance is None:
            cls._process_instance = cls(config)
            cls._process_config = config
            atexit.register(cls._process_instance.close)
        elif config != cls._process_config:
            raise RuntimeError(
                "MoveEvaluator.for_process() was called with a different EngineConfig after the process-local evaluator was already initialized."
            )

        return cls._process_instance

    async def score(self, fen: str, answer_text: str) -> MoveResult:
        """Evaluate ``answer_text`` as a move from ``fen`` via the resulting position's eval.

        The move is applied and the resulting board is evaluated from the moving side's
        perspective (higher = better for the side that moved). An illegal/unparseable move has
        no resulting position, so its ``score`` is ``None``. Turning a :class:`MoveResult`
        into a scalar reward is left to ``rewards.compute_reward``.
        """
        board = chess.Board(fen)
        move = parse_move(answer_text, board)
        if move is None:
            return MoveResult(fen=fen, parse_success=False)

        board.push(move)
        info = await self.engine.evaluate(board)

        if (score := info.get("score")) is None:
            raise RuntimeError(f"Engine analysis did not return a score: {info}")

        # After push(), board.turn is the opponent; score from the mover's POV (higher = better).
        mover_score = score.pov(not board.turn)
        return MoveResult(
            fen=fen,
            parse_success=True,
            agent_move=move.uci(),
            score=mover_score,
            cp=mover_score.score(mate_score=self.config.mate_score),  # for logging/metadata
        )

    def close(self) -> None:
        if self.engine is None:
            return

        self.engine.close()
        self.engine = None  # type: ignore[assignment]

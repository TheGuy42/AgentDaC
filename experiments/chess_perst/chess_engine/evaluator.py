from __future__ import annotations

import atexit
import re
from dataclasses import dataclass
from typing import ClassVar

import chess

from experiments.chess_perst.chess_engine.config import EngineConfig
from experiments.chess_perst.chess_engine.engine import LocalEngine


@dataclass(frozen=True)
class MoveScore:
    """Outcome of scoring one proposed move.

    ``cp``/``agent_move`` are ``None`` for an illegal or unparseable move, which has no
    resulting position to evaluate.
    """

    reward: float
    parse_success: bool
    agent_move: str | None = None
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

    def __init__(self, config: EngineConfig, engine: LocalEngine | None = None) -> None:
        self._config = config
        self._engine = engine if engine is not None else LocalEngine(config)

    @classmethod
    def for_process(cls, config: EngineConfig) -> MoveEvaluator:
        """Return the per-process shared evaluator, starting its engine on first use.

        The engine is started lazily and closed at interpreter exit. The first call's
        ``config`` wins for the lifetime of the process.
        """
        if cls._process_instance is None:
            cls._process_instance = cls(config)
            atexit.register(cls._process_instance.close)
        return cls._process_instance

    async def score(self, fen: str, answer_text: str) -> MoveScore:
        """Score ``answer_text`` as a move from ``fen`` via the resulting position's eval.

        The move is applied and the resulting board is evaluated from the moving side's
        perspective (higher = better for the side that moved). An illegal/unparseable move
        has no resulting position and scores ``config.illegal_score``.
        """
        board = chess.Board(fen)
        move = parse_move(answer_text, board)
        if move is None:
            return MoveScore(reward=self._config.illegal_score, parse_success=False)

        mover = board.turn
        board.push(move)
        white_cp = await self._engine.evaluate(board)
        cp = white_cp if mover == chess.WHITE else -white_cp

        return MoveScore(reward=self._reward(cp), parse_success=True, agent_move=move.uci(), cp=cp)

    def _reward(self, cp: int) -> float:
        """Shape a mover-POV centipawn score into a reward per ``config.reward``."""
        if self._config.reward == "win_prob":
            return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))  # expected score in [0, 1]
        return float(cp)

    def close(self) -> None:
        self._engine.close()

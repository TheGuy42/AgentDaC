from __future__ import annotations

from chess.engine import Cp, Score
from experiments.chess_perst.chess_engine.config import EngineConfig
from experiments.chess_perst.chess_engine.evaluator import MoveResult


def compute_cp(chess_score: Score, config: EngineConfig) -> float:
    """
    Compute the centipawn reward for a given chess score.
    """
    return float(chess_score.score(mate_score=config.mate_score))


def compute_wp(chess_score: Score, config: EngineConfig) -> float:
    """
    Compute the win probability reward for a given chess score.
    Mates occupy reserved end-bands (so mate-in-1 > mate-in-5 > any centipawn eval,
    and a faster mate against you is worse), while non-mate centipawns map through a logistic
    into the middle band [mate_margin, 1 - mate_margin] (win_scale controls saturation: larger = gentler).
    """
    mate_margin = config.mate_margin
    mate_decay = config.mate_decay
    win_scale = config.win_scale

    mate = chess_score.mate()
    if mate is not None:
        # `mate`'s sign is ambiguous at 0: an immediate checkmate the mover just delivered is
        # `MateGiven` (mate()==0, a win) while being mated now is `Mate(0)` (a loss). Decide
        # win-vs-loss from the score's order (> Cp(0)), not the sign of mate().
        plies = abs(mate)
        if chess_score > Cp(0):  # mover delivers mate (incl. MateGiven) -> top band, faster = higher
            return max(min(1.0, 1.0 - mate_decay * (plies - 1)), 1.0 - mate_margin + 1e-6)
        # mover gets mated -> bottom band [0.0, margin), slower mate = less bad
        return min(max(0.0, mate_decay * (plies - 1)), mate_margin - 1e-6)

    cp = chess_score.score(mate_score=config.mate_score)
    wp = 1.0 / (1.0 + 10.0 ** (-cp / win_scale))  # expected score in [0, 1]
    return mate_margin + (1.0 - 2.0 * mate_margin) * wp


def compute_reward(result: MoveResult, config: EngineConfig) -> float:
    """
    Map an engine :class:`MoveResult` into a scalar reward per ``config.reward``.
    """

    if result.score is None:
        return config.illegal_score

    if config.reward == "win_prob":
        return compute_wp(result.score, config)

    elif config.reward == "centipawns":
        return compute_cp(result.score, config)

    else:
        raise ValueError(f"Unknown reward type: {config.reward}")

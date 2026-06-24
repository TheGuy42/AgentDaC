from __future__ import annotations

from chess.engine import Cp, Score
from experiments.chess_perst.chess_engine.config import EngineConfig
from experiments.chess_perst.chess_engine.evaluator import MoveResult


def compute_cp(chess_score: Score, config: EngineConfig) -> float:
    """
    Compute the centipawn reward for a given chess score, linearly rescaled to [0, 1].

    Args:
        chess_score (Score): The chess score to convert.
            Should be from our perspective (i.e. positive = good for us).
        config (EngineConfig): The engine configuration.

    Returns:
        float: The rescaled score in [0, 1] (0.5 = equal; mates near 0/1, by distance).
    """
    cp = chess_score.score(mate_score=config.mate_score)
    return (cp + config.mate_score) / (2.0 * config.mate_score)


def compute_wp(chess_score: Score, config: EngineConfig) -> float:
    """
    Compute the win probability reward for a given chess score.
    Returns a value in [0, 1], with the following special handling for mate scores:
    - If the mover delivers mate (positive mate score), return a value in [1 - margin, 1.0], with faster mates closer to 1.0.
        Linear decay is applied to the mate score, so that a mate in 1 is worth more than a mate in 2, etc.
    - If the mover gets mated (negative mate score), return a value in [0.0, margin], with slower mates closer to 0.0.
        Linear decay is applied to the mate score, so that a mate in 1 is worth less than a mate in 2, etc.
    - If the score is a centipawn score, return a value in [margin, 1 - margin] based on the expected score,
        and normalized to the range [0, 1] using a logistic function.


    Args:
        chess_score (Score): The chess score to convert to win probability.
            Should be from our perspective (i.e. positive = good for us).
        config (EngineConfig): The engine configuration.

    Returns:
        float: The win probability in [0, 1], with special handling for mate scores
    """
    mate_margin = config.mate_margin
    mate_decay = config.mate_decay
    win_scale = config.win_scale

    if (mate := chess_score.mate()) is not None:
        plies = abs(mate)
        if chess_score > Cp(0):
            # mover delivers mate (incl. MateGiven) -> top band, faster = higher
            return max(min(1.0, 1.0 - mate_decay * (plies - 1)), 1.0 - mate_margin + 1e-6)
        else:
            # mover gets mated -> bottom band [0.0, margin), slower mate = less bad
            return min(max(0.0, mate_decay * (plies - 1)), mate_margin - 1e-6)

    cp = chess_score.score(mate_score=config.mate_score)
    wp = 1.0 / (1.0 + 10.0 ** (-cp / win_scale))
    return mate_margin + (1.0 - 2.0 * mate_margin) * wp


def compute_reward(result: MoveResult, config: EngineConfig) -> float:
    """
    Map an engine :class:`MoveResult` into a scalar reward per `config.reward`.
    """

    if result.score is None:
        return -config.mate_margin

    if config.reward == "win_prob":
        return compute_wp(result.score, config)

    elif config.reward == "centipawns":
        return compute_cp(result.score, config)

    else:
        raise ValueError(f"Unknown reward type: {config.reward}")

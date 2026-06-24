from __future__ import annotations

from experiments.chess_perst.chess_engine.config import EngineConfig
from experiments.chess_perst.chess_engine.evaluator import MoveResult


def compute_reward(result: MoveResult, config: EngineConfig) -> float:
    """Map an engine :class:`MoveResult` into a scalar reward per ``config.reward``.

    An illegal/unparseable move has no resulting position (``result.score is None``) and
    scores ``config.illegal_score``. Otherwise the mover-POV engine score is shaped:

    - ``win_prob``: mates occupy reserved end-bands (so mate-in-1 > mate-in-5 > any centipawn
      eval, and a faster mate against you is worse), while non-mate centipawns map through a
      logistic into the middle band ``[mate_margin, 1 - mate_margin]`` (``win_scale`` controls
      saturation: larger = gentler).
    - ``centipawns``: the raw mover-POV centipawn score.
    """
    if result.score is None:
        return config.illegal_score

    score = result.score
    if config.reward != "win_prob":
        return float(score.score(mate_score=config.mate_score))

    margin = config.mate_margin
    mate = score.mate()
    if mate is not None:
        if mate > 0:  # mover delivers mate -> top band (1 - margin, 1.0], faster = higher
            return max(1.0 - config.mate_decay * (mate - 1), 1.0 - margin + 1e-6)
        # mover gets mated -> bottom band [0.0, margin), slower mate = less bad
        return min(config.mate_decay * (-mate - 1), margin - 1e-6)

    cp = score.score()
    wp = 1.0 / (1.0 + 10.0 ** (-cp / config.win_scale))  # expected score in [0, 1]
    return margin + (1.0 - 2.0 * margin) * wp

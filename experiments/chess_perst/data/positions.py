from __future__ import annotations
import random
from typing import Callable

import chess
import chess.engine
from tqdm import tqdm
from src.utils.logging import create_logger


logger = create_logger(__name__)


# A game source plays out a board to (at most) ``target_ply`` plies given an RNG.
GameSource = Callable[[random.Random, int], chess.Board]


def random_game(rng: random.Random, target_ply: int) -> chess.Board:
    """Play ``target_ply`` uniformly-random legal moves from the start position."""
    board = chess.Board()
    for _ in range(target_ply):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


def make_engine_game(engine: chess.engine.SimpleEngine, limit: chess.engine.Limit, epsilon: float) -> GameSource:
    """Build a game source where the engine plays, with ``epsilon`` random moves mixed in.

    Engine-guided play reaches realistic mid-game positions; the random fraction keeps the
    openings diverse so the dataset isn't a handful of repeated engine lines.
    """

    def play(rng: random.Random, target_ply: int) -> chess.Board:
        board = chess.Board()
        for _ in range(target_ply):
            if board.is_game_over():
                break
            if rng.random() < epsilon:
                board.push(rng.choice(list(board.legal_moves)))
            else:
                board.push(engine.play(board, limit).move)
        return board

    return play


def generate_positions(num: int, min_ply: int, max_ply: int, seed: int, source: GameSource) -> list[dict]:
    """Generate ``num`` distinct non-terminal positions by playing games via ``source``.

    Each game runs for a random number of plies in ``[min_ply, max_ply]``. Terminal
    positions (checkmate/stalemate/draw) are skipped and duplicates (by FEN) are dropped,
    so the dataset is self-contained and reproducible from ``seed``.
    """
    rng = random.Random(seed)
    positions: list[dict] = []
    seen: set[str] = set()

    max_attempts = max(num * 20, 1000)
    attempts = 0

    with tqdm(total=num, desc="Generating positions") as pbar:
        while len(positions) < num and attempts < max_attempts:
            attempts += 1
            board = source(rng, rng.randint(min_ply, max_ply))

            if board.is_game_over():
                continue

            fen = board.fen()
            if fen in seen:
                continue
            seen.add(fen)
            positions.append({"fen": fen, "ply": board.ply()})
            pbar.update(1)

    if len(positions) < num:
        logger.warning(f"Only generated {len(positions)}/{num} positions after {attempts} attempts.")

    return positions

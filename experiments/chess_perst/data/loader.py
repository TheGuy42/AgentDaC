from __future__ import annotations

import pathlib
import random
from enum import Enum
from typing import Callable

import chess
import datasets
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

from src.utils.logging import create_logger


logger = create_logger(__name__)

# Each loader caches its materialized rows here so repeated runs skip re-streaming.
CACHE_DIR = pathlib.Path(__file__).parent / "cache"


class ChessDataset(str, Enum):
    PUZZLES = "lichess-puzzles"
    OPENINGS = "lichess-openings"
    EVALS = "lichess-evals"


SUPPORTED_DATASETS = [e.value for e in ChessDataset]

# Cap for the streaming shuffle window. The buffer holds this many fully-streamed rows in
# memory at once, so it must stay bounded regardless of how many rows we ultimately want; a
# window this size decorrelates the stream plenty for our purposes.
MAX_SHUFFLE_BUFFER = 100_000


def _shuffle_buffer(limit: int) -> int:
    """A bounded shuffle-buffer size — never larger than ``MAX_SHUFFLE_BUFFER``."""
    return min(max(limit, 10_000), MAX_SHUFFLE_BUFFER)


def _board_row(board: chess.Board) -> dict:
    """The normalized row schema shared by every chess dataset."""
    return {"fen": board.fen(), "ply": board.ply()}


def _materialize(
    it_ds: IterableDataset,
    limit: int,
    map_fn: Callable[[dict], dict | None],
    desc: str,
) -> list[dict]:
    """Stream rows through ``map_fn`` (which returns None to skip) until ``limit`` collected."""
    rows: list[dict] = []
    with tqdm(total=limit, desc=desc) as pbar:
        for example in it_ds:
            row = map_fn(example)
            if row is None:
                continue
            rows.append(row)
            pbar.update(1)
            if len(rows) >= limit:
                break
    return rows


def _rating_suffix(min_rating: int | None, max_rating: int | None) -> str:
    """Cache-name suffix encoding a rating filter (empty when none is set)."""
    if min_rating is None and max_rating is None:
        return ""
    return f"_r{min_rating if min_rating is not None else 'min'}-{max_rating if max_rating is not None else 'max'}"


def _build_and_cache(rows: list[dict], cache_path: pathlib.Path) -> Dataset:
    """Wrap materialized rows in a ``Dataset``, save it to ``cache_path``, and return it."""
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(cache_path))
    logger.info(f"Saved {len(ds)} positions to disk at {cache_path}")
    return ds


def load_puzzles(limit: int, seed: int, min_rating: int | None = None, max_rating: int | None = None) -> Dataset:
    """
    Tactical positions from ``Lichess/chess-puzzles``. The dataset's ``FEN`` is the position
    *before* the opponent's setup move, so we apply ``Moves[0]`` to reach the position the
    solver actually faces. Optionally filtered to a Glicko-2 rating range.
    """
    cache_path = CACHE_DIR / f"lichess-puzzles_n{limit}_s{seed}{_rating_suffix(min_rating, max_rating)}"
    if cache_path.exists():
        logger.info(f"Loading dataset from disk at {cache_path}")
        return Dataset.load_from_disk(str(cache_path))

    def map_fn(example: dict) -> dict | None:
        rating = example["Rating"]
        if (min_rating is not None and rating < min_rating) or (max_rating is not None and rating > max_rating):
            return None
        moves = example["Moves"].split()
        if not moves:
            return None
        board = chess.Board(example["FEN"])
        board.push_uci(moves[0])
        if board.is_game_over():
            return None
        return _board_row(board)

    ds: IterableDataset = datasets.load_dataset("Lichess/chess-puzzles", split="train", streaming=True)  # type: ignore[assignment]
    ds = ds.select_columns(["FEN", "Moves", "Rating"])
    ds = ds.shuffle(seed=seed, buffer_size=_shuffle_buffer(limit))
    items = _materialize(ds, limit, map_fn, "Loading lichess-puzzles")
    return _build_and_cache(items, cache_path)


def load_openings(limit: int, seed: int) -> Dataset:
    """
    Opening positions from ``Lichess/chess-openings`` (the ECO book). We replay the ``uci``
    line from the start so the FEN carries correct move counters and ply.
    """
    cache_path = CACHE_DIR / f"lichess-openings_n{limit}_s{seed}"
    if cache_path.exists():
        logger.info(f"Loading dataset from disk at {cache_path}")
        return Dataset.load_from_disk(str(cache_path))

    def map_fn(example: dict) -> dict | None:
        board = chess.Board()
        for uci in example["uci"].split():
            board.push_uci(uci)
        if board.is_game_over():
            return None
        return _board_row(board)

    ds: IterableDataset = datasets.load_dataset("Lichess/chess-openings", split="train", streaming=True)  # type: ignore[assignment]
    ds = ds.select_columns(["uci"])  # skip the heavy ``img`` column
    ds = ds.shuffle(seed=seed, buffer_size=_shuffle_buffer(limit))
    items = _materialize(ds, limit, map_fn, "Loading lichess-openings")
    return _build_and_cache(items, cache_path)


def load_evals(limit: int, seed: int) -> Dataset:
    """
    Stockfish-analyzed positions from ``Lichess/chess-position-evaluations`` (~342M unique).
    Its ``fen`` is an EPD (no move counters), so we pad it to a full FEN; ``ply`` therefore
    only reflects the side to move, not the true game ply. We just stream ``limit`` of them.
    """
    cache_path = CACHE_DIR / f"lichess-evals_n{limit}_s{seed}"
    if cache_path.exists():
        logger.info(f"Loading dataset from disk at {cache_path}")
        return Dataset.load_from_disk(str(cache_path))

    def map_fn(example: dict) -> dict | None:
        fen = example["fen"]
        if fen.count(" ") == 3:  # EPD -> append halfmove clock / fullmove number
            fen = f"{fen} 0 1"
        board = chess.Board(fen)
        if board.is_game_over():
            return None
        return _board_row(board)

    ds: IterableDataset = datasets.load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)  # type: ignore[assignment]
    ds = ds.select_columns(["fen"])
    ds = ds.shuffle(seed=seed, buffer_size=_shuffle_buffer(limit))
    items = _materialize(ds, limit, map_fn, "Loading lichess-evals")
    return _build_and_cache(items, cache_path)


def load_raw_dataset(name: str, *, limit: int, seed: int, min_rating: int | None = None, max_rating: int | None = None) -> Dataset:
    """Load (and cache) up to ``limit`` normalized rows from a single supported dataset."""
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: {SUPPORTED_DATASETS}")

    if name == ChessDataset.PUZZLES:
        return load_puzzles(limit, seed, min_rating, max_rating)

    elif name == ChessDataset.OPENINGS:
        return load_openings(limit, seed)

    elif name == ChessDataset.EVALS:
        return load_evals(limit, seed)

    else:
        raise ValueError(f"Unsupported dataset: {name}")


def load_dataset(
    names: str | list[str],
    *,
    num_train: int,
    num_val: int,
    seed: int = 1234,
    min_rating: int | None = None,
    max_rating: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load chess positions from one or more datasets and return disjoint train/val splits.

    Each row has a ``fen`` (the non-terminal position to move in) and the ``ply`` it occurs
    at. When multiple datasets are given they are pooled together, deduplicated by FEN, and
    shuffled before splitting, so a split mixes positions from all of them.

    Args:
        names: A dataset name (or list of names) from ``SUPPORTED_DATASETS``.
        num_train / num_val: Sizes of the (disjoint) train and validation splits.
        seed: Seed for reproducible streaming/shuffling.
        min_rating / max_rating: Puzzle rating bounds (``lichess-puzzles`` only).

    Returns:
        tuple[Dataset, Dataset]: The train and validation ``Dataset`` splits.
    """
    if isinstance(names, str):
        names = [names]
    if not names:
        raise ValueError("At least one dataset name must be provided.")

    total = num_train + num_val

    # Pull up to ``total`` rows from each dataset (each loader caches its own) so a single
    # source can fill the request.
    pool: list[dict] = []
    for i, name in enumerate(names):
        ds = load_raw_dataset(name, limit=total, seed=seed + i, min_rating=min_rating, max_rating=max_rating)
        pool.extend(ds.to_list())

    # Deduplicate by FEN so train/val never share a position.
    seen: set[str] = set()
    unique: list[dict] = []
    for row in pool:
        if row["fen"] in seen:
            continue
        seen.add(row["fen"])
        unique.append(row)

    random.Random(seed).shuffle(unique)

    if len(unique) < total:
        logger.warning(f"Only {len(unique)} unique positions available; requested {total} ({num_train} train + {num_val} val).")

    val_rows = unique[:num_val]
    train_rows = unique[num_val : num_val + num_train]
    logger.info(f"Loaded {len(train_rows)} train / {len(val_rows)} val positions from {names}.")
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)

from __future__ import annotations

import argparse
import pathlib
from typing import Literal

import chess.engine
from datasets import Dataset

from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine import EngineConfig
from experiments.chess_perst.data import positions


logger = create_logger(__name__)

Source = Literal["random", "engine"]
GENERATED_DIR = pathlib.Path(__file__).parent / "generated"
DEFAULT_EPSILON = 0.25


def dataset_path(split: str, source: Source, num: int, min_ply: int, max_ply: int, seed: int) -> pathlib.Path:
    return GENERATED_DIR / f"{split}_{source}_{num}_{min_ply}-{max_ply}_{seed}.parquet"


def generate(
    source: Source,
    num: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    engine_config: EngineConfig | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> list[dict]:
    """Generate positions with the given source (opening an engine only when needed)."""
    if source == "random":
        return positions.generate_positions(num, min_ply, max_ply, seed, positions.random_game)

    if source == "engine":
        config = engine_config or EngineConfig(limit=chess.engine.Limit(depth=12))
        engine = chess.engine.SimpleEngine.popen_uci(config.engine_path)
        try:
            engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
            game = positions.make_engine_game(engine, config.limit, epsilon)
            return positions.generate_positions(num, min_ply, max_ply, seed, game)
        finally:
            engine.quit()

    raise ValueError(f"Unknown position source: {source!r}")


def build_and_save(
    split: str,
    source: Source,
    num: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    engine_config: EngineConfig | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Dataset:
    """Generate a split, cache it as parquet under ``generated/``, and return it."""
    path = dataset_path(split, source, num, min_ply, max_ply, seed)
    rows = generate(source, num, min_ply, max_ply, seed, engine_config, epsilon)
    dataset = Dataset.from_list(rows)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    logger.info(f"Saved {len(dataset)} positions to {path}")
    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and cache chess position datasets.")
    parser.add_argument("--source", choices=["random", "engine"], default="random")
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=200)
    parser.add_argument("--min_ply", type=int, default=8)
    parser.add_argument("--max_ply", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Random-move fraction for the engine source.")
    parser.add_argument("--engine_path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    engine_config = EngineConfig(engine_path=args.engine_path, limit=chess.engine.Limit(depth=args.depth))
    build_and_save("train", args.source, args.num_train, args.min_ply, args.max_ply, args.seed, engine_config, args.epsilon)
    build_and_save("val", args.source, args.num_val, args.min_ply, args.max_ply, args.seed + 1, engine_config, args.epsilon)


if __name__ == "__main__":
    main()

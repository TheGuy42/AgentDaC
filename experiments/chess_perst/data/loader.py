from __future__ import annotations

from datasets import Dataset

from src.utils.logging import create_logger
from experiments.chess_perst.chess_engine import EngineConfig
from experiments.chess_perst.data.build import Source, build_and_save, dataset_path


logger = create_logger(__name__)


def load_dataset(
    split: str,
    *,
    source: Source,
    num: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    engine_config: EngineConfig | None = None,
) -> Dataset:
    """Load a cached position split, building and caching it on first use."""
    path = dataset_path(split, source, num, min_ply, max_ply, seed)
    if path.exists():
        logger.info(f"Loading cached positions from {path}")
        return Dataset.from_parquet(str(path))
    return build_and_save(split, source, num, min_ply, max_ply, seed, engine_config)


__all__ = ["load_dataset"]

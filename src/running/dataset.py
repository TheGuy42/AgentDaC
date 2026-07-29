from __future__ import annotations
import functools
from abc import ABC, abstractmethod
from typing import Any, Mapping
import datasets
from src.running.stage import RolloutStage


class TaskDataset(ABC):
    """A source of samples for one experiment.

    Subclasses implement `load`, reading their parameters from `self.params`.
    Sources are read whole and then partitioned, so `load` returns every split at once and
    runs at most once per instance; `load_split` and `prepare_split` read from its cache.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def load(self) -> Mapping[RolloutStage, datasets.Dataset]:
        """Read the source once and return the rows of every stage it provides.

        Keys must be `RolloutStage` members; a dataset need not provide all of them.
        """

    @functools.cached_property
    def splits(self) -> dict[RolloutStage, datasets.Dataset]:
        """Every stage this dataset provides, loaded once on first access."""
        splits = dict(self.load())
        return {RolloutStage(stage): ds for stage, ds in splits.items()}

    def load_split(self, stage: RolloutStage) -> datasets.Dataset:
        """The rows of one stage, loading the source on first access."""
        if stage not in self.splits:
            raise ValueError(f"{type(self).__name__} provides no {str(stage)!r} split; available: {sorted(s.value for s in self.splits)}.")
        return self.splits[stage]

    def prepare_split(self, stage: RolloutStage, size: int | None = None, seed: int = 0) -> datasets.Dataset:
        """`load_split` + deterministic shuffle + size slice."""
        ds = self.load_split(stage).shuffle(seed=seed)
        if size is not None:
            ds = ds.select(range(min(int(size), len(ds))))
        return ds

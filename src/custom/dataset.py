from __future__ import annotations
import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset
from abc import ABC, abstractmethod


class DynamicDataset(RLHFDataset, ABC):
    """`RLHFDataset` that builds its rows in-worker from a source instead of parquet.

    Subclasses implement :meth:`load_split`; this base handles the split marker, the
    deterministic shuffle + size slice, the AgentLoop carrier columns, and the
    inherited prompt-length filtering.
    """

    def __init__(self, data_files: str | list[str], *args, **kwargs):

        if isinstance(data_files, str):
            data_files = [data_files]

        self.split_name = data_files[0]
        if self.split_name not in ("train", "val", "test"):
            raise ValueError(f"Expected split to be 'train', 'val', or 'test', got {self.split_name}")

        super().__init__(data_files, *args, **kwargs)

    def _download(self, use_origin_parquet: bool = False) -> None:
        return  # data is built from a source, nothing to fetch

    def _read_files_and_tokenize(self) -> None:
        # data_files is the split marker ("train"/"val") set in the verl config;
        cfg = self.config.custom_dataset
        ds = self.load_split(self.split_name)

        # Deterministic shuffle + size slice (previously done in ExperimentRunner._main).
        ds = ds.shuffle(seed=cfg.seed)
        size = cfg.train_size if self.split_name == "train" else cfg.val_size
        if size is not None:
            ds = ds.select(range(min(int(size), len(ds))))

        # AgentLoop carrier columns (stamped non-destructively: never clobber a source column).
        def _add_columns(row: dict) -> dict:
            return {
                # read by VerlTrainer._stage()
                "training_stage": self.split_name,
                # verl namespaces val metrics: val-core/<data_source>/...
                "data_source": cfg.data_source,
                # Placeholder only so RLHFDataset.__getitem__ can build raw_prompt without a KeyError
                self.prompt_key: row.get(self.prompt_key, [{"role": "user", "content": ""}]),
            }

        ds = ds.map(_add_columns)

        self.dataframe = ds
        print(f"dataset len: {len(self.dataframe)}")

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset | None = None):
        # The rollout prompt is built at agent-loop time (VerlTrainer.format_prompt), not from the
        # dataset 'prompt' column, so length-filtering rows here is meaningless. Forbid it so it is
        # never silently run (it would also require the prompt column to be a list[dict] chat).
        raise RuntimeError(
            "DynamicDataset does not filter by prompt length: the rollout prompt is built at "
            "agent-loop time (VerlTrainer.format_prompt), not from the dataset 'prompt' column."
        )

    @abstractmethod
    def load_split(self, split: str) -> datasets.Dataset:
        """Return the raw source rows for `split` ("train" or "val" or "test").

        Subclasses load from their source (e.g. HuggingFace) and apply any
        experiment-specific filtering, reading params from `self.config.custom_dataset`.
        """
        raise NotImplementedError

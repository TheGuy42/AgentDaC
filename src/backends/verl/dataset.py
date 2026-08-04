# pyright: reportMissingImports=false

from __future__ import annotations
from typing import Any

import datasets
from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.import_utils import load_class_from_fqn

from src.running.dataset import TaskDataset
from src.running.stage import RolloutStage


class VerlDataset(RLHFDataset):
    """`RLHFDataset` that builds its rows in-worker from a `TaskDataset` instead of parquet.

    Registered once as `data.custom_cls` for every experiment: the experiment's own
    `TaskDataset` is named by FQDN in `data.custom_dataset.task_dataset` and imported here.

    This adapter handles the split marker, the AgentLoop carrier columns, and disabling the
    inherited prompt-length filtering. The shuffle + size slice live in `TaskDataset.prepare_split`.
    """

    def __init__(self, data_files: str | list[str], *args, **kwargs):

        if isinstance(data_files, str):
            data_files = [data_files]

        # `data_files` is the split marker set by the runner, not a path; RolloutStage validates it.
        self.stage = RolloutStage(data_files[0])

        super().__init__(data_files, *args, **kwargs)

    def _download(self, use_origin_parquet: bool = False) -> None:
        return  # data is built from a source, nothing to fetch

    def _read_files_and_tokenize(self) -> None:
        # data_files is the split marker ("train"/"val") set in the verl config;
        # `custom_dataset` carries the load params (the dataset only sees `config.data`).
        params: dict[str, Any] = OmegaConf.to_container(self.config.custom_dataset, resolve=True)  # type: ignore[assignment]

        task_dataset_cls = load_class_from_fqn(params["task_dataset"], description="TaskDataset")
        task_dataset: TaskDataset = task_dataset_cls(params)

        size = params["train_size"] if self.stage == RolloutStage.TRAIN else params["val_size"]
        ds = task_dataset.prepare_split(self.stage, size=size, seed=params["seed"])

        # AgentLoop carrier columns (stamped non-destructively: never clobber a source column).
        def _add_columns(row: dict) -> dict:
            return {
                # read by VerlLoop._stage()
                "training_stage": self.stage.value,
                # verl namespaces val metrics: val-core/<data_source>/...
                "data_source": params["data_source"],
                # Placeholder only so RLHFDataset.__getitem__ can build raw_prompt without a KeyError
                self.prompt_key: row.get(self.prompt_key, [{"role": "user", "content": ""}]),
            }

        self.dataframe = ds.map(_add_columns)
        print(f"dataset len: {len(self.dataframe)}")

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset | None = None):
        # The rollout prompt is built at agent-loop time (VerlLoop.format_prompt), not from the
        # dataset 'prompt' column, so length-filtering rows here is meaningless. Forbid it so it is
        # never silently run (it would also require the prompt column to be a list[dict] chat).
        raise RuntimeError(
            "VerlDataset does not filter by prompt length: the rollout prompt is built at "
            "agent-loop time (VerlLoop.format_prompt), not from the dataset 'prompt' column."
        )

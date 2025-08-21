from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum
from src.utils.io import save_base_model


class RolloutStage(str, Enum):
    Train = "train"
    Val = "val"
    Test = "test"


class RolloutConfig(BaseModel):
    kwargs: dict = Field(default_factory=dict)
    train_kwargs: dict = Field(default_factory=dict)
    val_kwargs: dict = Field(default_factory=dict)
    test_kwargs: dict = Field(default_factory=dict)

    def get_kwargs(self, stage: RolloutStage) -> dict:
        kwargs = self.kwargs or {}
        kwargs = kwargs.copy()

        if stage == RolloutStage.Train:
            kwargs.update(self.train_kwargs or {})
        if stage == RolloutStage.Val:
            kwargs.update(self.val_kwargs or {})
        if stage == RolloutStage.Test:
            kwargs.update(self.test_kwargs or {})
        return kwargs

    def save(self, dir_name: str, file_name: str = "rollout_config.json") -> None:
        """
        Save the rollout configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

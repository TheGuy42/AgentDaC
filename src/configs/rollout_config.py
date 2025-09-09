from __future__ import annotations
from pydantic import Field
from src.utils.logging import create_logger
from src.configs.base_config import BaseConfig


logger = create_logger(__name__)


class RolloutConfig(BaseConfig):
    kwargs: dict = Field(default_factory=dict)
    train_kwargs: dict = Field(default_factory=dict)
    val_kwargs: dict = Field(default_factory=dict)
    test_kwargs: dict = Field(default_factory=dict)

    def get_kwargs(self, stage: str) -> dict:
        kwargs = self.kwargs or {}
        kwargs = kwargs.copy()

        if stage == "train":
            kwargs.update(self.train_kwargs or {})
        elif stage == "val":
            kwargs.update(self.val_kwargs or {})
        elif stage == "test":
            kwargs.update(self.test_kwargs or {})
        else:
            logger.warning(f"Unknown stage '{stage}'. Using base kwargs only.")
        return kwargs

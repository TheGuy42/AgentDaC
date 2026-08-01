from pydantic import Field

from src.configs.base_config import BaseConfig


class DataConfig(BaseConfig, extra="allow"):
    """Dataset sizes plus experiment-specific loader parameters."""

    train_size: int | None = Field(default=None, ge=0)
    val_size: int | None = Field(default=None, ge=0)
    test_size: int | None = Field(default=None, ge=0)
    seed: int = 0

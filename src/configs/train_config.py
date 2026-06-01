from typing import Any
from src.utils.logging import create_logger
from src.configs.base_config import BaseConfig


logger = create_logger(__name__)


class TrainingConfig(BaseConfig, extra="allow"):
    n_runners: int = 1
    train_size: int | None = None
    val_size: int | None = None
    verl_config: dict[str, Any]

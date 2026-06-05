from src.utils.logging import create_logger
from src.configs.base_config import BaseConfig


logger = create_logger(__name__)


class TrainingConfig(BaseConfig):
    n_runners: int = 16
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None

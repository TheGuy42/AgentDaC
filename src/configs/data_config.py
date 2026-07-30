from src.configs.base_config import BaseConfig


class DataConfig(BaseConfig, extra="allow"):
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    seed: int = 0

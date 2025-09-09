from pydantic import Field
import art
from src.utils.logging import create_logger
from src.configs.base_config import BaseConfig


logger = create_logger(__name__)


class RulerConfig(BaseConfig):
    judge_model: str | None = None
    extra_litellm_params: dict | None = None
    rubric: str | None = None
    swallow_exceptions: bool = True
    debug: bool = False


class TrainingConfig(BaseConfig, extra="allow"):
    epochs: int = 1
    num_groups: int = 12
    group_size: int = 8

    train_size: int | None = None
    val_log_steps: int = 5
    val_size: int | None = None
    delete_checkpoints: bool = True
    checkpoint_metric: str = "reward"

    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None
    ruler_config: RulerConfig | None = Field(default_factory=RulerConfig)

    verbose: bool = False
    max_exceptions: int | float = 0

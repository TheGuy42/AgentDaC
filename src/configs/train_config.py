from pydantic import BaseModel, Field
from pathlib import Path

import art
from src.utils.logging import create_logger
from src.utils.io import save_base_model


logger = create_logger(__name__)


class RulerConfig(BaseModel):
    judge_model: str | None = None
    extra_litellm_params: dict | None = None
    rubric: str | None = None
    swallow_exceptions: bool = True
    debug: bool = False


class TrainingConfig(BaseModel, extra="allow"):
    epochs: int = 1
    num_groups: int = 12
    group_size: int = 8
    min_reward_stdev: float | None = None

    train_log_steps: int = 1
    train_size: int | None = None
    val_log_steps: int = 5
    val_size: int | None = None
    delete_checkpoints: bool = True
    checkpoint_metric: str = "reward"

    rollout_kwargs: dict = Field(default_factory=dict)
    art_config: art.types.TrainConfig = Field(default_factory=art.types.TrainConfig)
    dev_art_config: art.dev.train.TrainConfig | None = None
    ruler_config: RulerConfig | None = Field(default_factory=RulerConfig)

    verbose: bool = False
    max_exceptions: int | float = 0

    def save(self, dir_name: str, file_name: str = "train_config.json") -> None:
        """
        Save the training configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

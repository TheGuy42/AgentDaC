from pydantic import Field
from typing import Literal
from src.utils.logging import create_logger
from src.configs.base_config import BaseConfig


logger = create_logger(__name__)


class RulerParams(BaseConfig):
    judge_model: str | None = None
    extra_litellm_params: dict | None = None
    rubric: str | None = None
    swallow_exceptions: bool = True
    debug: bool = False


class TrainParams(BaseConfig):
    # Core training parameters
    learning_rate: float = 5e-6
    # KL-penalized advantage adjustment
    kl_penalty_coef: float = 0.0
    kl_penalty_reference_step: int | None = None
    kl_ref_adapter_path: str | None = None
    # RL algorithm settings
    ppo: bool = False
    epsilon: float | None = None
    epsilon_high: float | None = None
    # Advantage computation
    advantage_balance: float = 0.0
    scale_rewards: bool = True
    # Importance sampling
    importance_sampling_level: Literal["token", "sequence", "average", "geometric_average"] = "token"
    max_negative_advantage_importance_sampling_weight: float | None = None
    mask_prob_ratio: bool = False
    # Experimental parameters
    kimi_k2_tau: float | None = None
    precalculate_logprobs: bool = False
    # LocalBackend-specific parameters
    allow_training_without_logprobs: bool = False
    plot_tensors: bool = False
    truncated_importance_sampling: float | None = None
    scale_learning_rate_by_reward_std_dev: bool = False
    logprob_calculation_chunk_size: int = 1024
    num_trajectories_learning_rate_multiplier_power: float = 0.0


class TrainingConfig(BaseConfig, extra="allow"):
    epochs: int = 1
    num_groups: int = 12
    group_size: int = 8

    train_size: int | None = None
    val_log_steps: int = 5
    val_size: int | None = None
    delete_checkpoints: bool = True
    checkpoint_metric: str = "reward"

    train_params: TrainParams = Field(default_factory=TrainParams)
    ruler_params: RulerParams | None = Field(default_factory=RulerParams)

    verbose: bool = False
    max_exceptions: int | float = 0

# pyright: reportMissingImports=false

from __future__ import annotations
import os
from typing import Literal

from pydantic import Field
from art.dev import EngineArgs, InternalModelConfig, OpenAIServerConfig, ServerArgs
from art.dev.get_model_config import get_model_config

from src.configs.base_config import BaseConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


class ModelConfig(BaseConfig, frozen=False, extra="allow"):
    """Everything `art.TrainableModel` and its vLLM server need.

    There are two independent parser axes here; neither derives from the other:

    * `tool_parser` / `reasoning_parser` -- CLIENT side, used to build our `NativeParser`.
    * `openai_config["server_args"]`     -- SERVER side, vLLM's own parsing.
    """

    base_model: str
    internal_config: InternalModelConfig = Field(default_factory=InternalModelConfig)
    openai_config: OpenAIServerConfig | None = None

    tool_parser: str | None = None
    reasoning_parser: str | None = None

    def initialize(self, output_dir: str, port: int | None = None, seed: int | None = None) -> ModelConfig:
        """
        Fill in the ART/vLLM defaults that depend on the run (output dir, port, seed).

        Args:
            output_dir (str): The output directory for the model.
            port (int | None): The port for the vLLM server. If None, a free port will be found automatically.
            seed (int | None): The random seed for the model. If None, the default seed will be used.

        Returns:
            ArtModelConfig: The initialized configuration with defaults filled in.
        """

        self.internal_config = get_model_config(
            base_model=self.base_model,
            output_dir=output_dir,
            config=self.internal_config,
        )

        if self.openai_config is None:
            self.openai_config = OpenAIServerConfig()
        self.openai_config.setdefault("server_args", ServerArgs())
        self.openai_config.setdefault("engine_args", EngineArgs())

        self.internal_config["engine_args"].setdefault("seed", 0)  # type: ignore[index]

        # ART defaults to `enable_auto_tool_choice=True` + `tool_call_parser="hermes"`
        self.openai_config["server_args"].setdefault("enable_auto_tool_choice", None)  # type: ignore[index]
        self.openai_config["server_args"].setdefault("tool_call_parser", None)  # type: ignore[index]
        self.openai_config["server_args"].setdefault("reasoning_parser", None)  # type: ignore[index]

        if port is not None:
            self.openai_config["server_args"]["port"] = port  # type: ignore[index]

        if seed is not None:
            self.internal_config["init_args"]["random_state"] = seed  # type: ignore[index]
            self.internal_config["engine_args"]["seed"] = seed  # type: ignore[index]
            self.internal_config["peft_args"]["random_state"] = seed  # type: ignore[index]
            self.internal_config["trainer_args"]["seed"] = seed  # type: ignore[index]
            self.internal_config["trainer_args"]["data_seed"] = seed  # type: ignore[index]
            self.openai_config["engine_args"]["seed"] = seed  # type: ignore[index]

        if api_key := os.getenv("OPENAI_API_KEY"):
            self.openai_config["server_args"]["api_key"] = api_key  # type: ignore[index]

        return self


class TrainArgs(BaseConfig):
    """Exact `LocalBackend.train()` arguments."""

    learning_rate: float = 5e-6
    kl_penalty_coef: float = 0.0
    kl_penalty_reference_step: int | None = None
    kl_ref_adapter_path: str | None = None
    ppo: bool = False
    epsilon: float | None = None
    epsilon_high: float | None = None
    advantage_balance: float = 0.0
    scale_rewards: bool = True
    importance_sampling_level: Literal["token", "sequence", "average", "geometric_average"] = "token"
    max_negative_advantage_importance_sampling_weight: float | None = None
    mask_prob_ratio: bool = False
    kimi_k2_tau: float | None = None
    precalculate_logprobs: bool = False
    allow_training_without_logprobs: bool = False
    plot_tensors: bool = False
    truncated_importance_sampling: float | None = None
    scale_learning_rate_by_reward_std_dev: bool = False
    logprob_calculation_chunk_size: int = 1024
    num_trajectories_learning_rate_multiplier_power: float = 0.0


class ArtTrainConfig(BaseConfig, frozen=False):
    epochs: int = Field(default=1, ge=1)
    num_groups: int = Field(default=12, ge=1)
    train_episodes: bool = False
    group_size: int = Field(default=8, ge=1)
    val_log_steps: int = Field(default=5, ge=1)
    delete_checkpoints: bool = True
    checkpoint_metric: str = "reward"
    max_exceptions: int | float = Field(default=0, ge=0)
    verbose: bool = False

    train_params: TrainArgs = Field(default_factory=TrainArgs)


class ArtConfig(BaseConfig, frozen=False):
    model: ModelConfig
    train: ArtTrainConfig = Field(default_factory=ArtTrainConfig)

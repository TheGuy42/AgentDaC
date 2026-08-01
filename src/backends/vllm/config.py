from __future__ import annotations

from pydantic import Field

from src.configs.base_config import BaseConfig
from src.running.stage import RolloutStage


class AgentConfig(BaseConfig):
    """Client-side parsing settings used to build the agent."""

    tool_parser: str | None = None
    reasoning_parser: str | None = None
    tokenizer: str | None = None
    """Tokenizer name for the parsers. Defaults to the served model name."""


class ServerConfig(BaseConfig, frozen=False):
    """How to reach the running `vllm serve`."""

    model_name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"


class InferenceConfig(BaseConfig, frozen=False):
    splits: list[RolloutStage] = Field(default_factory=lambda: [RolloutStage.TEST])
    group_size: int = 1
    max_concurrency: int = 512


class VllmConfig(BaseConfig, frozen=False):
    server: ServerConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    wandb_project: str | None = None

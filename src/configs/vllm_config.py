from __future__ import annotations
from pydantic import Field, model_validator
import os
from art.dev import OpenAIServerConfig, get_openai_server_config
from src.configs.base_config import BaseConfig


class VllmConfig(BaseConfig, frozen=False, extra="allow"):
    """
    Configuration for a served vLLM model.
    """

    id: str = ""  # NOTE: not supported yet
    base_model: str
    openai_config: OpenAIServerConfig = Field(default_factory=OpenAIServerConfig)

    @model_validator(mode="after")
    def validate_identifier(self) -> VllmConfig:
        """
        Validate that the identifier is set to the base model name if not provided.
        """
        if self.id == "" or self.id is None:
            self.id = self.base_model
        return self

    def initialize(self, port: int, seed: int | None = None) -> VllmConfig:
        self.openai_config = get_openai_server_config(
            model_name=self.base_model,
            base_model=self.base_model,
            log_file="",
            lora_path=None,
            config=self.openai_config,
        )
        self.openai_config["server_args"]["port"] = port  # type: ignore
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.openai_config["server_args"]["api_key"] = api_key  # type: ignore
        self.openai_config["engine_args"]["enable_lora"] = True  # type: ignore
        self.openai_config["engine_args"]["seed"] = seed or 0  # type: ignore
        self.openai_config["engine_args"]["num_scheduler_steps"] = 1  # type: ignore
        return self

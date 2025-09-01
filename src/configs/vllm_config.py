from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import os

from art.dev import OpenAIServerConfig, get_openai_server_config
from src.utils.io import save_base_model


class VllmConfig(BaseModel, frozen=False, extra="allow"):
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
        self.openai_config["engine_args"]["num_scheduler_steps"] = 1 # type: ignore
        return self

    def save(self, dir_name: str, file_name: str = "vllm_config.json") -> None:
        """
        Save the vLLM configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

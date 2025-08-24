from __future__ import annotations
import os
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from art.dev.get_model_config import get_model_config
from art.dev import InternalModelConfig, EngineArgs, OpenAIServerConfig, ServerArgs

from src.utils.io import save_base_model


class ArtConfig(BaseModel, frozen=False, extra="allow"):
    """
    Configuration for an ART model.
    """

    id: str = ""  # NOTE: not supported yet
    base_model: str
    internal_config: InternalModelConfig = Field(default_factory=InternalModelConfig)
    openai_config: OpenAIServerConfig | None = None

    @model_validator(mode="after")
    def validate_identifier(self) -> ArtConfig:
        """
        Validate that the identifier is set to the base model name if not provided.
        """
        if not self.id:
            self.id = self.base_model
        return self

    def initialize(self, output_dir: str, seed: int | None = None) -> ArtConfig:
        self.internal_config = get_model_config(
            base_model=self.base_model,
            output_dir=output_dir,
            config=self.internal_config,
        )

        if self.openai_config is None:
            self.openai_config = OpenAIServerConfig()
        self.openai_config.setdefault("server_args", ServerArgs())
        self.openai_config.setdefault("engine_args", EngineArgs())

        self.internal_config["engine_args"].setdefault("seed", 0)  # type: ignore

        if seed is not None:
            self.internal_config["init_args"]["random_state"] = seed # type: ignore
            self.internal_config["engine_args"]["seed"] = seed # type: ignore
            self.internal_config["peft_args"]["random_state"] = seed # type: ignore
            self.internal_config["trainer_args"]["seed"] = seed # type: ignore
            self.internal_config["trainer_args"]["data_seed"] = seed # type: ignore
            self.openai_config["engine_args"]["seed"] = seed  # type: ignore  
            
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.openai_config["server_args"]["api_key"] = api_key  # type: ignore

        return self

    def save(self, dir_name: str, file_name: str = "art_config.json") -> None:
        save_base_model(self, Path(dir_name) / file_name)

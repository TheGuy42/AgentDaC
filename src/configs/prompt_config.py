from __future__ import annotations
from typing import Literal
from pathlib import Path
from pydantic import Field

from src.configs.base_config import BaseConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


def read_prompt(file_path: str | None, encoding: str = "utf-8") -> str | None:
    if file_path is None:
        return None
    return Path(file_path).read_text(encoding=encoding).strip()


class ToolArg(BaseConfig):
    type: Literal["string"] = "string"
    description: str
    required: bool = True


class ToolSpec(BaseConfig):
    name: str
    description: str
    arguments: dict[str, ToolArg] = Field(default_factory=dict)


class ToolSpecs(BaseConfig):
    tools: dict[str, ToolSpec]


class PromptConfig(BaseConfig):
    mode: Literal["text", "path"] = "path"
    system_root: str | None = None
    system_inter: str | None = None
    system_leaf: str | None = None
    tools: str | ToolSpecs | None = None

    def initialize(self, encoding: str = "utf-8") -> PromptConfig:
        if not self.mode == "path":
            return self

        self.system_root = read_prompt(self.system_root, encoding)
        self.system_inter = read_prompt(self.system_inter, encoding)
        self.system_leaf = read_prompt(self.system_leaf, encoding)
        if isinstance(self.tools, str):
            self.tools = ToolSpecs.load_from_path(self.tools, do_raise=True)
        self.mode = "text"
        return self

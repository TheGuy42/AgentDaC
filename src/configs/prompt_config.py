from __future__ import annotations
from typing import Literal
from pathlib import Path
from src.configs.base_config import BaseConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


def read_prompt(file_path: str | None, encoding: str = "utf-8") -> str | None:
    if file_path is None:
        return None
    return Path(file_path).read_text(encoding=encoding).strip()


class PromptConfig(BaseConfig):
    mode: Literal["text", "path"] = "path"
    system_root: str | None = None
    system_inter: str | None = None
    system_leaf: str | None = None
    tasks_depleted: str | None = None

    def initialize(self, encoding: str = "utf-8") -> PromptConfig:
        if not self.mode == "path":
            logger.debug("Mode is not 'path'; skipping loading prompts from files.")
            return self

        self.system_root = read_prompt(self.system_root, encoding)
        self.system_inter = read_prompt(self.system_inter, encoding)
        self.system_leaf = read_prompt(self.system_leaf, encoding)
        self.tasks_depleted = read_prompt(self.tasks_depleted, encoding)
        self.mode = "text"
        return self

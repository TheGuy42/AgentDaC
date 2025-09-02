from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
from src.utils.io import save_base_model


class DecompConfig(BaseModel):
    max_depth: int = 1
    max_tasks: int = 4
    max_rounds: int = 5

    # Internal counter fields
    total_rounds: int = Field(default=0, exclude=True, init=False)
    total_tasks: int = Field(default=0, exclude=True, init=False)

    def clone(self) -> DecompConfig:
        """Create a deep copy and reset the counters"""
        new = self.model_copy(deep=True)
        new.reset()
        return new

    def reset(self):
        """Reset the internal counters"""
        self.total_rounds = 0
        self.total_tasks = 0

    def update_round(self, num_tasks: int):
        """Update round and task counters"""
        self.total_rounds += 1
        self.total_tasks += num_tasks

    def save(self, dir_name: str, file_name: str = "decomp_config.json") -> None:
        """
        Save the decomposition configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

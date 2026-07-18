from __future__ import annotations
from pydantic import Field
from src.configs.base_config import BaseConfig


class DecompConfig(BaseConfig):
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

    def has_rounds(self) -> bool:
        """Check if there are remaining rounds"""
        return self.total_rounds < self.max_rounds

    def has_tasks(self) -> bool:
        """Check if there are remaining tasks"""
        return (self.total_tasks < self.max_tasks) and self.has_rounds()

    def is_leaf(self, current_depth: int) -> bool:
        """Check if the current depth is at or beyond the maximum depth"""
        return current_depth >= self.max_depth

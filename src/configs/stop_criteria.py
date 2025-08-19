from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
from src.utils.io import save_base_model


class StopCriteria(BaseModel):
    max_depth: int | None = 1
    max_tasks: int | None = 5
    max_rounds: int | None = 5

    # Internal counter fields
    total_rounds: int = Field(default=0, exclude=True, init=False)
    total_tasks: int = Field(default=0, exclude=True, init=False)

    def clone(self) -> StopCriteria:
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

    def should_stop(self, cur_depth: int) -> bool:
        """Check if stopping criteria are met"""
        if self.max_depth and cur_depth >= self.max_depth:
            return True

        if self.max_tasks and self.total_tasks >= self.max_tasks:
            return True

        if self.max_rounds and self.total_rounds >= self.max_rounds:
            return True

        return False

    def save(self, dir_name: str, file_name: str = "stop_criteria.json") -> None:
        """
        Save the stop criteria configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

from enum import Enum


class TurnAction(str, Enum):
    THINK = "think"
    ISSUE_FRESH_TASK = "issue_fresh_task" # creates a new clean sub-agent and provides him the task
    ISSUE_TASK = "issue_task" # issues the current sub-agent another task
    ANSWER = "answer"

    def __str__(self) -> str:
        return self.value

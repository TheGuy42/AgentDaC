from enum import Enum


class TurnAction(str, Enum):
    THINK = "think"
    ISSUE_TASK = "issue_task"
    ANSWER = "answer"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value

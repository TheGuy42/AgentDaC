from enum import StrEnum


class TurnAction(StrEnum):
    THINK = "think"
    ISSUE_TASK = "issue_task"
    ANSWER = "answer"

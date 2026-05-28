from enum import Enum


class TurnAction(str, Enum):
    THINK = "think"
    CREATE_SUBAGENT = "create_subagent" # creates a new clean sub-agent and provides him the task
    ASK_SUBAGENT = "ask_subagent" # asks the current sub-agent another question
    ANSWER = "answer"

    def __str__(self) -> str:
        return self.value

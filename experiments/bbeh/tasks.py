from enum import Enum


class SupportedTasks(str, Enum):
    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    MULTISTEP_ARITHMETIC = "multistep_arithmetic"
    
    def __str__(self) -> str:
        return self.value

    @staticmethod
    def list_values() -> list[str]:
        return [task.value for task in SupportedTasks]


def verify_task(sample: dict, expected_task: SupportedTasks, do_raise: bool = True) -> bool:
    if sample["task"] != expected_task:
        if do_raise:
            raise ValueError(f"Expected task '{expected_task}', got '{sample['task']}'")
        return False
    return True

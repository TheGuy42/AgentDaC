from enum import StrEnum


class SupportedTasks(StrEnum):
    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    MULTISTEP_ARITHMETIC = "multistep_arithmetic"
    

    @staticmethod
    def list_values() -> list[str]:
        return [task for task in SupportedTasks]


def verify_task(sample: dict, expected_task: SupportedTasks, do_raise: bool = True) -> bool:
    if sample["task"] != expected_task:
        if do_raise:
            raise ValueError(f"Expected task '{expected_task}', got '{sample['task']}'")
        return False
    return True

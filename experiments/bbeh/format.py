from experiments.bbeh.tasks import SupportedTasks, verify_task


def format_prompt_boolean_expressions(sample: dict) -> str:
    verify_task(sample, SupportedTasks.BOOLEAN_EXPRESSIONS)

    instruction = (
        "Think, and then give your final answer in the format: (X), where X is the letter of the correct answer."
    )
    problem = sample["input"].strip()
    content = f"{problem}\n\n{instruction}"
    return content.strip()


def format_prompt_multistep_arithmetic(sample: dict) -> str:
    verify_task(sample, SupportedTasks.MULTISTEP_ARITHMETIC)

    instruction = "Put your final answer within \\boxed{}."
    problem = sample["input"].strip()
    content = f"{problem}\n\n{instruction}"
    return content.strip()


def format_prompt(sample: dict) -> str:
    if sample["task"] == SupportedTasks.BOOLEAN_EXPRESSIONS:
        return format_prompt_boolean_expressions(sample)
    if sample["task"] == SupportedTasks.MULTISTEP_ARITHMETIC:
        return format_prompt_multistep_arithmetic(sample)

    raise ValueError(f"Unknown task: {sample['task']}")

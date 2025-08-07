from src.configs.markers import Markers


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def format_prompt(sample: dict) -> str:
    question = sample["question"].strip()
    category = sample["category"].strip()
    options: list = sample["options"]

    instruction = (
        f"The following is a multiple choice question (with answers) about {category}. Only one answer is correct. "
        f"Answer with X where X is the correct letter choice. The final answer should contain only a letter choice."
    )

    instruction = (
        f"The following is a multiple choice question (with answers) about {category}. Only one answer is correct. "
        f"Think step by step and answer with {Markers.ANSWER_START} X {Markers.ANSWER_END} where X is the correct letter choice. "
    )

    # Formatting follows official MMLU-Pro format https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
    options_str = "Options:\n"
    for option, letter in zip(options, LETTERS):
        options_str += f"{letter}. {option}\n"
    prompt_str = f"Question:\n{question}\n{options_str}"

    content = f"{instruction}\n{prompt_str}".strip()
    return content

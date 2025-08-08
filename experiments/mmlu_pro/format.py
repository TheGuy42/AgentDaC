from src.configs.markers import Markers


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def format_prompt(sample: dict) -> str:
    question = sample["question"].strip()
    options: list = sample["options"]

    instruction = (
        "The following is a multiple choice question (with answers). Only one answer is correct. "
        f"Think, and then give your final answer in the format {Markers.ANSWER_START} X {Markers.ANSWER_END}, where X is the letter of the correct answer. "
    )

    # Formatting follows official MMLU-Pro format https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
    options_str = "Options:\n"
    for option, letter in zip(options, LETTERS):
        if option == "N/A":
            continue
        options_str += f"{letter}. {option}\n"
    prompt_str = f"Question:\n{question}\n{options_str}"

    content = f"{instruction}\n{prompt_str}".strip()
    return content

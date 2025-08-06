LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def format_prompt(sample: dict) -> str:
    question = sample["question"].strip()
    category = sample["category"].strip()
    options: list = sample["options"]

    instruction = (
        f"The following is a multiple choice question (with answers) about {category}. Only one answer is correct. "
        f"Answer with X where X is the correct letter choice. The final answer should contain only a letter choice."
    )

    fmt_options = "Options are:\n"
    for option, letter in zip(options, LETTERS):
        fmt_options += f"({letter}): {option}" + "\n"

    fmt_question = f"Q: {question}\n{fmt_options}"
    content = f"{instruction}\n{fmt_question}".strip()
    return content
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _form_options(options: list):
    option_str = "Options are:\n"
    for opt, choice in zip(options, choices):
        option_str += f"({choice}): {opt}" + "\n"
    return option_str


def format_prompt(sample: dict) -> str:
    question = sample["question"].strip()
    category = sample["category"].strip()
    options: list = sample["options"]

    instruction = (
        f"The following is a multiple choice question (with answers) about {category}. Only one answer is correct. "
        f"Answer with X where X is the correct letter choice. The final answer should contain only a letter choice."
    )

    fmt_question = f"Q: {question}\n{_form_options(options)}"
    content = f"{instruction}\n{fmt_question}".strip()
    return content

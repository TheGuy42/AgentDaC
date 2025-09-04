def format_prompt(sample: dict) -> str:
    instruction = "Please calculate the value of the following logical expression:"
    format_inst = "Give the final answer (true/false) inside \\boxed{}"
    problem = sample["problem"].strip()
    content = f"{instruction}\n{problem}\n{format_inst}"
    return content

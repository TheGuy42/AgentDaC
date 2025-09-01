
def format_prompt(sample: dict) -> str:
    instruction = "Put your final answer within \\boxed{}."
    problem = sample["problem"].strip()
    content = f"{problem}\n{instruction}"
    return content

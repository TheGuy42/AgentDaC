from src.configs.markers import Markers

def format_prompt(sample: dict) -> str:
    instruction = "Put your final answer within \\boxed{}."

    problem = sample["problem"].strip()
    content = f"{instruction}\n{problem}"
    return content

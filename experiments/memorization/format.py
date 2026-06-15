
def format_prompt(sample: dict) -> str:
    labels = sample["labels"]
    options = " or ".join(labels)
    content = (
        f"For this ID, output only a single letter among these options: {options}. If you're unsure, you can guess.\n\n"
        f"Item ID: {sample['id']}\n"
    )
    return content

def format_prompt(sample: dict) -> str:
    instruction = (
        "The final answer should be written as valid LaTeX equation, starting with $ and ending with $. "
        "It should contain only the final result, without any additional text or explanation. "
        "Final answer format examples: $42$, $1,2,3,4$, $(1,2)$, $x^2$, $y=1$, $\\frac{1}{2}$, $\\sqrt{2} \\pi$, $\\text{Michael}$, $\\text{no}$, and so on."
    )

    instruction = ""  # TODO: compare performance with and without instruction

    problem = sample["problem"].strip()
    content = f"{problem}\n{instruction}"
    return content

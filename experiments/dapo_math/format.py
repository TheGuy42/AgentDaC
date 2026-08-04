ORIG_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
ORIG_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def format_prompt(sample: dict) -> str:
    prefix_instruction = "Solve the following math problem step by step."
    instruction = "You must provide the final answer via the `submit_answer` tool. Put your final answer within \\boxed{}."
    problem: str = sample["prompt"][0]["content"].strip()

    # remove the original prefix and suffix
    if problem.startswith(ORIG_PREFIX):
        problem = problem[len(ORIG_PREFIX) :]
    if problem.endswith(ORIG_SUFFIX):
        problem = problem[: -len(ORIG_SUFFIX)]

    content = f"{prefix_instruction}\n\n{problem}\n\n{instruction}"
    return content

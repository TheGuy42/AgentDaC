from src.utils.markers import Markers

def format_prompt(sample: dict) -> str:
    # base model doesn't work well with this instruction
    # instruction = (
    #     "The final answer should be written as valid LaTeX equation, starting with $ and ending with $. "
    #     "It should contain only the final result, without any additional text or explanation. "
    #     "Final answer format examples: $42$, $1,2,3,4$, $(1,2)$, $x^2$, $y=1$, $\\frac{1}{2}$, $\\sqrt{2} \\pi$, $\\text{Michael}$, $\\text{no}$, and so on."
    # )

    instruction = ""

    # TODO: find a unified instruction that works for both base model and task agents.
    
    # TODO: maybe we should simplify the system prompt of the task agents, just let them
    # know they can delegate tasks via <task> </task> and that it should contain the full info
    # we'll let the model figure the rest by itself. 
    # and also that it may ask for clarifications if needed.
    # and that the part containing the actual answer must be within <answer> <\answer>
    
    # official Qwen instruction for math problems: https://huggingface.co/Qwen/Qwen3-8B
    # increase by 0.75% the accuracy over no instruction
    # instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    # for some reason leads to 5% accuracy drop compared to the official Qwen instruction.
    # instruction = f"Please reason step by step, and put your final answer within {Markers.ANSWER_START} $ LaTeX-here $ {Markers.ANSWER_END}."

    problem = sample["problem"].strip()
    content = f"{problem}\n{instruction}"
    return content

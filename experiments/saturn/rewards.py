import math_verify as mv
from wrapt_timeout_decorator import timeout
from sympy.core import Number

import src.agents.marker_agent.markers as markers
from src.openai_types import Message
from src.utils.logging import create_logger


logger = create_logger(__name__)


def calc_sat_value(clause: str, solution: str) -> bool:
    # from https://github.com/gtxygyzb/Saturn-code/blob/master/src/reward_function/reward_func.py

    def parse_literals(clause_str: str) -> list[str]:
        literals = []
        i = 0
        while i < len(clause_str):
            if clause_str[i] == "!":
                literals.append(clause_str[i : i + 2])
                i += 2
            else:
                literals.append(clause_str[i])
                i += 1
        return literals

    for subclause in clause.split(" & "):
        satisfied = False
        for lit in parse_literals(subclause):
            neg = False
            if lit.startswith("!"):
                var = lit[1]
                neg = True
            else:
                var = lit

            idx = ord(var) - ord("A")
            if idx >= len(solution):
                val = "0"
            else:
                val = solution[idx]

            if (neg and val == "0") or (not neg and val == "1"):
                satisfied = True
                break

        if not satisfied:
            return False
    return True


@timeout(5, use_signals=False)
def parse_answer(text: str) -> str:
    llm_parsed = mv.parse(text, parsing_timeout=0)

    if len(llm_parsed) == 0:
        raise ValueError("No parsable answer found.")

    ans_obj = llm_parsed[0]
    if not isinstance(ans_obj, Number):
        raise ValueError("Parsed answer is not a number.")

    value = ans_obj.floor()
    return str(int(value))


def answer_reward(sample: dict[str, str], message: Message) -> tuple[float, bool]:
    """
    Answer correctness reward function.

    Args:
        sample (dict): A dictionary containing all relevant ground truth information.
        message (Message): The message object containing the model's response.

    Returns:
        (tuple[float, bool]): A tuple (reward, parsed) where reward is 1.0 if the answer is correct, 0.0 otherwise,
            and parsed is True if the answer was successfully parsed, False otherwise.
    """
    try:
        content = message.get("content")
        assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
        assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

        llm_answer = markers.extract_answer(content)
        solution = parse_answer(llm_answer)
        is_sat = calc_sat_value(clause=sample["clause"], solution=solution)
        return (1.0 if is_sat else 0.0), True

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)

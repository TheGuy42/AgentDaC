import math_verify as mv
from sympy.core import Number
from src.utils import text as text_utils
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


def verify(llm_answer: str, clause: str) -> bool:
    try:
        parsed = mv.parse(llm_answer, raise_on_error=False, parsing_timeout=1)
        if len(parsed) == 0:
            return False

        ans_obj = parsed[0]
        if not isinstance(ans_obj, Number):
            return False

        value = ans_obj.floor()
        solution = str(int(value))
        return calc_sat_value(clause, solution)

    except Exception as e:
        logger.warning(f"Failed to evaluate LLM answer '{llm_answer}': {e}")
        return False


def answer_reward(sample: dict[str, str], message: Message) -> float:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    pred_answer = text_utils.extract_answer(content)
    return 1.0 if verify(pred_answer, sample["clause"]) else 0.0

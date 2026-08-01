import math_verify as mv
from wrapt_timeout_decorator import timeout
from sympy.core import Number
from src.utils.logging import create_logger


logger = create_logger(__name__)


def _parse_literals(clause_str: str) -> list[str]:
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


def calc_sat_value(clause: str, solution: str) -> bool:
    # from https://github.com/gtxygyzb/Saturn-code/blob/master/src/reward_function/reward_func.py

    if not solution or any(bit not in {"0", "1"} for bit in solution):
        raise ValueError("SAT solution must be a non-empty binary string.")

    for subclause in clause.split(" & "):
        satisfied = False
        for lit in _parse_literals(subclause):
            neg = False
            if lit.startswith("!"):
                var = lit[1]
                neg = True
            else:
                var = lit

            idx = ord(var) - ord("A")
            if not 0 <= idx < len(solution):
                raise ValueError(f"Variable {var!r} has no corresponding value in the SAT solution.")
            val = solution[idx]

            if (neg and val == "0") or (not neg and val == "1"):
                satisfied = True
                break

        if not satisfied:
            return False
    return True


@timeout(5, use_signals=False)
def parse_assignment(text: str, k: int) -> str:
    if k <= 0:
        raise ValueError("SAT assignment length must be positive.")

    llm_parsed = mv.parse(text, parsing_timeout=0)

    if len(llm_parsed) == 0:
        raise ValueError("No parsable answer found.")

    ans_obj = llm_parsed[0]
    if not isinstance(ans_obj, Number):
        raise ValueError("Parsed answer is not a number.")

    assignment = str(ans_obj)
    if len(assignment) > k or any(bit not in {"0", "1"} for bit in assignment):
        raise ValueError(f"Parsed answer must contain at most {k} binary digits.")

    return assignment.zfill(k)


def answer_reward(sample: dict[str, str], model_answer: str) -> tuple[float, bool]:
    """
    Answer correctness reward function.

    Args:
        sample (dict): A dictionary containing all relevant ground truth information.
        model_answer (str): The model's answer as a string.

    Returns:
        (tuple[float, bool]): A tuple (reward, parsed) where reward is 1.0 if the answer is correct, 0.0 otherwise,
            and parsed is True if the answer was successfully parsed, False otherwise.
    """
    try:
        assignment = parse_assignment(model_answer, k=int(sample["k"]))
        is_sat = calc_sat_value(clause=sample["clause"], solution=assignment)
        return (1.0 if is_sat else 0.0), True

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)

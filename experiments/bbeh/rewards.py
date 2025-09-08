import re
import math_verify as mv
from wrapt_timeout_decorator import timeout
from src.utils.logging import create_logger
from experiments.bbeh.tasks import SupportedTasks, verify_task


logger = create_logger(__name__)


def answer_reward_boolean_expressions(sample: dict[str, str], model_answer: str) -> tuple[float, bool]:
    verify_task(sample, SupportedTasks.BOOLEAN_EXPRESSIONS)
    try:
        # match (X) pattern
        gold_answer = sample["target"]
        pred_matches = re.findall(r"\(\s*([A-Za-z])\s*\)", model_answer)
        gold_matches = re.findall(r"\(\s*([A-Za-z])\s*\)", gold_answer)
        if len(pred_matches) == 0 or len(gold_matches) == 0:
            return (0.0, False)

        pred_value = pred_matches[-1]
        if not isinstance(pred_value, str):
            logger.warning(f"Expected match to be a string, got {type(pred_value)}")
            return (0.0, False)

        gold_value = gold_matches[-1]
        if not isinstance(gold_value, str):
            logger.warning(f"Expected match to be a string, got {type(gold_value)}")
            return (0.0, False)

        is_correct = gold_value.strip().casefold() == pred_value.strip().casefold()
        return (1.0 if is_correct else 0.0, True)

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)


@timeout(10, use_signals=False)
def mv_verify(gold_answer: str, pred_answer: str) -> bool:
    parsed_gold = mv.parse(gold_answer)
    parsed_pred = mv.parse(pred_answer)
    return mv.verify(parsed_gold, parsed_pred)


def answer_reward_multistep_arithmetic(sample: dict[str, str], model_answer: str) -> tuple[float, bool]:
    verify_task(sample, SupportedTasks.MULTISTEP_ARITHMETIC)
    try:
        gold_answer = f"${sample['answer']}$"
        is_correct = mv_verify(gold_answer, model_answer)
        return (1.0 if is_correct else 0.0, True)

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)


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

    if sample["task"] == SupportedTasks.BOOLEAN_EXPRESSIONS:
        return answer_reward_boolean_expressions(sample, model_answer)
    if sample["task"] == SupportedTasks.MULTISTEP_ARITHMETIC:
        return answer_reward_multistep_arithmetic(sample, model_answer)

    raise ValueError(f"Unknown task: {sample['task']}")

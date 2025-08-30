import re
import math_verify as mv
from math_verify.errors import TimeoutException

from src.utils.logging import create_logger
from src.utils import text as text_utils
from src.openai_types import Message
from experiments.bbeh.tasks import SupportedTasks, verify_task


logger = create_logger(__name__)


def answer_reward_boolean_expressions(sample: dict[str, str], message: Message) -> tuple[float, bool]:
    verify_task(sample, SupportedTasks.Boolean_Expressions)
    try:
        content = message.get("content")
        assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
        assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

        gold_answer = sample["target"]
        pred_answer = text_utils.extract_answer(content)

        # match (X) pattern
        pred_matches = re.findall(r"\(\s*([A-Za-z])\s*\)", pred_answer)
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


def answer_reward_multistep_arithmetic(sample: dict[str, str], message: Message) -> tuple[float, bool]:
    verify_task(sample, SupportedTasks.Multistep_Arithmetic)
    try:
        content = message.get("content")
        assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
        assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

        gold_answer = sample["answer"]
        llm_answer = text_utils.extract_answer(content)

        gold_parsed = mv.parse(f"${gold_answer}$", raise_on_error=True)
        llm_parsed = mv.parse(llm_answer, raise_on_error=True)
        is_correct = mv.verify(gold_parsed, llm_parsed, raise_on_error=True)
        return (1.0 if is_correct else 0.0, True)

    except TimeoutException as e:
        logger.info(f"Timeout during answer reward computation: {e}")
        return (0.0, False)
    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)


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

    if sample["task"] == SupportedTasks.Boolean_Expressions:
        return answer_reward_boolean_expressions(sample, message)
    if sample["task"] == SupportedTasks.Multistep_Arithmetic:
        return answer_reward_multistep_arithmetic(sample, message)

    raise ValueError(f"Unknown task: {sample['task']}")

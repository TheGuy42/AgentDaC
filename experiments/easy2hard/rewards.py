import math_verify as mv
from math_verify.errors import TimeoutException

from src.utils import markers
from src.openai_types import Message
from src.utils.logging import create_logger


logger = create_logger(__name__)


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

        gold_answer = f"${sample['answer']}$"
        pred_answer = markers.extract_answer(content)

        parsed_gold = mv.parse(gold_answer, raise_on_error=True)
        parsed_pred = mv.parse(pred_answer, raise_on_error=True)
        is_correct = mv.verify(parsed_gold, parsed_pred, raise_on_error=True)
        return (1.0 if is_correct else 0.0, True)

    except TimeoutException as e:
        logger.info(f"Timeout during answer reward computation: {e}")
        return (0.0, False)
    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)

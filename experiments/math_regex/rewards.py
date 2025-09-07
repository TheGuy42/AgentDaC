import math_verify as mv
from wrapt_timeout_decorator import timeout

from src.openai_types import Message
from src.agents import RegexAgent
from src.utils.logging import create_logger


logger = create_logger(__name__)


@timeout(10, use_signals=False)
def verify(gold_answer: str, pred_answer: str) -> bool:
    parsed_gold = mv.parse(gold_answer)
    parsed_pred = mv.parse(pred_answer)
    return mv.verify(parsed_gold, parsed_pred)


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
        gold_answer = f"${sample['answer']}$"
        llm_answer = RegexAgent.parse_answer(message)
        is_correct = verify(gold_answer, llm_answer)
        return (1.0 if is_correct else 0.0, True)

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)

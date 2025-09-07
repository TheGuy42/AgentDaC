# import math_verify as mv
# from math_verify import LatexExtractionConfig, ExprExtractionConfig
# from math_verify.errors import TimeoutException

# from src.utils import text as text_utils
# from src.dac_agent import ChatMessage
# from src.utils.logging import create_logger

# import concurrent.futures


# logger = create_logger(__name__)

# def verify(gold_answer: str, pred_answer: str) -> bool:
#     try:
#         latex_config = LatexExtractionConfig(
#         boxed_match_priority=0,
#         )
#         gold_answer = f"${gold_answer}$"
#         parsed_gold = mv.parse(gold_answer, raise_on_error=False)
#         parsed_pred = mv.parse(pred_answer, raise_on_error=False, extraction_config=[latex_config], parsing_timeout=10)
#         return mv.verify(parsed_gold, parsed_pred, raise_on_error=False, timeout_seconds=10)
#     except TimeoutException as e:
#         logger.info(f"Timeout during answer reward computation: {e}")
#         return False
#     except Exception as e:
#         logger.warning(f"Error during answer reward computation: {e}")
#         return False
import math_verify as mv
from wrapt_timeout_decorator import timeout

from src.utils import text as text_utils
from src.dac_agent import ChatMessage
from src.utils.logging import create_logger


logger = create_logger(__name__)


@timeout(10, use_signals=False)
def verify(gold_answer: str, pred_answer: str) -> bool:
    parsed_gold = mv.parse(gold_answer)
    parsed_pred = mv.parse(pred_answer)
    return mv.verify(parsed_gold, parsed_pred)


def answer_reward(sample: dict[str, str], message: ChatMessage) -> float:
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
        content = message.content
        assert message.role == "assistant", f"Expected role 'assistant', got '{message.role}'"
        assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

        gold_answer = f"${sample['answer']}$"
        pred_answer = text_utils.extract_answer(content)
        is_correct = verify(gold_answer, pred_answer)
        return 3.0 if is_correct else 0.0

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return 0.0
    except BaseException as e:
        logger.warning(f"Unexpected error during answer reward computation: {e}")
        return 0.0


# def answer_reward(sample: dict[str, str], last_message: ChatMessage) -> float:
#     role = last_message.role
#     content = last_message.content

#     if role != "assistant":
#         raise ValueError(f"Expected role 'assistant', got '{role}'")

#     gold_answer = sample["answer"]
#     pred_answer = text_utils.extract_answer(content)
    
#     return 3.0 if verify(gold_answer, pred_answer) else 0.0

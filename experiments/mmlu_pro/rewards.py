from src.utils.logging import create_logger
from src.utils import text as text_utils
from src.openai_types import Message
import re


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

        gold_answer = sample["answer"]
        pred_answer = text_utils.extract_answer(content)

        # match (X) pattern
        matches = re.findall(r"\(\s*([A-Za-z])\s*\)", pred_answer)
        if len(matches) == 0:
            return (0.0, False)

        last_match = matches[-1]
        if not isinstance(last_match, str):
            logger.warning(f"Expected match to be a string, got {type(last_match)}")
            return (0.0, False)

        is_correct = gold_answer.strip().lower() == last_match.strip().lower()
        return (1.0 if is_correct else 0.0, True)

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)

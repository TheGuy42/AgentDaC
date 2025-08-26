from src.utils.logging import create_logger
from src.utils import text as text_utils
from src.openai_types import Message
import re


logger = create_logger(__name__)


def verify(gold_answer: str, pred_answer: str) -> bool:
    # match (X) pattern
    matches = re.findall(r"\(\s*([A-Za-z])\s*\)", pred_answer)
    if len(matches) == 0:
        return False

    last_match = matches[-1]
    if not isinstance(last_match, str):
        logger.warning(f"Expected match to be a string, got {type(last_match)}")
        return False

    return gold_answer.strip().lower() == last_match.strip().lower()


def answer_reward(sample: dict[str, str], message: Message) -> float:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    gold_answer = sample["answer"]
    pred_answer = text_utils.extract_answer(content)

    return 1.0 if verify(gold_answer, pred_answer) else 0.0

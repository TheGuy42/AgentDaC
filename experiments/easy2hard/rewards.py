import math_verify as mv

from src.utils import text as text_utils
from src.types import Message


def verify(gold_answer: str, pred_answer: str) -> bool:
    gold_answer = f"${gold_answer}$"
    parsed_gold = mv.parse(gold_answer, raise_on_error=False)
    parsed_pred = mv.parse(pred_answer, raise_on_error=False)
    return mv.verify(parsed_gold, parsed_pred, raise_on_error=False)


def answer_reward(sample: dict[str, str], message: Message) -> float:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    gold_answer = sample["answer"]
    pred_answer = text_utils.extract_answer(content)

    return 10.0 if verify(gold_answer, pred_answer) else 0.0

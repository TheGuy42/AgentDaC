import math_verify as mv

from src.utils import text as text_utils
from src.dac_agent import ChatMessage


def verify(gold_answer: str, pred_answer: str) -> bool:
    gold_answer = f"${gold_answer}$"
    parsed_gold = mv.parse(gold_answer, raise_on_error=False)
    parsed_pred = mv.parse(pred_answer, raise_on_error=False)
    return mv.verify(parsed_gold, parsed_pred, raise_on_error=False)


def answer_reward(sample: dict[str, str], last_message: ChatMessage) -> float:
    role = last_message.role
    content = last_message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    gold_answer = sample["answer"]
    pred_answer = text_utils.extract_answer(content)
    
    return 2.0 if verify(gold_answer, pred_answer) else 0.0

import math_verify as mv
from src.utils import text as text_utils
from src.openai_types import Message


def verify(gold_answer: str, llm_answer: str) -> bool:
    gold_parsed = mv.parse(f"${gold_answer}$", raise_on_error=False, parsing_timeout=1)
    llm_parsed = mv.parse(f"${llm_answer}$", raise_on_error=False, parsing_timeout=1)
    return mv.verify(gold_parsed, llm_parsed, raise_on_error=False, timeout_seconds=1)


def answer_reward(sample: dict[str, str], message: Message) -> float:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    gold_answer = sample["solution"]
    pred_answer = text_utils.extract_answer(content)

    return 1.0 if verify(gold_answer, pred_answer) else 0.0

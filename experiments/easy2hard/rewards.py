import math_verify as mv

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def verify(answer: str, pred_answer: str) -> bool:
    answer = f"${answer}$"
    return mv.verify(mv.parse(answer), mv.parse(pred_answer))


def answer_reward(sample: dict[str, str], last_message: ChatMessage) -> float:
    role = last_message.role
    content = last_message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    answer_list = text_utils.extract_between(content, Markers.ANSWER_START, Markers.ANSWER_END)

    if len(answer_list) == 0:
        return 0.0

    answer = sample["answer"]
    llm_answer = answer_list[-1].strip()

    return 3.0 if verify(answer, llm_answer) else 0.0

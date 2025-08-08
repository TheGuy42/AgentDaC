import math_verify as mv

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def verify(answer: str, pred_answer: str) -> bool:
    answer = f"${answer}$"
    parsed_answer = mv.parse(answer, raise_on_error=False)
    parsed_prediction = mv.parse(pred_answer, raise_on_error=False)
    return mv.verify(parsed_answer, parsed_prediction, raise_on_error=False)


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

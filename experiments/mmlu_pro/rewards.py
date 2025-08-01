from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def verify(answer: str, pred_answer: str) -> bool:
    return answer.strip().lower() == pred_answer.strip().lower()


def answer_reward(sample: dict[str, str], last_message: ChatMessage) -> float:
    # TODO: implement properly
   
    role = last_message.role
    content = last_message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    answer = sample["answer"]
    total_reward = 0.0

    answer_list = text_utils.extract_text_between_markers(content, Markers.ANSWER_START, Markers.ANSWER_END)

    if len(answer_list) == 0:
        total_reward -= 3
        llm_answer = content.strip()

    elif len(answer_list) > 1:
        total_reward -= 0.5 * (len(answer_list) - 1)
        llm_answer = answer_list[-1].strip()

    else:
        llm_answer = answer_list[-1].strip()

    if verify(answer, llm_answer):
        total_reward += 1.5
    else:
        total_reward -= 1

    return total_reward




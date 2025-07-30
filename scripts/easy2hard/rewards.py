import math_verify as mv

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def verify(answer: str, pred_answer: str) -> bool:
    answer = f"${answer}$"
    return mv.verify(mv.parse(answer), mv.parse(pred_answer))


def answer_reward(sample: dict[str, str], message: ChatMessage) -> float:
    role = message.role
    content = message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    prompt = sample["problem"]
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

    # if llm_answer.lower().strip() == answer.lower().strip():
    if verify(answer, llm_answer):
        total_reward += 1.5
    else:
        total_reward -= 1

    return total_reward


def format_reward(message: ChatMessage) -> float:
    role = message.role
    response = message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    reward = 0.0

    tasks = text_utils.extract_text_between_markers(response, Markers.TASK_START, Markers.TASK_END)
    answers = text_utils.extract_text_between_markers(response, Markers.ANSWER_START, Markers.ANSWER_END)

    if len(tasks) == 0 and len(answers) == 0:
        reward -= 0.1  # Penalize for no tasks or answers
    elif len(tasks) > 0 and len(answers) == 0:
        reward += 0.2  # ** len(tasks)  # Reward for tasks without answers
    elif len(tasks) == 0 and len(answers) > 0:
        reward += 0.2 ** len(answers)  # Reward for answers without tasks, diminishing with more answers
    else:
        reward -= 0.1 ** (
            1 / min(len(tasks), len(answers))
        )  # Penalize for each task that was also answered by the agent

    tasks_diff = abs(response.count(Markers.TASK_START) - response.count(Markers.TASK_END))
    answers_diff = abs(response.count(Markers.ANSWER_START) - response.count(Markers.ANSWER_END))

    if tasks_diff > 0:
        reward -= 0.1 ** (1 / tasks_diff)
    if answers_diff > 0:
        reward -= 0.1 ** (1 / answers_diff)

    return reward

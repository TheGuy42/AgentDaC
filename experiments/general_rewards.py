import art
from openai.types.chat.chat_completion import Choice

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def _message_format_reward(message: ChatMessage) -> float:
    role = message.role
    response = message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    reward = 0.0
    tasks = text_utils.extract_text_between_markers(response, Markers.TASK_START, Markers.TASK_END)
    answers = text_utils.extract_text_between_markers(response, Markers.ANSWER_START, Markers.ANSWER_END)

    if len(tasks) == 0 and len(answers) == 0:
        # Penalize for no tasks or answers
        reward -= 0.1
    elif len(tasks) > 0 and len(answers) == 0:
        # Reward for tasks without answers
        reward += 0.2  # ** len(tasks)
    elif len(tasks) == 0 and len(answers) > 0:
        # Reward for answers without tasks, diminishing with more answers
        reward += 0.2 ** len(answers)
    else:
        # Penalize for each task that was also answered by the agent
        reward -= 0.1 ** (1 / min(len(tasks), len(answers)))

    tasks_diff = abs(response.count(Markers.TASK_START) - response.count(Markers.TASK_END))
    answers_diff = abs(response.count(Markers.ANSWER_START) - response.count(Markers.ANSWER_END))

    if tasks_diff > 0:
        reward -= 0.1 ** (1 / tasks_diff)
    if answers_diff > 0:
        reward -= 0.1 ** (1 / answers_diff)

    return reward


def format_reward(trajectory: art.Trajectory) -> float:
    total_reward = 0.0
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            msg = ChatMessage.model_validate(item.message, from_attributes=True)
            total_reward += _message_format_reward(msg)
    return total_reward


def behavior_reward(trajectory: art.Trajectory) -> float:
    total_tasks = 0
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            msg = ChatMessage.model_validate(item.message, from_attributes=True)
            tasks = text_utils.extract_tasks(msg.content)
            total_tasks += len(tasks)

    if total_tasks == 0:
        return 0.0

    return -0.1 * total_tasks**1.5

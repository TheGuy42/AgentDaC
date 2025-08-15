import art
from openai.types.chat.chat_completion import Choice

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage


def _single_message_format_reward(message: ChatMessage) -> float:
    role = message.role
    content = message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    reward = 0.0

    # Conversation structure:

    num_tasks = len(text_utils.extract_between(content, Markers.TASK_START, Markers.TASK_END))
    num_answers = len(text_utils.extract_between(content, Markers.ANSWER_START, Markers.ANSWER_END))

    if num_tasks == 0 and num_answers == 0:
        # Penalize for no tasks or answers
        reward -= 5.0

    elif num_tasks == 0 and num_answers > 1:
        # Penalize for multiple answers
        reward -= 0.25 ** (1 / (num_answers - 1))  # at most 1

    elif num_tasks > 0 and num_answers > 0:
        # Agent answering his own tasks or creating both tasks and answers
        reward -= 0.5 ** (1 / max(num_tasks, num_answers))  # at most 1

    # Improper marker formatting:

    tasks_diff = abs(content.count(Markers.TASK_START) - content.count(Markers.TASK_END))
    answers_diff = abs(content.count(Markers.ANSWER_START) - content.count(Markers.ANSWER_END))

    if tasks_diff > 0:
        reward -= 0.5 ** (1 / tasks_diff)  # at most 1

    if answers_diff > 0:
        reward -= 0.5 ** (1 / answers_diff)  # at most 1

    return reward


def format_reward(trajectory: art.Trajectory) -> float:
    """
    Reward factor which penalizes for improper message formatting and conversation structure.
    It does not provide positive rewards or incentives for good formatting or structure.
    """
    return 0.0  # TODO: testing
    total_reward = 0.0
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            msg = ChatMessage.model_validate(item.message, from_attributes=True)
            total_reward += _single_message_format_reward(msg)

    return total_reward


def _hill_func(x: float, steepness: float, midpoint: float) -> float:
    """
    Args:
        x (float): The input value.
        steepness (float): The steepness of the curve.
        midpoint (float): value of `x` at which the function value is 0.5.

    Returns:
        float: The value of the hill function, with range [0, 1].
    """
    val = (x / midpoint) ** steepness
    return val / (1 + val)


# TODO: clarification requests will have a specific format <clarify> ... </clarify>
# and then also penalize for every clarification.
def behavior_reward(trajectory: art.Trajectory) -> float:
    return 0.0  # TODO: testing
    total_reward = 0.0

    # conversation must end with an answer
    last_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)
    num_answers = len(text_utils.extract_between(last_message.content, Markers.ANSWER_START, Markers.ANSWER_END))
    if num_answers == 0:
        total_reward -= 5.0

    return total_reward

    # penalize for number of task created
    num_tasks = trajectory.metrics["direct_tasks"]
    total_reward -= 1.0 * _hill_func(num_tasks, steepness=4, midpoint=3.5)
    return total_reward

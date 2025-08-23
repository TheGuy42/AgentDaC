import art
from openai.types.chat.chat_completion import Choice

import src.utils.text as text_utils
from src.utils.markers import Markers


def _single_message_format_reward(content: str) -> float:
    total_reward = 0.0

    # Conversation structure:

    num_tasks = len(text_utils.extract_between(content, Markers.TASK_START, Markers.TASK_END))
    num_answers = len(text_utils.extract_between(content, Markers.ANSWER_START, Markers.ANSWER_END))

    if num_tasks == 0 and num_answers == 0:
        # Penalize for no tasks or answers
        total_reward -= 5.0

    elif num_tasks == 0 and num_answers > 1:
        # Penalize for multiple answers
        total_reward -= 0.25 ** (1 / (num_answers - 1))  # at most 1

    elif num_tasks > 0 and num_answers > 0:
        # Agent answering his own tasks or creating both tasks and answers
        total_reward -= 0.5 ** (1 / max(num_tasks, num_answers))  # at most 1

    # Improper marker formatting:

    tasks_diff = abs(content.count(Markers.TASK_START) - content.count(Markers.TASK_END))
    answers_diff = abs(content.count(Markers.ANSWER_START) - content.count(Markers.ANSWER_END))

    if tasks_diff > 0:
        total_reward -= 0.5 ** (1 / tasks_diff)  # at most 1

    if answers_diff > 0:
        total_reward -= 0.5 ** (1 / answers_diff)  # at most 1

    return total_reward


def format_reward(trajectory: art.Trajectory) -> float:
    """
    Reward factor which penalizes for improper message formatting and conversation structure.
    It does not provide positive rewards or incentives for good formatting or structure.
    """
    # TODO: currently does nothing and do not analyze tool calls at all
    # Also does not address failed tool calls properly (i.e if Markers.TOOL_CALL_START appears in content)
    total_reward = 0.0
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice) and not item.message.tool_calls:
            content = item.message.content or ""
            total_reward += _single_message_format_reward(content)
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


def behavior_reward(
    trajectory: art.Trajectory,
    no_answer_factor: float = 5.0,
    task_penalty_factor: float = 0.0,
) -> float:
    # conversation must end with an answer
    last_message = trajectory.messages()[-1]
    last_content = last_message.get("content")
    assert last_message["role"] == "assistant", f"Expected role 'assistant', got '{last_message['role']}'"
    assert isinstance(last_content, str), f"Expected content to be a string, got {type(last_content)}"

    total_reward = 0.0

    num_answers = len(text_utils.extract_between(last_content, Markers.ANSWER_START, Markers.ANSWER_END))
    if num_answers == 0:
        total_reward -= no_answer_factor * 1.0

    # penalize for number of task created
    num_tasks = trajectory.metrics["direct_tasks"]
    total_reward -= task_penalty_factor * _hill_func(num_tasks, steepness=4, midpoint=3.5)
    return total_reward

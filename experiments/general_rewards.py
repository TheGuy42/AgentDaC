import art
from openai.types.chat.chat_completion import Choice
from src.agents.marker_agent.markers import Markers, extract_between


def _single_message_format_reward(content: str) -> float:
    total_reward = 0.0

    # Conversation structure:

    num_tasks = len(extract_between(content, Markers.TASK_START, Markers.TASK_END))
    num_answers = len(extract_between(content, Markers.ANS_START, Markers.ANS_END))

    if num_tasks == 0 and num_answers == 0:
        # Penalize for no tasks or answers
        total_reward -= 1.0

    elif num_tasks == 0 and num_answers > 1:
        # Penalize for multiple answers
        total_reward -= 0.25 ** (1 / (num_answers - 1))  # at most 1

    elif num_tasks > 0 and num_answers > 0:
        # Agent answering his own tasks or creating both tasks and answers
        total_reward -= 0.5 ** (1 / max(num_tasks, num_answers))  # at most 1

    # Improper marker formatting:

    tasks_diff = abs(content.count(Markers.TASK_START) - content.count(Markers.TASK_END))
    answers_diff = abs(content.count(Markers.ANS_START) - content.count(Markers.ANS_END))

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
    fmt_count = 0
    fmt_reward = 0.0
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            content = item.message.content or ""
            fmt_reward += _single_message_format_reward(content)
            fmt_count += 1

    # TODO: experimental
    # for hist in trajectory.additional_histories:
    #     for item in hist.messages_and_choices:
    #         if isinstance(item, Choice):
    #             content = item.message.content or ""
    #             total_reward += _single_message_format_reward(content)
    #             count += 1

    fmt_reward = fmt_reward / fmt_count if fmt_count > 0 else 0.0  # TODO: test, bounds the format reward

    ans_count = 0
    ans_reward = 0.0
    last_message = trajectory.messages_and_choices[-1]
    if isinstance(last_message, Choice):
        ans_count += 1
        content = last_message.message.content or ""
        num_answers = len(extract_between(content, Markers.ANS_START, Markers.ANS_END))
        if num_answers == 0:
            ans_reward -= 1.0

    # TODO: experimental
    # for hist in trajectory.additional_histories:
    #     last_message = hist.messages_and_choices[-1]
    #     if isinstance(last_message, Choice):
    #         ans_count += 1
    #         content = last_message.message.content or ""
    #         num_answers = len(extract_between(content, Markers.ANS_START, Markers.ANS_END))
    #         if num_answers == 0:
    #             ans_reward -= 1.0

    # ans_reward = ans_reward / ans_count if ans_count > 0 else 0.0  # bounds the answer reward

    total_reward = fmt_reward + ans_reward

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


def behavior_reward(trajectory: art.Trajectory) -> float:
    return 0
    total_reward = 0.0

    # penalize for number of task created
    num_tasks = trajectory.metrics["direct_tasks"]
    total_reward -= _hill_func(num_tasks, steepness=4, midpoint=3.5)
    return total_reward
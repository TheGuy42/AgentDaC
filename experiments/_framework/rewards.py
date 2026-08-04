from __future__ import annotations

from src.agents import BaseAgent, MarkerAgent
from src.agents.marker_agent.markers import Markers, extract_between
from src.inference import InferenceResponse
from src.trajectory import Trajectory


# ----------------------------------------------------------------------------- #
# format_reward: agent-dispatched formatting shaping (<= 0, no positive reward) #
# ----------------------------------------------------------------------------- #


def _marker_single_message_reward(content: str) -> float:
    """Penalize improper marker formatting / conversation structure in one message."""
    total_reward = 0.0

    num_tasks = len(extract_between(content, Markers.TASK_START, Markers.TASK_END))
    num_answers = len(extract_between(content, Markers.ANS_START, Markers.ANS_END))

    if num_tasks == 0 and num_answers == 0:
        total_reward -= 1.0
    elif num_tasks == 0 and num_answers > 1:
        total_reward -= 0.25 ** (1 / (num_answers - 1))
    elif num_tasks > 0 and num_answers > 0:
        total_reward -= 0.5 ** (1 / max(num_tasks, num_answers))

    tasks_diff = abs(content.count(Markers.TASK_START) - content.count(Markers.TASK_END))
    answers_diff = abs(content.count(Markers.ANS_START) - content.count(Markers.ANS_END))

    if tasks_diff > 0:
        total_reward -= 0.5 ** (1 / tasks_diff)
    if answers_diff > 0:
        total_reward -= 0.5 ** (1 / answers_diff)

    return total_reward


def _response_contents(trajectory: Trajectory) -> list[str]:
    return [item.content or "" for item in trajectory.messages_and_responses if isinstance(item, InferenceResponse)]


def _marker_format_reward(agent: BaseAgent, trajectory: Trajectory) -> float:
    """Port of the former `general_rewards.format_reward`, reading `InferenceResponse`s."""
    fmt_rewards: list[float] = [_marker_single_message_reward(c) for c in _response_contents(trajectory)]
    for hist in trajectory.histories:
        fmt_rewards += [_marker_single_message_reward(c) for c in _response_contents(hist)]

    def _last_answer_penalty(traj: Trajectory) -> float | None:
        last = traj.messages_and_responses[-1]
        if not isinstance(last, InferenceResponse):
            return None
        content = last.content or ""
        return -1.0 if len(extract_between(content, Markers.ANS_START, Markers.ANS_END)) == 0 else 0.0

    ans_rewards: list[float] = [p for p in [_last_answer_penalty(trajectory)] if p is not None]
    for hist in trajectory.histories:
        p = _last_answer_penalty(hist)
        if p is not None:
            ans_rewards.append(p)

    fmt_reward = sum(fmt_rewards) / len(fmt_rewards) if fmt_rewards else 0.0
    ans_reward = sum(ans_rewards) / len(ans_rewards) if ans_rewards else 0.0
    return fmt_reward + ans_reward


def format_reward(agent: BaseAgent, trajectory: Trajectory) -> float:
    """Agent-dispatched formatting penalty."""
    # if isinstance(agent, MarkerAgent):
    #     return _marker_format_reward(agent, trajectory)

    num_turns = len([m for m in trajectory.messages_and_responses if isinstance(m, InferenceResponse)])
    num_turns = max(num_turns, 1)  # avoid division by zero
    return -1.0 * len(trajectory.errors) / num_turns


def behavior_reward(agent: BaseAgent, trajectory: Trajectory) -> float:
    """Agent-dispatched behavior shaping."""
    return 0.0

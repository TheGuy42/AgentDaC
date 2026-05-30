from art.trajectories import Trajectory as ArtTrajectory, History as ArtHistory
from src import Trajectory, History


def convert_history(history: History) -> ArtHistory:
    """
    Converts Agent's history format to the framework specific `art.trajectories.History`:

    Args:
        history (src.History): The history to convert.
    """
    return ArtHistory(messages_and_choices=history.messages_and_choices, tools=history.tools)


def convert_trajectory(trajectory: Trajectory) -> ArtTrajectory:
    """
    Converts Agent's trajectory format to the framework specific `art.trajectories.Trajectory`:

    Args:
        trajectory (src.Trajectory): The trajectory to convert.
    """
    return ArtTrajectory(
        messages_and_choices=trajectory.messages_and_choices,
        tools=trajectory.tools,
        additional_histories=[convert_history(h) for h in trajectory.additional_histories],
        reward=trajectory.reward,
        metrics=trajectory.metrics,
        metadata=trajectory.metadata,
        logs=trajectory.logs,
        start_time=trajectory.start_time,
    )

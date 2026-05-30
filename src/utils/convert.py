from art.trajectories import Trajectory as ArtTrajectory, History as ArtHistory
from src import Trajectory, History
from src.aliases import Message, Response, Choice


def convert_messages(messages: list[Message | Response]) -> list[Message | Choice]:
    return [msg.choices[0] if isinstance(msg, Response) else msg for msg in messages]


def convert_history(history: History) -> ArtHistory:
    """
    Converts Agent's history format to the framework specific `art.trajectories.History`:

    Args:
        history (src.History): The history to convert.
    """
    return ArtHistory(
        messages_and_choices=convert_messages(history.messages_and_responses),
        tools=history.tools,
    )


def convert_trajectory(trajectory: Trajectory) -> ArtTrajectory:
    """
    Converts Agent's trajectory format to the framework specific `art.trajectories.Trajectory`:

    Args:
        trajectory (src.Trajectory): The trajectory to convert.
    """
    return ArtTrajectory(
        messages_and_choices=convert_messages(trajectory.messages_and_responses),
        tools=trajectory.tools,
        additional_histories=[convert_history(h) for h in trajectory.additional_histories],
        reward=trajectory.reward,
        metrics=trajectory.metrics,
        metadata=trajectory.metadata,
        logs=trajectory.logs,
        start_time=trajectory.start_time,
    )

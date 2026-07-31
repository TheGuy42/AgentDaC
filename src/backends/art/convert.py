from art.trajectories import Trajectory as ArtTrajectory, History as ArtHistory
from src.trajectory import Trajectory, TrajectoryError
from src.inference import OAIResponse, InferenceResponse
from src.aliases import Message, Choice


def convert_messages(messages: list[Message | InferenceResponse]) -> list[Message | Choice]:
    art_messages = []
    for msg in messages:
        if isinstance(msg, InferenceResponse):
            if not isinstance(msg, OAIResponse):
                raise ValueError(f"Unexpected InferenceResponse type: {type(msg)}")
            art_messages.append(msg.choice)

        else:
            art_messages.append(msg)
    return art_messages


def convert_errors(errors: list[TrajectoryError]) -> list[str]:
    return [str(error) for error in errors]


def convert_history(history: Trajectory) -> ArtHistory:
    return ArtHistory(
        messages_and_choices=convert_messages(history.messages_and_responses),
        tools=history.tools,
    )


def convert_trajectory(trajectory: Trajectory) -> ArtTrajectory:
    """
    Converts `Trajectory` format to the framework specific `art.trajectories.Trajectory`.

    Args:
        trajectory (src.Trajectory): The trajectory to convert.
    """
    return ArtTrajectory(
        messages_and_choices=convert_messages(trajectory.messages_and_responses),
        tools=trajectory.tools,
        reward=trajectory.reward,
        additional_histories=[convert_history(h) for h in trajectory.histories],
        metrics=trajectory.metrics,
        metadata=trajectory.metadata,
        logs=trajectory.logs + convert_errors(trajectory.errors),
        start_time=trajectory.start_time,
    )

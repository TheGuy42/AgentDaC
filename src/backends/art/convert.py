from art.trajectories import Trajectory as ArtTrajectory, History as ArtHistory
from src.trajectory import Trajectory, TrajectoryError, get_messages
from src.inference import OAIResponse, InferenceResponse
from src.aliases import Message, Choice
from src.utils.logging import create_logger


logger = create_logger(__name__)


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


def construct_episodes(trajectory: Trajectory) -> list[Trajectory]:
    """
    Splits a trajectory into episodes.

    An episode is defined as a sequence of messages and responses that ends with an InferenceResponse.
    Each episode contains all messages and responses up to and including the InferenceResponse.
    Only the last message in each episode is an InferenceResponse; all previous messages are regular messages.
    """

    episodes = []
    for i, msg in enumerate(trajectory.messages_and_responses):
        if isinstance(msg, InferenceResponse):
            context = get_messages(trajectory.messages_and_responses[:i])
            episode = Trajectory(messages_and_responses=context + [msg], tools=trajectory.tools)
            episodes.append(episode)

    if len(episodes) == 0:
        logger.warning("No `InferenceResponse` messages found in trajectory; returning the entire trajectory as a single episode.")
        episodes.append(trajectory)

    return episodes


def convert_trajectory(trajectory: Trajectory, episodic_split: bool) -> ArtTrajectory:
    """
    Converts `Trajectory` format to the framework specific `art.trajectories.Trajectory`.

    Args:
        trajectory (src.Trajectory): The trajectory to convert.
        episodic_split (bool): If True, split the trajectory into episodes based on InferenceResponse messages.
            Each episode will contain all messages and responses up to and including the InferenceResponse.
            If False, the entire trajectory will be treated as a single episode.
    """

    if episodic_split:
        main_episodes = construct_episodes(trajectory)
        main_trajectory = main_episodes.pop(-1)

        all_episodes = []
        all_episodes.extend(main_episodes)
        for history in trajectory.histories:
            all_episodes.extend(construct_episodes(history))

        trajectory = Trajectory(
            messages_and_responses=main_trajectory.messages_and_responses,
            tools=trajectory.tools,
            histories=all_episodes,
            reward=trajectory.reward,
            metrics=trajectory.metrics,
            metadata=trajectory.metadata,
            logs=trajectory.logs,
            errors=trajectory.errors,
            start_time=trajectory.start_time,
        )

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

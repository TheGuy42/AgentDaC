import art
from art.utils import trajectory_logging as art_logging
import polars as pl


def average_metrics(trajectories: list[art.Trajectory]) -> dict[str, float]:
    """
    Compute the average metrics from a list of trajectories.

    Args:
        trajectories (list[art.Trajectory]): List of trajectories to compute averages from.

    Returns:
        dict[str, float]: A dictionary containing the average of each metric across all trajectories.
    """

    if not trajectories:
        return {}

    keys = trajectories[0].metrics.keys()
    num = len(trajectories)
    avg_metrics = {key: sum(tr.metrics[key] for tr in trajectories) / num for key in keys}
    return avg_metrics


def to_dataframe(trajectories: list[art.Trajectory]) -> pl.DataFrame:
    """
    Convert a list of trajectories to a Polars DataFrame.

    Args:
        trajectories (list[art.Trajectory]): List of trajectories to convert.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the metadata, metrics, and rewards of the trajectories.
    """
    if not trajectories:
        return pl.DataFrame()
    data = []
    for tr in trajectories:
        row = art_logging.trajectory_to_dict(tr)
        data.append(row)
    return pl.DataFrame(data)


def read_trajectory_groups(file_path: str) -> list[art.TrajectoryGroup]:
    if not file_path.endswith(".jsonl"):
        raise ValueError("File must be a JSONL file.")
    
    with open(file_path, "r") as f:
        trajectory_groups = art_logging.deserialize_trajectory_groups(f.read())
    return trajectory_groups
"""
General replay buffer for loading and managing trajectory data from multiple JSONL files.

Key Features:
- Abstract base class with concrete implementations for different sorting/sampling strategies
- Automatically sorts trajectories within each group using _sort_group method
- Uses sample_group method for intelligent trajectory sampling
- Groups trajectories by specified metadata keys
- Dynamic loading of new files with automatic re-sorting
- Maintains sorted order in all data structures and DataFrames
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import polars as pl
import art
from art.utils import trajectory_logging as art_logging
from collections import defaultdict
import glob
from abc import ABC, abstractmethod
from src.utils.logging import create_logger


logger = create_logger(__name__)


# TODO: integrate into trainer
class GeneralReplayBuffer(ABC):
    """
    A general replay buffer that loads trajectory data from multiple JSONL files in a directory.

    This class automatically discovers and loads trajectory JSONL files, groups them by specified
    metadata keys, and provides functionality to update with new files as they appear.

    This is an abstract base class - subclasses must implement _sort_group and sample_group methods.
    """

    def __init__(
        self,
        directory: str,
        grouping_keys: Optional[Union[str, List[str]]] = None,
        buffer_size: Optional[int] = None,
    ):
        """
        Initialize the general replay buffer.

        Args:
            directory (str): Directory path containing trajectory JSONL files.
            grouping_keys (Optional[Union[str, List[str]]]): Key(s) from trajectory metadata
                to group trajectories by. Can be a single key or list of keys.
            buffer_size (Optional[int]): If specified, only load the last buffer_size epochs.
                Files are sorted by name and only the most recent buffer_size files are loaded.
        """
        self.directory = Path(directory)
        self.grouping_keys = [grouping_keys] if isinstance(grouping_keys, str) else grouping_keys or []
        self.buffer_size = buffer_size

        # Storage for loaded data
        self.trajectory_groups: List[art.TrajectoryGroup] = []
        self.file_paths: Dict[str, Path] = {}  # filename -> path mapping
        self.loaded_files: set = set()  # Track which files have been loaded
        self.grouped_trajectories: Dict[Tuple, List[art.Trajectory]] = defaultdict(list)
        self.df: Optional[pl.DataFrame] = None

        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Initial load
        self.update_trajectories()

    def _discover_jsonl_files(self) -> Dict[str, Path]:
        """
        Discover JSONL files in the directory, respecting buffer_size if specified.

        Returns:
            Dict[str, Path]: Mapping of filename to full path.
        """
        jsonl_pattern = str(self.directory / "*.jsonl")
        jsonl_files = glob.glob(jsonl_pattern)

        # Sort files by name to ensure consistent ordering
        jsonl_files.sort()

        # If buffer_size is specified, only keep the last buffer_size files
        if self.buffer_size is not None and len(jsonl_files) > self.buffer_size:
            jsonl_files = jsonl_files[-self.buffer_size :]
            logger.info(
                f"Buffer size limit: using last {self.buffer_size} files out of {len(glob.glob(jsonl_pattern))} total files"
            )

        file_mapping = {}
        for file_path in jsonl_files:
            path_obj = Path(file_path)
            file_mapping[path_obj.name] = path_obj

        return file_mapping

    def _load_file(self, file_path: Path) -> List[art.TrajectoryGroup]:
        """
        Load trajectory groups from a single JSONL file.

        Args:
            file_path (Path): Path to the JSONL file.

        Returns:
            List[art.TrajectoryGroup]: List of loaded trajectory groups.
        """
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                if not content:  # Skip empty files
                    return []
                return art_logging.deserialize_trajectory_groups(content)
        except Exception as e:
            logger.warning(f"Failed to load file {file_path}: {e}")
            return []

    def _get_grouping_key(self, trajectory: art.Trajectory) -> Tuple:
        """
        Extract the grouping key from a trajectory's metadata.

        Args:
            trajectory (art.Trajectory): The trajectory to extract key from.

        Returns:
            Tuple: Tuple of values for the grouping keys, or empty tuple if no keys specified.
        """
        if not self.grouping_keys:
            return ()

        key_values = []
        metadata = trajectory.metadata

        for key in self.grouping_keys:
            value = metadata.get(key, None)
            key_values.append(value)

        return tuple(key_values)

    def _get_group_epoch(self, group: art.TrajectoryGroup) -> Optional[Any]:
        """
        Extract the epoch identifier from a trajectory group.

        Args:
            group (art.TrajectoryGroup): The trajectory group to extract epoch from.

        Returns:
            Optional[Any]: The epoch identifier, or None if not found.
        """
        if group.trajectories:
            # Get epoch from the first trajectory's metadata
            first_traj = group.trajectories[0]
            if hasattr(first_traj, "metadata") and first_traj.metadata:
                return first_traj.metadata.get("epoch")
        return None

    def _group_trajectories(self) -> None:
        """
        Group all loaded trajectories by the specified grouping keys and sort each group.
        """
        self.grouped_trajectories.clear()

        for group in self.trajectory_groups:
            for trajectory in group.trajectories:
                grouping_key = self._get_grouping_key(trajectory)
                self.grouped_trajectories[grouping_key].append(trajectory)

        # Sort each group using the _sort_group method
        for group_key in self.grouped_trajectories:
            self.grouped_trajectories[group_key] = self._sort_group(self.grouped_trajectories[group_key])

    def update_trajectories(self) -> int:
        """
        Check for new JSONL files in the directory and load any new trajectories.
        Enforces buffer_size limit by discarding old files when new ones are added.

        Returns:
            int: Number of new files loaded.
        """
        # Discover current files (respects buffer_size)
        current_files = self._discover_jsonl_files()
        current_filenames = set(current_files.keys())

        # Find new files
        new_files = current_filenames - self.loaded_files

        # Find files to remove (old files no longer in current_files due to buffer_size)
        files_to_remove = self.loaded_files - current_filenames

        # Remove old files if buffer_size enforcement requires it
        if files_to_remove:
            logger.info(f"Buffer size enforcement: removing {len(files_to_remove)} old files")
            # Get epochs to remove
            epochs_to_remove = {Path(f).stem for f in files_to_remove}
            # Remove trajectory groups associated with old files
            self.trajectory_groups = [
                group for group in self.trajectory_groups if str(self._get_group_epoch(group)) not in epochs_to_remove
            ]
            # Update loaded files set
            self.loaded_files -= files_to_remove

        if not new_files:
            # Even if no new files, we may have removed old ones, so regroup
            if files_to_remove:
                self._group_trajectories()
                self.df = None
            return 0

        # Load new files
        new_trajectory_groups = []
        for filename in new_files:
            file_path = current_files[filename]
            logger.info(f"Loading new trajectory file: {filename}")

            groups = self._load_file(file_path)
            epoch = Path(file_path).stem  # Use filename (without extension) as epoch identifier
            # Add epoch metadata to each trajectory
            for group in groups:
                for traj in group.trajectories:
                    if not hasattr(traj, "metadata") or traj.metadata is None:
                        traj.metadata = {}
                    traj.metadata["epoch"] = epoch
            new_trajectory_groups.extend(groups)
            self.loaded_files.add(filename)

        # Add to existing trajectory groups
        self.trajectory_groups.extend(new_trajectory_groups)

        # Update file paths mapping
        self.file_paths.update(current_files)

        # Regroup all trajectories
        self._group_trajectories()

        # Clear cached DataFrame since data has changed
        self.df = None

        return len(new_files)

    def get_grouped_trajectories(self) -> Dict[Tuple, List[art.Trajectory]]:
        """
        Get trajectories grouped by the specified metadata keys.

        Returns:
            Dict[Tuple, List[art.Trajectory]]: Dictionary mapping grouping key tuples
                to lists of trajectories.
        """
        return dict(self.grouped_trajectories)

    def get_group_keys(self) -> List[Tuple]:
        """
        Get all unique grouping keys.

        Returns:
            List[Tuple]: List of all unique grouping key tuples.
        """
        return list(self.grouped_trajectories.keys())

    def get_trajectories_for_group(self, group_key: Tuple) -> List[art.Trajectory]:
        """
        Get all trajectories for a specific group key.

        Args:
            group_key (Tuple): The grouping key tuple.

        Returns:
            List[art.Trajectory]: List of trajectories for the specified group.
        """
        return self.grouped_trajectories.get(group_key, [])

    def create_dataframe(self) -> pl.DataFrame:
        """
        Create a DataFrame from all loaded trajectories with grouping information.

        Returns:
            pl.DataFrame: DataFrame containing trajectory data with grouping columns.
        """
        if self.df is not None:
            return self.df

        data = []

        for group_key, trajectories in self.grouped_trajectories.items():
            for traj_idx, trajectory in enumerate(trajectories):
                # Convert trajectory to dict using art's built-in function
                traj_dict = art_logging.trajectory_to_dict(trajectory)

                # Add group key information
                row = {
                    "trajectory_id": traj_idx,
                    "reward": (trajectory.reward if hasattr(trajectory, "reward") else None),
                }

                # Add grouping key columns
                for i, key_name in enumerate(self.grouping_keys):
                    if i < len(group_key):
                        row[f"group_{key_name}"] = group_key[i]
                    else:
                        row[f"group_{key_name}"] = None

                # Add a combined group identifier
                if group_key:
                    row["group_id"] = str(group_key)
                else:
                    row["group_id"] = "default"

                # Add metrics with 'metric_' prefix
                if hasattr(trajectory, "metrics") and trajectory.metrics:
                    for key, value in trajectory.metrics.items():
                        row[f"metric_{key}"] = value

                # Add metadata with 'metadata_' prefix
                if hasattr(trajectory, "metadata") and trajectory.metadata:
                    for key, value in trajectory.metadata.items():
                        row[f"metadata_{key}"] = value

                # Add any additional fields from traj_dict
                for key, value in traj_dict.items():
                    if key not in row:
                        row[key] = value

                data.append(row)

        if not data:
            # Return empty DataFrame with expected columns
            columns = ["trajectory_id", "reward", "group_id"]
            columns.extend([f"group_{key}" for key in self.grouping_keys])
            self.df = pl.DataFrame({col: [] for col in columns})
        else:
            self.df = pl.DataFrame(data)

            # Clean up columns similar to TrajectoryGroupLoader
            columns_to_drop = [
                "metadata",
                "metrics",
                "messages_and_choices",
                "additional_histories",
                "tools",
            ]
            for col in columns_to_drop:
                if col in self.df.columns:
                    self.df = self.df.drop(col)

        return self.df

    def get_group_statistics(self) -> pl.DataFrame:
        """
        Get statistics for each trajectory group.

        Returns:
            pl.DataFrame: DataFrame with group-level statistics.
        """
        if self.df is None:
            self.create_dataframe()

        assert self.df is not None, "DataFrame should be created by now"

        if self.df.height == 0:  # Use height instead of is_empty()
            return pl.DataFrame()

        # Get reward statistics by group
        group_stats = (
            self.df.group_by("group_id")
            .agg(
                [
                    pl.col("trajectory_id").count().alias("num_trajectories"),
                    pl.col("reward").mean().alias("mean_reward"),
                    pl.col("reward").std().alias("std_reward"),
                    pl.col("reward").min().alias("min_reward"),
                    pl.col("reward").max().alias("max_reward"),
                ]
            )
            .sort("group_id")
        )

        return group_stats

    def filter_by_group_key(self, **kwargs) -> pl.DataFrame:
        """
        Filter trajectories by specific grouping key values.

        Args:
            **kwargs: Key-value pairs where keys are grouping key names and values
                     are the desired values to filter by.

        Returns:
            pl.DataFrame: Filtered DataFrame.
        """
        if self.df is None:
            self.create_dataframe()

        assert self.df is not None, "DataFrame should be created by now"

        filtered_df = self.df

        for key, value in kwargs.items():
            column_name = f"group_{key}"
            if column_name in filtered_df.columns:
                filtered_df = filtered_df.filter(pl.col(column_name) == value)

        return filtered_df

    def sample_trajectories(self, n_per_group: int, seed: Optional[int] = None) -> pl.DataFrame:
        """
        Sample n trajectories from each group using the sample_group method.

        Args:
            n_per_group (int): Number of trajectories to sample per group.
            seed (Optional[int]): Random seed for reproducible sampling.

        Returns:
            pl.DataFrame: DataFrame with sampled trajectories.
        """
        if seed is not None:
            pl.set_random_seed(seed)

        # Sample from trajectory groups directly using sample_group method
        sampled_trajectory_groups = {}
        for group_key, trajectories in self.grouped_trajectories.items():
            if len(trajectories) > 0:
                sampled_trajectories = self._sample_group(trajectories, n_per_group)
                sampled_trajectory_groups[group_key] = sampled_trajectories

        # Create DataFrame from sampled trajectories
        data = []
        for group_key, trajectories in sampled_trajectory_groups.items():
            for traj_idx, trajectory in enumerate(trajectories):
                # Convert trajectory to dict using art's built-in function
                traj_dict = art_logging.trajectory_to_dict(trajectory)

                # Add group key information
                row = {
                    "trajectory_id": traj_idx,
                    "reward": (trajectory.reward if hasattr(trajectory, "reward") else None),
                }

                # Add grouping key columns
                for i, key_name in enumerate(self.grouping_keys):
                    if i < len(group_key):
                        row[f"group_{key_name}"] = group_key[i]
                    else:
                        row[f"group_{key_name}"] = None

                # Add a combined group identifier
                if group_key:
                    row["group_id"] = str(group_key)
                else:
                    row["group_id"] = "default"

                # Add metrics with 'metric_' prefix
                if hasattr(trajectory, "metrics") and trajectory.metrics:
                    for key, value in trajectory.metrics.items():
                        row[f"metric_{key}"] = value

                # Add metadata with 'metadata_' prefix
                if hasattr(trajectory, "metadata") and trajectory.metadata:
                    for key, value in trajectory.metadata.items():
                        row[f"metadata_{key}"] = value

                # Add any additional fields from traj_dict
                for key, value in traj_dict.items():
                    if key not in row:
                        row[key] = value

                data.append(row)

        if not data:
            # Return empty DataFrame with expected columns
            columns = ["trajectory_id", "reward", "group_id"]
            columns.extend([f"group_{key}" for key in self.grouping_keys])
            return pl.DataFrame({col: [] for col in columns})
        else:
            sampled_df = pl.DataFrame(data)

            # Clean up columns similar to create_dataframe
            columns_to_drop = [
                "metadata",
                "metrics",
                "messages_and_choices",
                "additional_histories",
                "tools",
            ]
            for col in columns_to_drop:
                if col in sampled_df.columns:
                    sampled_df = sampled_df.drop(col)

            return sampled_df.sort("group_id")

    def export_grouped_data(self, output_dir: str, format: str = "parquet") -> None:
        """
        Export each group's data to separate files.

        Args:
            output_dir (str): Directory to save the files.
            format (str): Format to save files in ('parquet' or 'csv').
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.df is None:
            self.create_dataframe()

        assert self.df is not None, "DataFrame should be created by now"

        for group_id in self.df["group_id"].unique():
            group_df = self.df.filter(pl.col("group_id") == group_id)

            # Create safe filename
            safe_group_id = str(group_id).replace("(", "").replace(")", "").replace(", ", "_").replace("'", "")
            filename = f"group_{safe_group_id}.{format}"
            file_path = output_path / filename

            if format == "parquet":
                group_df.write_parquet(file_path)
            elif format == "csv":
                group_df.write_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

    @property
    def dataframe(self) -> pl.DataFrame:
        """
        Get the DataFrame (creates it if not already created).

        Returns:
            pl.DataFrame: The trajectory DataFrame.
        """
        if self.df is None:
            self.create_dataframe()

        assert self.df is not None, "DataFrame should be created by now"
        return self.df

    @property
    def num_files_loaded(self) -> int:
        """
        Get the number of files currently loaded.

        Returns:
            int: Number of loaded files.
        """
        return len(self.loaded_files)

    @property
    def total_trajectories(self) -> int:
        """
        Get the total number of trajectories across all groups.

        Returns:
            int: Total number of trajectories.
        """
        return sum(len(trajectories) for trajectories in self.grouped_trajectories.values())

    @property
    def num_groups(self) -> int:
        """
        Get the number of unique groups.

        Returns:
            int: Number of groups.
        """
        return len(self.grouped_trajectories)

    def __repr__(self) -> str:
        """String representation of the replay buffer."""
        buffer_info = f", buffer_size={self.buffer_size}" if self.buffer_size else ""
        return (
            f"GeneralReplayBuffer(directory='{self.directory}', "
            f"grouping_keys={self.grouping_keys}, "
            f"files_loaded={self.num_files_loaded}, "
            f"groups={self.num_groups}, "
            f"total_trajectories={self.total_trajectories}{buffer_info})"
        )

    @abstractmethod
    def _sort_group(self, trajectories: List[art.Trajectory]) -> List[art.Trajectory]:
        """
        Sort trajectories within a group.

        This method should be implemented by subclasses to define meaningful sorting
        for the specific use case (e.g., by reward, by timestamp, etc.).

        Args:
            trajectories (List[art.Trajectory]): List of trajectories to sort.

        Returns:
            List[art.Trajectory]: Sorted list of trajectories.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _sample_group(self, trajectories: List[art.Trajectory], n: int) -> List[art.Trajectory]:
        """
        Sample n trajectories from a group.

        This method should be implemented by subclasses to define the sampling strategy
        (e.g., top-n, random, weighted sampling, etc.).

        Args:
            trajectories (List[art.Trajectory]): List of trajectories in the group (already sorted).
            n (int): Number of trajectories to sample.

        Returns:
            List[art.Trajectory]: List of n sampled trajectories.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def sample_group(self, group_key: tuple, n: int) -> List[art.Trajectory]:
        """
        Sample n trajectories from a specific group identified by group_key.

        Args:
            group_key (Tuple): The grouping key tuple identifying the group.
            n (int): Number of trajectories to sample.

        Returns:
            List[art.Trajectory]: List of n sampled trajectories from the specified group.
        """
        trajectories = self.get_trajectories_for_group(group_key)
        if not trajectories:
            return []
        return self._sample_group(trajectories, n)


class RewardBasedDoubleQuantileReplayBuffer(GeneralReplayBuffer):
    """
    A concrete implementation of GeneralReplayBuffer that sorts trajectories by reward
    and samples n trajectories from the top and bottom k quantiles in each group.
    """

    def __init__(
        self,
        directory: str,
        grouping_keys: Optional[Union[str, List[str]]] = None,
        quantile_fraction: float = 0.2,
        upper_only: bool = True,
        buffer_size: Optional[int] = None,
    ):
        """
        Initialize the reward-based double quantile replay buffer.

        Args:
            directory (str): Directory path containing trajectory JSONL files.
            grouping_keys (Optional[Union[str, List[str]]]): Key(s) from trajectory metadata
                to group trajectories by. Can be a single key or list of keys.
            quantile_fraction (float): Fraction of trajectories to sample from top and bottom quantiles.
            buffer_size (Optional[int]): If specified, only load the last buffer_size epochs.
        """
        super().__init__(directory, grouping_keys, buffer_size)
        if not (0 < quantile_fraction < 0.5):
            raise ValueError("quantile_fraction must be between 0 and 0.5")
        self.quantile_fraction = quantile_fraction
        self.upper_only = upper_only

    def _sort_group(self, trajectories: List[art.Trajectory]) -> List[art.Trajectory]:
        """
        Sort trajectories by reward in descending order (highest reward first).

        Args:
            trajectories (List[art.Trajectory]): List of trajectories to sort.

        Returns:
            List[art.Trajectory]: Sorted list of trajectories (highest reward first).
        """
        return sorted(
            trajectories,
            key=lambda traj: traj.reward,
            reverse=True,
        )

    def _sample_group(self, trajectories: List[art.Trajectory], n: int) -> List[art.Trajectory]:
        """
        Sample the top n trajectories by reward from a group.

        Args:
            trajectories (List[art.Trajectory]): List of trajectories in the group (already sorted by reward).
            n (int): Number of top trajectories to sample.

        Returns:
            List[art.Trajectory]: List of top n trajectories by reward.
        """
        n_traj = len(trajectories)
        top_k_idx = int(n_traj * (1 - self.quantile_fraction))
        bottom_k_idx = int(n_traj * self.quantile_fraction)

        top_k = trajectories[top_k_idx - n // 4 :][: n // 2]
        bottom_k = trajectories[: bottom_k_idx + n // 4][-n // 2 :]

        if self.upper_only:
            return top_k
        else:
            return bottom_k + top_k


class RewardBasedReplayBuffer(GeneralReplayBuffer):
    """
    A concrete implementation of GeneralReplayBuffer that sorts trajectories by reward
    and samples the top-performing trajectories from each group.
    """

    def _sort_group(self, trajectories: List[art.Trajectory]) -> List[art.Trajectory]:
        """
        Sort trajectories by reward in descending order (highest reward first).

        Args:
            trajectories (List[art.Trajectory]): List of trajectories to sort.

        Returns:
            List[art.Trajectory]: Sorted list of trajectories (highest reward first).
        """
        return sorted(
            trajectories,
            key=lambda traj: traj.reward,
            reverse=True,
        )

    def _sample_group(self, trajectories: List[art.Trajectory], n: int) -> List[art.Trajectory]:
        """
        Sample the top n trajectories by reward from a group.

        Args:
            trajectories (List[art.Trajectory]): List of trajectories in the group (already sorted by reward).
            n (int): Number of top trajectories to sample.

        Returns:
            List[art.Trajectory]: List of top n trajectories by reward.
        """
        return trajectories[: min(n, len(trajectories))]


class RandomReplayBuffer(GeneralReplayBuffer):
    """
    A concrete implementation of GeneralReplayBuffer that keeps original order
    and samples randomly from each group.
    """

    def _sort_group(self, trajectories: List[art.Trajectory]) -> List[art.Trajectory]:
        """
        Keep trajectories in their original order.

        Args:
            trajectories (List[art.Trajectory]): List of trajectories to sort.

        Returns:
            List[art.Trajectory]: Trajectories in original order.
        """
        return trajectories

    def _sample_group(self, trajectories: List[art.Trajectory], n: int) -> List[art.Trajectory]:
        """
        Randomly sample n trajectories from a group.

        Args:
            trajectories (List[art.Trajectory]): List of trajectories in the group.
            n (int): Number of trajectories to sample.

        Returns:
            List[art.Trajectory]: List of n randomly sampled trajectories.
        """
        import random

        return random.sample(trajectories, min(n, len(trajectories)))


# Example usage:
"""
# Create a reward-based replay buffer that groups trajectories by 'experiment_name' metadata
# and sorts by reward, sampling top performers
replay_buffer = RewardBasedReplayBuffer(
    directory="/path/to/trajectory/logs",
    grouping_keys=["experiment_name"]
)

# Or create a random sampling replay buffer
random_buffer = RandomReplayBuffer(
    directory="/path/to/trajectory/logs",
    grouping_keys=["experiment_name"]
)

# Check initial state
print(replay_buffer)
print(f"Loaded {replay_buffer.num_files_loaded} files")
print(f"Found {replay_buffer.num_groups} groups")

# Get all group keys
group_keys = replay_buffer.get_group_keys()
print(f"Group keys: {group_keys}")

# Update with any new files (returns number of new files loaded)
# Note: This will re-sort all groups according to _sort_group implementation
new_files = replay_buffer.update_trajectories()
print(f"Loaded {new_files} new files")

# Get DataFrame with all trajectories (sorted within each group)
df = replay_buffer.dataframe
print(f"DataFrame shape: {df.shape}")

# Filter by specific group
filtered_df = replay_buffer.filter_by_group_key(experiment_name="my_experiment")

# Get group statistics
stats = replay_buffer.get_group_statistics()
print(stats)

# Sample trajectories from each group (uses sample_group method)
# For RewardBasedReplayBuffer: samples top n by reward
# For RandomReplayBuffer: samples randomly
sampled = replay_buffer.sample_trajectories(n_per_group=10, seed=42)

# Export grouped data
replay_buffer.export_grouped_data("/path/to/output", format="parquet")

# For multiple grouping keys (e.g., by experiment and model):
multi_group_buffer = RewardBasedReplayBuffer(
    directory="/path/to/trajectory/logs",
    grouping_keys=["experiment_name", "model_name"]
)

# Filter by multiple keys
filtered = multi_group_buffer.filter_by_group_key(
    experiment_name="my_experiment",
    model_name="gpt-4"
)

# Custom implementation example:
class TimestampBasedReplayBuffer(GeneralReplayBuffer):
    '''Custom replay buffer that sorts by timestamp and samples recent trajectories.'''
    
    def _sort_group(self, trajectories):
        # Sort by timestamp (newest first)
        return sorted(trajectories, 
                     key=lambda traj: getattr(traj.metadata, 'timestamp', 0), 
                     reverse=True)
    
    def sample_group(self, trajectories, n):
        # Sample the most recent n trajectories
        return trajectories[:min(n, len(trajectories))]
"""

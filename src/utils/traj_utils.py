"""
Utilities for loading and processing trajectory data from JSONL files.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import polars as pl
import art
from art.utils import trajectory_logging as art_logging


class TrajectoryGroupLoader:
    """
    A class for loading trajectory group JSONL files and converting them to DataFrames.
    
    This class creates a DataFrame where each row represents a separate trajectory,
    with columns for metrics, metadata, rewards, and group identifiers.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the trajectory group loader.
        
        Args:
            file_path (str): Path to the trajectory group JSONL file.
        """
        self.file_path = Path(file_path)
        self.trajectory_groups: List[art.TrajectoryGroup] = []
        self.df: Optional[pl.DataFrame] = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.file_path.suffix == '.jsonl':
            raise ValueError("File must be a JSONL file")
    
    def load_trajectory_groups(self) -> List[art.TrajectoryGroup]:
        """
        Load trajectory groups from the JSONL file.
        
        Returns:
            List[art.TrajectoryGroup]: List of loaded trajectory groups.
        """
        with open(self.file_path, "r") as f:
            self.trajectory_groups = art_logging.deserialize_trajectory_groups(f.read())
        return self.trajectory_groups
    
    def create_dataframe(self) -> pl.DataFrame:
        """
        Create a DataFrame from the loaded trajectory groups.
        
        Each row represents a trajectory with the following columns:
        - group_id: Identifier for the trajectory group
        - trajectory_id: Index of the trajectory within its group
        - reward: Trajectory reward value
        - All metrics from trajectory.metrics
        - All metadata from trajectory.metadata
        
        Returns:
            pl.DataFrame: DataFrame containing trajectory data.
        """
        if not self.trajectory_groups:
            self.load_trajectory_groups()
        
        data = []
        
        for group_idx, group in enumerate(self.trajectory_groups):
            for traj_idx, trajectory in enumerate(group.trajectories):
                # Convert trajectory to dict using art's built-in function
                traj_dict = art_logging.trajectory_to_dict(trajectory)
                
                # Add group and trajectory identifiers
                row = {
                    'group_id': group_idx,
                    'trajectory_id': traj_idx,
                    'reward': trajectory.reward if hasattr(trajectory, 'reward') else None,
                }
                
                # Add metrics with 'metric_' prefix
                if hasattr(trajectory, 'metrics') and trajectory.metrics:
                    for key, value in trajectory.metrics.items():
                        row[f'metric_{key}'] = value
                
                # Add metadata with 'metadata_' prefix
                if hasattr(trajectory, 'metadata') and trajectory.metadata:
                    for key, value in trajectory.metadata.items():
                        row[f'metadata_{key}'] = value
                
                # Add any additional fields from traj_dict
                for key, value in traj_dict.items():
                    if key not in row:
                        row[key] = value
                
                data.append(row)
        
        self.df = pl.DataFrame(data)
        if 'metadata' in self.df.columns:
            self.df = self.df.drop('metadata')  # Drop metadata column if it exists
        if 'metrics' in self.df.columns:
            self.df = self.df.drop('metrics')
        
        self.df = self.df.drop('messages_and_choices', 'additional_histories', 'tools')

        return self.df
    
    def get_group_statistics(self) -> pl.DataFrame:
        """
        Get statistics for each trajectory group.
        
        Returns:
            pl.DataFrame: DataFrame with group-level statistics including:
                - group_id: Group identifier
                - num_trajectories: Number of trajectories in the group
                - mean_reward: Average reward in the group
                - std_reward: Standard deviation of rewards in the group
                - min_reward: Minimum reward in the group
                - max_reward: Maximum reward in the group
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        # Get reward statistics by group
        group_stats = self.df.group_by('group_id').agg([
            pl.col('trajectory_id').count().alias('num_trajectories'),
            pl.col('reward').mean().alias('mean_reward'),
            pl.col('reward').std().alias('std_reward'),
            pl.col('reward').min().alias('min_reward'),
            pl.col('reward').max().alias('max_reward'),
        ]).sort('group_id')
        
        return group_stats
    
    def get_metric_statistics(self) -> Dict[str, pl.DataFrame]:
        """
        Get statistics for each metric across all trajectories.
        
        Returns:
            Dict[str, pl.DataFrame]: Dictionary mapping metric names to their statistics.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        metric_stats = {}
        
        # Find all metric columns
        metric_columns = [col for col in self.df.columns if col.startswith('metric_')]
        
        for metric_col in metric_columns:
            metric_name = metric_col.replace('metric_', '')
            
            # Calculate statistics for this metric
            stats = self.df.select([
                pl.col(metric_col).mean().alias('mean'),
                pl.col(metric_col).std().alias('std'),
                pl.col(metric_col).min().alias('min'),
                pl.col(metric_col).max().alias('max'),
                pl.col(metric_col).count().alias('count'),
                pl.col(metric_col).null_count().alias('null_count'),
            ])
            
            metric_stats[metric_name] = stats
        
        return metric_stats
    
    def filter_by_group(self, group_ids: List[int]) -> pl.DataFrame:
        """
        Filter trajectories by group IDs.
        
        Args:
            group_ids (List[int]): List of group IDs to include.
            
        Returns:
            pl.DataFrame: Filtered DataFrame containing only trajectories from specified groups.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        return self.df.filter(pl.col('group_id').is_in(group_ids))
    
    def filter_by_reward_range(self, min_reward: Optional[float] = None, 
                              max_reward: Optional[float] = None) -> pl.DataFrame:
        """
        Filter trajectories by reward range.
        
        Args:
            min_reward (Optional[float]): Minimum reward threshold (inclusive).
            max_reward (Optional[float]): Maximum reward threshold (inclusive).
            
        Returns:
            pl.DataFrame: Filtered DataFrame containing trajectories within the reward range.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        filtered_df = self.df
        
        if min_reward is not None:
            filtered_df = filtered_df.filter(pl.col('reward') >= min_reward)
        
        if max_reward is not None:
            filtered_df = filtered_df.filter(pl.col('reward') <= max_reward)
        
        return filtered_df
    
    def get_top_trajectories_per_group(self, n: int = 1) -> pl.DataFrame:
        """
        Get the top N trajectories from each group based on reward.
        
        Args:
            n (int): Number of top trajectories to get per group.
            
        Returns:
            pl.DataFrame: DataFrame containing the top N trajectories from each group.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        return (self.df
                .sort(['group_id', 'reward'], descending=[False, True])
                .group_by('group_id')
                .head(n)
                .sort(['group_id', 'reward'], descending=[False, True]))
    
    def export_to_csv(self, output_path: str) -> None:
        """
        Export the DataFrame to a CSV file.
        
        Args:
            output_path (str): Path where to save the CSV file.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        self.df.write_csv(output_path)
    
    def export_to_parquet(self, output_path: str) -> None:
        """
        Export the DataFrame to a Parquet file.
        
        Args:
            output_path (str): Path where to save the Parquet file.
        """
        if self.df is None:
            self.create_dataframe()
        
        assert self.df is not None, "DataFrame should be created by now"
        
        self.df.write_parquet(output_path)
    
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
    def num_groups(self) -> int:
        """
        Get the number of trajectory groups.
        
        Returns:
            int: Number of trajectory groups.
        """
        if not self.trajectory_groups:
            self.load_trajectory_groups()
        return len(self.trajectory_groups)
    
    @property
    def total_trajectories(self) -> int:
        """
        Get the total number of trajectories across all groups.
        
        Returns:
            int: Total number of trajectories.
        """
        if not self.trajectory_groups:
            self.load_trajectory_groups()
        return sum(len(group.trajectories) for group in self.trajectory_groups)


def load_trajectory_dataframe(file_path: str) -> pl.DataFrame:
    """
    Convenience function to load a trajectory JSONL file and return a DataFrame.
    
    Args:
        file_path (str): Path to the trajectory group JSONL file.
        
    Returns:
        pl.DataFrame: DataFrame containing trajectory data.
    """
    loader = TrajectoryGroupLoader(file_path)
    return loader.create_dataframe()


def combine_trajectory_groups(file_paths: List[str], labels: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Load and combine multiple trajectory group files.
    
    Args:
        file_paths (List[str]): List of paths to trajectory group JSONL files.
        labels (Optional[List[str]]): Optional labels for each file. If not provided,
                                    file names will be used.
        
    Returns:
        pl.DataFrame: Combined DataFrame with an additional 'source' column indicating
                     which file each trajectory came from.
    """
    if labels is None:
        labels = [Path(path).stem for path in file_paths]
    
    if len(labels) != len(file_paths):
        raise ValueError("Number of labels must match number of file paths")
    
    combined_data = []
    
    for file_path, label in zip(file_paths, labels):
        loader = TrajectoryGroupLoader(file_path)
        df = loader.create_dataframe()
        df = df.with_columns(pl.lit(label).alias('epoch'))
        combined_data.append(df)
    
    return pl.concat(combined_data, how="diagonal")

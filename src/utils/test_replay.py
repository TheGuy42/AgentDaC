"""
Test file for replay buffer implementations.

This module contains comprehensive tests for the GeneralReplayBuffer abstract class
and its concrete implementations (RewardBasedReplayBuffer, RandomReplayBuffer, etc.).
"""

import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
import polars as pl
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay import (
    GeneralReplayBuffer,
    RewardBasedReplayBuffer,
    RandomReplayBuffer,
    RewardBasedDoubleQuantileReplayBuffer,
)


class MockTrajectory:
    """Mock trajectory class for testing."""
    
    def __init__(self, reward: float, metadata: Dict[str, Any] = None, metrics: Dict[str, Any] = None):
        self.reward = reward
        self.metadata = metadata if metadata is not None else {}
        self.metrics = metrics if metrics is not None else {}


class MockTrajectoryGroup:
    """Mock trajectory group class for testing."""
    
    def __init__(self, trajectories: List[MockTrajectory]):
        self.trajectories = trajectories


def create_mock_trajectory_group(trajectories: List[MockTrajectory]) -> MockTrajectoryGroup:
    """Create a mock trajectory group for testing."""
    return MockTrajectoryGroup(trajectories)


def create_test_jsonl_file(file_path: Path, trajectory_groups: List[MockTrajectoryGroup]):
    """Create a test JSONL file with trajectory groups."""
    # For testing purposes, we'll create a simplified JSONL format
    
    data = []
    for group in trajectory_groups:
        group_data = {
            "trajectories": []
        }
        for traj in group.trajectories:
            traj_data = {
                "reward": traj.reward,
                "metadata": traj.metadata,
                "metrics": traj.metrics,
                # Add minimal required fields for art compatibility
                "messages_and_choices": [],
                "additional_histories": [],
                "tools": []
            }
            group_data["trajectories"].append(traj_data)
        data.append(group_data)
    
    with open(file_path, 'w') as f:
        for group_data in data:
            f.write(json.dumps(group_data) + '\n')


class TestableRewardBasedReplayBuffer(RewardBasedReplayBuffer):
    """Testable version that doesn't require actual art trajectory deserialization."""
    
    def _load_file(self, file_path: Path):
        """Override to load our mock data."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                trajectory_groups = []
                
                for line in lines:
                    if line.strip():
                        group_data = json.loads(line)
                        trajectories = []
                        
                        for traj_data in group_data["trajectories"]:
                            mock_traj = MockTrajectory(
                                reward=traj_data["reward"],
                                metadata=traj_data["metadata"],
                                metrics=traj_data["metrics"]
                            )
                            trajectories.append(mock_traj)
                        
                        trajectory_groups.append(create_mock_trajectory_group(trajectories))
                
                return trajectory_groups
        except Exception as e:
            print(f"Warning: Failed to load file {file_path}: {e}")
            return []


class TestableRandomReplayBuffer(RandomReplayBuffer):
    """Testable version that doesn't require actual art trajectory deserialization."""
    
    def _load_file(self, file_path: Path):
        """Override to load our mock data."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                trajectory_groups = []
                
                for line in lines:
                    if line.strip():
                        group_data = json.loads(line)
                        trajectories = []
                        
                        for traj_data in group_data["trajectories"]:
                            mock_traj = MockTrajectory(
                                reward=traj_data["reward"],
                                metadata=traj_data["metadata"],
                                metrics=traj_data["metrics"]
                            )
                            trajectories.append(mock_traj)
                        
                        trajectory_groups.append(create_mock_trajectory_group(trajectories))
                
                return trajectory_groups
        except Exception as e:
            print(f"Warning: Failed to load file {file_path}: {e}")
            return []


class TestableDoubleQuantileReplayBuffer(RewardBasedDoubleQuantileReplayBuffer):
    """Testable version that doesn't require actual art trajectory deserialization."""
    
    def _load_file(self, file_path: Path):
        """Override to load our mock data."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                trajectory_groups = []
                
                for line in lines:
                    if line.strip():
                        group_data = json.loads(line)
                        trajectories = []
                        
                        for traj_data in group_data["trajectories"]:
                            mock_traj = MockTrajectory(
                                reward=traj_data["reward"],
                                metadata=traj_data["metadata"],
                                metrics=traj_data["metrics"]
                            )
                            trajectories.append(mock_traj)
                        
                        trajectory_groups.append(create_mock_trajectory_group(trajectories))
                
                return trajectory_groups
        except Exception as e:
            print(f"Warning: Failed to load file {file_path}: {e}")
            return []


def test_reward_based_replay_buffer():
    """Test RewardBasedReplayBuffer functionality."""
    print("Testing RewardBasedReplayBuffer...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test trajectories with different rewards and experiment names
        trajectories_exp1 = [
            MockTrajectory(reward=0.8, metadata={"experiment_name": "exp1", "model": "gpt-4"}),
            MockTrajectory(reward=0.6, metadata={"experiment_name": "exp1", "model": "gpt-4"}),
            MockTrajectory(reward=0.9, metadata={"experiment_name": "exp1", "model": "gpt-4"}),
            MockTrajectory(reward=0.4, metadata={"experiment_name": "exp1", "model": "gpt-4"}),
        ]
        
        trajectories_exp2 = [
            MockTrajectory(reward=0.7, metadata={"experiment_name": "exp2", "model": "gpt-3.5"}),
            MockTrajectory(reward=0.5, metadata={"experiment_name": "exp2", "model": "gpt-3.5"}),
            MockTrajectory(reward=0.85, metadata={"experiment_name": "exp2", "model": "gpt-3.5"}),
        ]
        
        # Create trajectory groups
        group1 = create_mock_trajectory_group(trajectories_exp1)
        group2 = create_mock_trajectory_group(trajectories_exp2)
        
        # Create test files
        file1 = temp_path / "test1.jsonl"
        file2 = temp_path / "test2.jsonl"
        
        create_test_jsonl_file(file1, [group1])
        create_test_jsonl_file(file2, [group2])
        
        # Test single grouping key
        replay_buffer = TestableRewardBasedReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name"]
        )
        
        # Test basic properties
        assert replay_buffer.num_files_loaded == 2
        assert replay_buffer.total_trajectories == 7
        assert replay_buffer.num_groups == 2
        
        # Test group keys
        group_keys = replay_buffer.get_group_keys()
        assert len(group_keys) == 2
        assert ("exp1",) in group_keys
        assert ("exp2",) in group_keys
        
        # Test sorting within groups
        exp1_trajectories = replay_buffer.get_trajectories_for_group(("exp1",))
        exp2_trajectories = replay_buffer.get_trajectories_for_group(("exp2",))
        
        # Check that trajectories are sorted by reward (highest first)
        exp1_rewards = [traj.reward for traj in exp1_trajectories]
        exp2_rewards = [traj.reward for traj in exp2_trajectories]
        
        assert exp1_rewards == [0.9, 0.8, 0.6, 0.4]  # Sorted descending
        assert exp2_rewards == [0.85, 0.7, 0.5]  # Sorted descending
        
        # Test sampling (should get top performers)
        sampled_df = replay_buffer.sample_trajectories(n_per_group=2)
        
        # Should have 2 trajectories per group
        exp1_sampled = sampled_df.filter(pl.col("group_experiment_name") == "exp1")
        exp2_sampled = sampled_df.filter(pl.col("group_experiment_name") == "exp2")
        
        assert exp1_sampled.height == 2
        assert exp2_sampled.height == 2
        
        # Check that we got the top rewards
        exp1_sampled_rewards = exp1_sampled["reward"].to_list()
        exp2_sampled_rewards = exp2_sampled["reward"].to_list()
        
        assert set(exp1_sampled_rewards) == {0.9, 0.8}  # Top 2 from exp1
        assert set(exp2_sampled_rewards) == {0.85, 0.7}  # Top 2 from exp2
        
        print("✓ RewardBasedReplayBuffer tests passed!")


def test_random_replay_buffer():
    """Test RandomReplayBuffer functionality."""
    print("Testing RandomReplayBuffer...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test trajectories
        trajectories = [
            MockTrajectory(reward=0.8, metadata={"experiment_name": "exp1"}),
            MockTrajectory(reward=0.6, metadata={"experiment_name": "exp1"}),
            MockTrajectory(reward=0.9, metadata={"experiment_name": "exp1"}),
            MockTrajectory(reward=0.4, metadata={"experiment_name": "exp1"}),
        ]
        
        group = create_mock_trajectory_group(trajectories)
        file_path = temp_path / "test.jsonl"
        create_test_jsonl_file(file_path, [group])
        
        # Test random replay buffer
        replay_buffer = TestableRandomReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name"]
        )
        
        # Test that trajectories maintain original order (no sorting)
        exp1_trajectories = replay_buffer.get_trajectories_for_group(("exp1",))
        original_rewards = [traj.reward for traj in exp1_trajectories]
        
        assert original_rewards == [0.8, 0.6, 0.9, 0.4]  # Original order
        
        # Test random sampling (with seed for reproducibility)
        sampled_df = replay_buffer.sample_trajectories(n_per_group=2, seed=42)
        
        assert sampled_df.height == 2
        
        print("✓ RandomReplayBuffer tests passed!")


def test_double_quantile_replay_buffer():
    """Test RewardBasedDoubleQuantileReplayBuffer functionality."""
    print("Testing RewardBasedDoubleQuantileReplayBuffer...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test trajectories with a range of rewards
        trajectories = []
        for i in range(10):
            trajectories.append(
                MockTrajectory(reward=i * 0.1, metadata={"experiment_name": "exp1"})
            )
        
        group = create_mock_trajectory_group(trajectories)
        file_path = temp_path / "test.jsonl"
        create_test_jsonl_file(file_path, [group])
        
        # Test double quantile replay buffer
        replay_buffer = TestableDoubleQuantileReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name"],
            quantile_fraction=0.2
        )
        
        # Test that trajectories are sorted by reward (highest first)
        exp1_trajectories = replay_buffer.get_trajectories_for_group(("exp1",))
        rewards = [traj.reward for traj in exp1_trajectories]
        
        expected_rewards = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        assert rewards == expected_rewards  # Sorted descending
        
        # Test sampling (should get from top and bottom quantiles)
        sampled_df = replay_buffer.sample_trajectories(n_per_group=4, seed=42)
        
        assert sampled_df.height == 4
        
        # With quantile_fraction=0.2 and 10 trajectories:
        # bottom 20% (indices 0-1): rewards 0.0, 0.1
        # top 20% (indices 8-9): rewards 0.8, 0.9
        sampled_rewards = sorted(sampled_df["reward"].to_list())
        
        # Should include trajectories from both ends
        assert len(set(sampled_rewards) & {0.0, 0.1}) > 0  # Some from bottom
        assert len(set(sampled_rewards) & {0.8, 0.9}) > 0  # Some from top
        
        print("✓ RewardBasedDoubleQuantileReplayBuffer tests passed!")


def test_multiple_grouping_keys():
    """Test replay buffer with multiple grouping keys."""
    print("Testing multiple grouping keys...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test trajectories with multiple metadata keys
        trajectories = [
            MockTrajectory(reward=0.8, metadata={"experiment_name": "exp1", "model": "gpt-4"}),
            MockTrajectory(reward=0.6, metadata={"experiment_name": "exp1", "model": "gpt-3.5"}),
            MockTrajectory(reward=0.9, metadata={"experiment_name": "exp2", "model": "gpt-4"}),
            MockTrajectory(reward=0.4, metadata={"experiment_name": "exp2", "model": "gpt-3.5"}),
        ]
        
        group = create_mock_trajectory_group(trajectories)
        file_path = temp_path / "test.jsonl"
        create_test_jsonl_file(file_path, [group])
        
        # Test with multiple grouping keys
        replay_buffer = TestableRewardBasedReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name", "model"]
        )
        
        # Should have 4 groups (each trajectory in its own group)
        assert replay_buffer.num_groups == 4
        
        # Test group keys
        group_keys = replay_buffer.get_group_keys()
        expected_keys = {
            ("exp1", "gpt-4"),
            ("exp1", "gpt-3.5"),
            ("exp2", "gpt-4"),
            ("exp2", "gpt-3.5")
        }
        assert set(group_keys) == expected_keys
        
        # Test filtering by multiple keys
        df = replay_buffer.dataframe
        filtered_df = replay_buffer.filter_by_group_key(experiment_name="exp1", model="gpt-4")
        
        assert filtered_df.height == 1
        assert filtered_df["reward"].to_list()[0] == 0.8
        
        print("✓ Multiple grouping keys tests passed!")


def test_file_updates():
    """Test dynamic file loading."""
    print("Testing dynamic file updates...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create initial file
        trajectories1 = [
            MockTrajectory(reward=0.8, metadata={"experiment_name": "exp1"}),
            MockTrajectory(reward=0.6, metadata={"experiment_name": "exp1"}),
        ]
        group1 = create_mock_trajectory_group(trajectories1)
        file1 = temp_path / "test1.jsonl"
        create_test_jsonl_file(file1, [group1])
        
        # Initialize replay buffer
        replay_buffer = TestableRewardBasedReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name"]
        )
        
        # Check initial state
        assert replay_buffer.num_files_loaded == 1
        assert replay_buffer.total_trajectories == 2
        
        # Add new file
        trajectories2 = [
            MockTrajectory(reward=0.9, metadata={"experiment_name": "exp2"}),
            MockTrajectory(reward=0.7, metadata={"experiment_name": "exp2"}),
        ]
        group2 = create_mock_trajectory_group(trajectories2)
        file2 = temp_path / "test2.jsonl"
        create_test_jsonl_file(file2, [group2])
        
        # Update should detect new file
        new_files = replay_buffer.update_trajectories()
        assert new_files == 1
        assert replay_buffer.num_files_loaded == 2
        assert replay_buffer.total_trajectories == 4
        assert replay_buffer.num_groups == 2
        
        # Update again should find no new files
        new_files = replay_buffer.update_trajectories()
        assert new_files == 0
        
        print("✓ Dynamic file updates tests passed!")


def test_dataframe_creation():
    """Test DataFrame creation and structure."""
    print("Testing DataFrame creation...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test trajectories with metrics
        trajectories = [
            MockTrajectory(
                reward=0.8, 
                metadata={"experiment_name": "exp1", "timestamp": "2024-01-01"},
                metrics={"accuracy": 0.85, "speed": 1.2}
            ),
            MockTrajectory(
                reward=0.6, 
                metadata={"experiment_name": "exp1", "timestamp": "2024-01-02"},
                metrics={"accuracy": 0.75, "speed": 1.5}
            ),
        ]
        
        group = create_mock_trajectory_group(trajectories)
        file_path = temp_path / "test.jsonl"
        create_test_jsonl_file(file_path, [group])
        
        replay_buffer = TestableRewardBasedReplayBuffer(
            directory=str(temp_path),
            grouping_keys=["experiment_name"]
        )
        
        # Test DataFrame structure
        df = replay_buffer.dataframe
        
        # Check basic columns
        assert "trajectory_id" in df.columns
        assert "reward" in df.columns
        assert "group_id" in df.columns
        assert "group_experiment_name" in df.columns
        
        # Check metadata columns
        assert "metadata_timestamp" in df.columns
        
        # Check metrics columns
        assert "metric_accuracy" in df.columns
        assert "metric_speed" in df.columns
        
        # Check data integrity
        assert df.height == 2
        rewards = df["reward"].to_list()
        assert set(rewards) == {0.8, 0.6}
        
        print("✓ DataFrame creation tests passed!")


def test_empty_directory():
    """Test behavior with empty directory."""
    print("Testing empty directory...")
    
    # Create temporary empty directory
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_buffer = TestableRewardBasedReplayBuffer(
            directory=str(temp_dir),
            grouping_keys=["experiment_name"]
        )
        
        # Should handle empty directory gracefully
        assert replay_buffer.num_files_loaded == 0
        assert replay_buffer.total_trajectories == 0
        assert replay_buffer.num_groups == 0
        
        # DataFrame should be empty but valid
        df = replay_buffer.dataframe
        assert df.height == 0
        assert "trajectory_id" in df.columns
        assert "reward" in df.columns
        assert "group_id" in df.columns
        
        print("✓ Empty directory tests passed!")


def run_all_tests():
    """Run all tests."""
    print("Running comprehensive replay buffer tests...\n")
    
    try:
        test_reward_based_replay_buffer()
        test_random_replay_buffer()
        test_double_quantile_replay_buffer()
        test_multiple_grouping_keys()
        test_file_updates()
        test_dataframe_creation()
        test_empty_directory()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()

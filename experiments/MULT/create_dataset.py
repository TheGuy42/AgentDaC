import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

try:
    from datasets import Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_samples: int = 1000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    output_dir: str = "datasets"
    dataset_name: str = "multiplication_dataset"
    
    def __post_init__(self):
        """Validate configuration."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        if any(split < 0 or split > 1 for split in [self.train_split, self.val_split, self.test_split]):
            raise ValueError("All splits must be between 0 and 1")


class SampleGenerator(ABC):
    """Abstract base class for sample generators."""
    
    @abstractmethod
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single sample with the given index."""
        pass


class MultiplicationGenerator(SampleGenerator):
    """Generator for multiplication problems."""
    
    def __init__(
        self,
        min_numbers: int = 2,
        max_numbers: int = 4,
        min_value: int = 1,
        max_value: int = 999,
        seed: Optional[int] = None
    ):
        """
        Initialize multiplication generator.
        
        Args:
            min_numbers: Minimum number of numbers to multiply
            max_numbers: Maximum number of numbers to multiply
            min_value: Minimum value for each number
            max_value: Maximum value for each number
            seed: Random seed for reproducibility
        """
        self.min_numbers = min_numbers
        self.max_numbers = max_numbers
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random(seed)
    
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a multiplication problem sample."""
        # Determine number of numbers to multiply
        num_count = self.rng.randint(self.min_numbers, self.max_numbers)
        
        # Generate random numbers
        numbers = [
            self.rng.randint(self.min_value, self.max_value)
            for _ in range(num_count)
        ]
        
        # Create problem string
        problem = " * ".join(map(str, numbers))
        
        # Calculate answer
        answer = 1
        for num in numbers:
            answer *= num
        
        return {
            "index": index,
            "problem": problem,
            "answer": str(answer)
        }


class AdvancedMultiplicationGenerator(SampleGenerator):
    """Advanced multiplication generator with more complex patterns."""
    
    def __init__(
        self,
        difficulty_levels: List[str] = ["easy", "medium", "hard"],
        difficulty_weights: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize advanced multiplication generator.
        
        Args:
            difficulty_levels: List of difficulty levels
            difficulty_weights: Weights for sampling difficulty levels
            seed: Random seed for reproducibility
        """
        self.difficulty_levels = difficulty_levels
        self.difficulty_weights = difficulty_weights or [1.0] * len(difficulty_levels)
        self.rng = random.Random(seed)
        
        # Define difficulty parameters
        self.difficulty_params = {
            "easy": {"min_numbers": 2, "max_numbers": 2, "min_value": 1, "max_value": 99},
            "medium": {"min_numbers": 2, "max_numbers": 4, "min_value": 10, "max_value": 999},
            "hard": {"min_numbers": 3, "max_numbers": 5, "min_value": 100, "max_value": 9999}
        }
    
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a multiplication problem with varying difficulty."""
        # Sample difficulty level
        difficulty = self.rng.choices(self.difficulty_levels, weights=self.difficulty_weights)[0]
        params = self.difficulty_params[difficulty]
        
        # Generate numbers based on difficulty
        num_count = self.rng.randint(params["min_numbers"], params["max_numbers"])
        numbers = [
            self.rng.randint(params["min_value"], params["max_value"])
            for _ in range(num_count)
        ]
        
        # Create problem string
        problem = " * ".join(map(str, numbers))
        
        # Calculate answer
        answer = 1
        for num in numbers:
            answer *= num
        
        return {
            "index": index,
            "problem": problem,
            "answer": str(answer),
            "difficulty": difficulty
        }


class AdvancedMultiplicationGenerator(SampleGenerator):
    """Advanced multiplication generator with more complex patterns."""
    
    def __init__(
        self,
        difficulty_levels: List[str] = ["easy", "medium", "hard"],
        difficulty_weights: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize advanced multiplication generator.
        
        Args:
            difficulty_levels: List of difficulty levels
            difficulty_weights: Weights for sampling difficulty levels
            seed: Random seed for reproducibility
        """
        self.difficulty_levels = difficulty_levels
        self.difficulty_weights = difficulty_weights or [1.0] * len(difficulty_levels)
        self.rng = random.Random(seed)
        
        # Define difficulty parameters
        self.difficulty_params = {
            "easy": {"min_numbers": 2, "max_numbers": 2, "min_value": 1, "max_value": 99},
            "medium": {"min_numbers": 2, "max_numbers": 4, "min_value": 10, "max_value": 999},
            "hard": {"min_numbers": 3, "max_numbers": 5, "min_value": 100, "max_value": 9999}
        }
    
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a multiplication problem with varying difficulty."""
        # Sample difficulty level
        difficulty = self.rng.choices(self.difficulty_levels, weights=self.difficulty_weights)[0]
        params = self.difficulty_params[difficulty]
        
        # Generate numbers based on difficulty
        num_count = self.rng.randint(params["min_numbers"], params["max_numbers"])
        numbers = [
            self.rng.randint(params["min_value"], params["max_value"])
            for _ in range(num_count)
        ]
        
        # Create problem string
        problem = " * ".join(map(str, numbers))
        
        # Calculate answer
        answer = 1
        for num in numbers:
            answer *= num
        
        return {
            "index": index,
            "problem": problem,
            "answer": str(answer),
            "difficulty": difficulty
        }


class TextDatasetGenerator:
    """Main class for generating text datasets with multiplication problems."""
    
    def __init__(self, config: DatasetConfig, generator: SampleGenerator):
        """
        Initialize dataset generator.
        
        Args:
            config: Dataset configuration
            generator: Sample generator instance
        """
        self.config = config
        self.generator = generator
        self.samples: List[Dict[str, Any]] = []
        
        # Set random seed for reproducibility
        random.seed(config.seed)
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the complete dataset."""
        print(f"Generating {self.config.num_samples} samples...")
        
        self.samples = []
        for i in range(self.config.num_samples):
            sample = self.generator.generate_sample(i)
            self.samples.append(sample)
            
            # if (i + 1) % 100 == 0:
            #     print(f"Generated {i + 1}/{self.config.num_samples} samples")
        
        print("Dataset generation complete!")
        return self.samples
    
    def split_dataset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train, validation, and test sets."""
        if not self.samples:
            raise ValueError("No samples generated. Call generate_dataset() first.")
        
        # Shuffle samples for random split
        shuffled_samples = self.samples.copy()
        random.shuffle(shuffled_samples)
        
        # Calculate split indices
        total_samples = len(shuffled_samples)
        train_end = int(total_samples * self.config.train_split)
        val_end = train_end + int(total_samples * self.config.val_split)
        
        # Split the data
        train_data = shuffled_samples[:train_end]
        val_data = shuffled_samples[train_end:val_end]
        test_data = shuffled_samples[val_end:]
        
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Save dataset to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # random permutation
        random.shuffle(data)
        
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} samples to {filepath}")
        return str(filepath)
    
    def save_split_datasets(self) -> Dict[str, str]:
        """Generate and save train/val/test splits."""
        train_data, val_data, test_data = self.split_dataset()
        
        filepaths = {}
        
        # Save each split
        if train_data:
            filepaths['train'] = self.save_dataset(
                train_data, f"{self.config.dataset_name}_train.json"
            )
        
        if val_data:
            filepaths['val'] = self.save_dataset(
                val_data, f"{self.config.dataset_name}_val.json"
            )
        
        if test_data:
            filepaths['test'] = self.save_dataset(
                test_data, f"{self.config.dataset_name}_test.json"
            )
        
        return filepaths
    
    def save_full_dataset(self) -> str:
        """Save the complete dataset."""
        if not self.samples:
            raise ValueError("No samples generated. Call generate_dataset() first.")
        
        return self.save_dataset(self.samples, f"{self.config.dataset_name}_full.json")
    
    def get_sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated dataset."""
        if not self.samples:
            return {}
        
        # Basic statistics
        stats = {
            "total_samples": len(self.samples),
            "sample_fields": list(self.samples[0].keys()) if self.samples else []
        }
        
        # Problem length statistics
        problem_lengths = [len(sample["problem"]) for sample in self.samples]
        stats["problem_length"] = {
            "min": min(problem_lengths),
            "max": max(problem_lengths),
            "avg": sum(problem_lengths) / len(problem_lengths)
        }
        
        # Answer length statistics
        answer_lengths = [len(sample["answer"]) for sample in self.samples]
        stats["answer_length"] = {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "avg": sum(answer_lengths) / len(answer_lengths)
        }
        
        # Check for difficulty field (if using AdvancedMultiplicationGenerator)
        if self.samples and "difficulty" in self.samples[0]:
            difficulty_counts = {}
            for sample in self.samples:
                diff = sample["difficulty"]
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            stats["difficulty_distribution"] = difficulty_counts
        
        return stats
    
    def print_sample_examples(self, num_examples: int = 5):
        """Print some example samples from the dataset."""
        if not self.samples:
            print("No samples generated yet.")
            return
        
        print(f"\nExample samples (showing first {min(num_examples, len(self.samples))}):")
        print("-" * 80)
        
        for i in range(min(num_examples, len(self.samples))):
            sample = self.samples[i]
            print(f"Sample {i}:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
            print()


def load_dataset_as_datasetdict(
    dataset_dir: str,
    dataset_name: str,
    splits: Optional[List[str]] = None
) -> "DatasetDict":
    """
    Load generated JSON datasets as a HuggingFace DatasetDict object.
    
    Args:
        dataset_dir: Directory containing the dataset files
        dataset_name: Name of the dataset (used to construct filenames)
        splits: List of splits to load (default: ["train", "val", "test"])
    
    Returns:
        DatasetDict containing the loaded datasets
    
    Raises:
        ImportError: If datasets library is not available
        FileNotFoundError: If dataset files are not found
        ValueError: If no valid splits are found
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library is required for this function. "
            "Install with: pip install datasets"
        )
    
    if splits is None:
        splits = ["train", "val", "test"]
    
    dataset_dir_path = Path(dataset_dir)
    dataset_dict = {}
    
    # Load each split
    for split in splits:
        json_file = dataset_dir_path / f"{dataset_name}_{split}.json"
        
        if json_file.exists():
            print(f"Loading {split} split from {json_file}")
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to HuggingFace Dataset
            dataset_dict[split] = Dataset.from_list(data)
            print(f"Loaded {len(data)} samples for {split} split")
        else:
            print(f"Warning: {split} split file not found at {json_file}")
    
    if not dataset_dict:
        raise ValueError(f"No valid dataset files found in {dataset_dir} for dataset '{dataset_name}'")
    
    # Create DatasetDict
    dataset_dict_obj = DatasetDict(dataset_dict)
    
    print(f"\nDatasetDict created with splits: {list(dataset_dict_obj.keys())}")
    return dataset_dict_obj


def load_full_dataset_as_dataset(
    dataset_file: str
) -> "Dataset":
    """
    Load a single dataset JSON file as a HuggingFace Dataset object.
    
    Args:
        dataset_file: Path to the JSON dataset file
    
    Returns:
        Dataset object containing the loaded data
    
    Raises:
        ImportError: If datasets library is not available
        FileNotFoundError: If dataset file is not found
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library is required for this function. "
            "Install with: pip install datasets"
        )
    
    dataset_path = Path(dataset_file)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    print(f"Loading dataset from {dataset_file}")
    
    # Load JSON data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"Loaded {len(data)} samples")
    
    return dataset


def main():
    """Example usage of the dataset generator."""
    # # Example 1: Basic multiplication generator
    # print("=== Basic Multiplication Dataset ===")
    # config1 = DatasetConfig(
    #     num_samples=100,
    #     dataset_name="basic_multiplication",
    #     output_dir="datasets/basic"
    # )
    
    # generator1 = MultiplicationGenerator(
    #     min_numbers=2,
    #     max_numbers=4,
    #     min_value=1,
    #     max_value=999,
    #     seed=42
    # )
    
    # dataset1 = TextDatasetGenerator(config1, generator1)
    # dataset1.generate_dataset()
    # dataset1.print_sample_examples()
    
    # # Save splits
    # filepaths1 = dataset1.save_split_datasets()
    # print("Saved files:", filepaths1)
    
    # # Print statistics
    # stats1 = dataset1.get_sample_statistics()
    # print("Dataset statistics:", json.dumps(stats1, indent=2))
    
    # print("\n" + "="*80 + "\n")
    
    # Example 2: Advanced multiplication generator with difficulty levels
    print("=== Advanced Multiplication Dataset ===")
    config2 = DatasetConfig(
        num_samples=3000,
        dataset_name="advanced_multiplication",
        output_dir="datasets/advanced_v2",
        train_split=0.7,
        val_split=0.2,
    )
    
    generator2 = AdvancedMultiplicationGenerator(
        difficulty_levels=["easy", "medium", "hard"],
        difficulty_weights=[0.2, 0.4, 0.4],  # More medium difficulty
        seed=42
    )
    
    dataset2 = TextDatasetGenerator(config2, generator2)
    dataset2.generate_dataset()
    dataset2.print_sample_examples()
    
    # Save splits
    filepaths2 = dataset2.save_split_datasets()
    print("Saved files:", filepaths2)
    
    # Print statistics
    stats2 = dataset2.get_sample_statistics()
    print("Dataset statistics:", json.dumps(stats2, indent=2))
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Loading datasets as DatasetDict (if datasets library is available)
    if DATASETS_AVAILABLE:
        print("=== Loading Dataset as DatasetDict ===")
        try:
            # Load the basic multiplication dataset
            dataset_dict = load_dataset_as_datasetdict(
                dataset_dir="datasets/basic",
                dataset_name="basic_multiplication"
            )
            
            print(f"DatasetDict keys: {list(dataset_dict.keys())}")
            if 'train' in dataset_dict:
                print(f"Train dataset: {dataset_dict['train']}")
                print(f"First sample: {dataset_dict['train'][0]}")
            
            # Example of loading a single dataset file
            full_dataset = load_full_dataset_as_dataset(
                "datasets/basic/basic_multiplication_train.json"
            )
            print(f"Full dataset: {full_dataset}")
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
    else:
        print("=== Datasets library not available ===")
        print("To use DatasetDict loading functionality, install the datasets library:")
        print("pip install datasets")


if __name__ == "__main__":
    main()

import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field

from create_dataset import SampleGenerator, Sample

#######################################
############# Definitions #############
#######################################


class Argument(BaseModel):
    argument: str
    value: bool

class ArgumentGenerator(ABC):
    """Abstract base class for argument generators."""
    
    @abstractmethod
    def generate_argument(self) -> Argument:
        pass

class ArgumentCombinator(ArgumentGenerator, ABC):
    """Abstract base class for generating and combining multiple arguments."""
    def __init__(self, n_args, generators: List[ArgumentGenerator]):
        self.n_args = n_args
        self.generators = generators
        self.rng = random.Random()
        
    def generate_arguments(self) -> List[Argument]:
        return [self.rng.choice(self.generators).generate_argument() for _ in range(self.n_args)]
    
    @abstractmethod
    def combine_arguments(self, arguments: List[Argument]) -> Argument:
        pass

    def generate_argument(self) -> Argument:
        args = self.generate_arguments()
        return self.combine_arguments(args)


##########################################
############# Implementation #############
##########################################

class SimpleCombinator(ArgumentCombinator):
    """Combines arguments using logical AND or OR."""
    def __init__(
            self,
            n_args: int,
            generators: List[ArgumentGenerator],
            method: str|None = None,
            not_ratio: float = 0.0
            ):
        super().__init__(n_args, generators)
        if method not in ["AND", "OR", None]:
            raise ValueError("Method must be 'AND' or 'OR'")
        self.method = method if method is not None else self.rng.choice(["AND", "OR"])
        self.not_ratio = not_ratio
    
    def combine_arguments(self, arguments: List[Argument]) -> Argument:
        # Apply NOT to arguments based on not_ratio
        processed_args = []
        for arg in arguments:
            if self.rng.random() < self.not_ratio:
                # Apply NOT to this argument
                negated_expr = f"NOT({arg.argument})"
                negated_value = not arg.value
                processed_args.append(Argument(argument=negated_expr, value=negated_value))
            else:
                processed_args.append(arg)
        
        combined_expr = f" {self.method} ".join([arg.argument for arg in processed_args])
        combined_expr = f"({combined_expr})"
        if self.method == "AND":
            combined_value = all(arg.value for arg in processed_args)
        else:  # OR
            combined_value = any(arg.value for arg in processed_args)
        
        return Argument(argument=combined_expr, value=combined_value)

class ThresholdArgumentGenerator(ArgumentGenerator):
    """Generates a boolean argument based on a threshold."""
    def __init__(
            self,
            min_value: int = 1,
            max_value: int = 100,
            seed: Optional[int] = None
            ):
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random(seed)

    def generate_argument(self) -> Argument:
        num_1 = self.rng.randint(self.min_value, self.max_value)
        num_2 = self.rng.randint(self.min_value, self.max_value)
        operation = self.rng.choice(["+", "-", "*", "/"])
        exp = str(num_1) + operation + str(num_2)
        solution = 0
        if operation == "+":
            solution = num_1 + num_2
        elif operation == "-":
            solution = num_1 - num_2
        elif operation == "*":
            solution = num_1 * num_2
        elif operation == "/":
            solution = num_1 / num_2
        
        var = abs(solution) // 2 if abs(solution) > 2 else 1
        threshold = self.rng.randint(int(solution - var), int(solution + var))

        ineq = self.rng.choice([">", "<"])
        if ineq == ">":
            argument = f"{exp} {ineq} {threshold}"
            value = True if solution > threshold else False
        else:
            argument = f"{exp} {ineq} {threshold}"
            value = True if solution < threshold else False

        return Argument(argument=argument, value=value)

class ComparisonArgumentGenerator(ArgumentGenerator):
    """Generates boolean arguments by comparing two expressions."""
    def __init__(
            self,
            min_value: int = 1,
            max_value: int = 50,
            seed: Optional[int] = None
            ):
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random(seed)

    def generate_argument(self) -> Argument:
        # Generate two arithmetic expressions
        left_num1 = self.rng.randint(self.min_value, self.max_value)
        left_num2 = self.rng.randint(self.min_value, self.max_value)
        left_op = self.rng.choice(["+", "-", "*"])
        
        right_num1 = self.rng.randint(self.min_value, self.max_value)
        right_num2 = self.rng.randint(self.min_value, self.max_value)
        right_op = self.rng.choice(["+", "-", "*"])
        
        left_expr = f"{left_num1} {left_op} {left_num2}"
        right_expr = f"{right_num1} {right_op} {right_num2}"
        
        # Calculate values
        left_val = self._eval_expression(left_num1, left_op, left_num2)
        right_val = self._eval_expression(right_num1, right_op, right_num2)
        
        # Choose comparison operator
        comparison = self.rng.choice(["==", "!=", ">", "<", ">=", "<="])
        
        argument = f"({left_expr}) {comparison} ({right_expr})"
        value = self._eval_comparison(left_val, comparison, right_val)
        
        return Argument(argument=argument, value=value)
    
    def _eval_expression(self, num1: int, op: str, num2: int) -> float:
        if op == "+":
            return num1 + num2
        elif op == "-":
            return num1 - num2
        elif op == "*":
            return num1 * num2
        return 0
    
    def _eval_comparison(self, left: float, op: str, right: float) -> bool:
        if op == "==":
            return left == right
        elif op == "!=":
            return left != right
        elif op == ">":
            return left > right
        elif op == "<":
            return left < right
        elif op == ">=":
            return left >= right
        elif op == "<=":
            return left <= right
        return False

class ModuloArgumentGenerator(ArgumentGenerator):
    """Generates boolean arguments based on modulo operations."""
    def __init__(
            self,
            min_value: int = 1,
            max_value: int = 100,
            max_divisor: int = 10,
            seed: Optional[int] = None
            ):
        self.min_value = min_value
        self.max_value = max_value
        self.max_divisor = max_divisor
        self.rng = random.Random(seed)

    def generate_argument(self) -> Argument:
        num = self.rng.randint(self.min_value, self.max_value)
        divisor = self.rng.randint(2, self.max_divisor)
        remainder = self.rng.randint(0, divisor - 1)
        
        argument = f"{num} % {divisor} == {remainder}"
        value = (num % divisor) == remainder
        
        return Argument(argument=argument, value=value)

class PrimeArgumentGenerator(ArgumentGenerator):
    """Generates boolean arguments about prime numbers."""
    def __init__(
            self,
            min_value: int = 2,
            max_value: int = 100,
            seed: Optional[int] = None
            ):
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random(seed)

    def generate_argument(self) -> Argument:
        num = self.rng.randint(self.min_value, self.max_value)
        
        argument = f"is_prime({num})"
        value = self._is_prime(num)
        
        return Argument(argument=argument, value=value)
    
    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

class RangeArgumentGenerator(ArgumentGenerator):
    """Generates boolean arguments about number ranges."""
    def __init__(
            self,
            min_value: int = 1,
            max_value: int = 100,
            seed: Optional[int] = None
            ):
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random(seed)

    def generate_argument(self) -> Argument:
        num = self.rng.randint(self.min_value, self.max_value)
        
        # Generate a range
        range_start = self.rng.randint(self.min_value, self.max_value - 10)
        range_end = self.rng.randint(range_start + 5, self.max_value)
        
        argument = f"{range_start} <= {num} <= {range_end}"
        value = range_start <= num <= range_end
        
        return Argument(argument=argument, value=value)
        
class DifficultyArgument(BaseModel):
    difficulty: str
    min_depth: int
    max_depth: int
    min_args: int
    max_args: int
    leaf_generators: List[ArgumentGenerator]
    combinators: List[ArgumentCombinator]  # List of combinator instances
    
    class Config:
        arbitrary_types_allowed = True

class BBEHGenerator(SampleGenerator):
    """Boolean Expression Evaluation with Hierarchy generator."""
    
    def __init__(
        self,
        root_config: DifficultyArgument,
        inner_config: Optional[DifficultyArgument] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize BBEH generator.
        
        Args:
            root_config: Configuration for the root node of the expression tree
            inner_config: Configuration for inner nodes. If None, root_config is used for all nodes
            seed: Random seed for reproducibility
        """
        self.root_config = root_config
        self.inner_config = inner_config if inner_config is not None else root_config
        self.rng = random.Random(seed)
    
    def _create_generator_tree(self, depth: int, is_root: bool = False) -> ArgumentGenerator:
        """Recursively create a tree of generators."""
        # Choose config based on whether this is root or inner node
        config = self.root_config if is_root else self.inner_config
        
        if depth <= 0:
            # Base case: return a leaf generator
            return self.rng.choice(config.leaf_generators)
        
        # Create a combinator
        combinator = self.rng.choice(config.combinators)
        n_args = self.rng.randint(config.min_args, config.max_args)
        
        # Create child generators
        child_generators = []
        for _ in range(n_args):
            child_depth = depth - 1
            child_generator = self._create_generator_tree(child_depth, is_root=False)
            child_generators.append(child_generator)
        
        # Create a new combinator instance with the child generators
        # Handle different combinator types
        if isinstance(combinator, SimpleCombinator):
            new_combinator = SimpleCombinator(
                n_args=n_args,
                generators=child_generators,
                method=combinator.method,
                not_ratio=combinator.not_ratio
            )
        else:
            # For other combinator types, create a basic copy
            new_combinator = type(combinator)(n_args, child_generators)
        
        new_combinator.rng = self.rng  # Use the same random state
        
        return new_combinator
    
    def _create_argument_tree(self, depth: int, is_root: bool = True) -> Argument:
        """Create a complex argument tree by generating a generator tree and then creating an argument."""
        generator = self._create_generator_tree(depth, is_root)
        return generator.generate_argument()
    
    def generate_sample(self, index: int) -> Sample:
        """Generate a boolean expression evaluation sample."""
        # Generate the expression tree
        max_depth = self.rng.randint(self.root_config.min_depth, self.root_config.max_depth)
        main_argument = self._create_argument_tree(max_depth, is_root=True)
        
        # Create the problem statement
        problem = f"{main_argument.argument}"
        
        # The answer is the boolean value
        answer = str(main_argument.value).lower()  # "true" or "false"
        
        return Sample(
            index=index,
            problem=problem,
            answer=answer,
            metadata={
                "difficulty": self.root_config.difficulty,
                # "expression": main_argument.argument,
                "max_depth": max_depth,
                "actual_value": main_argument.value
            }
        )


def create_example_difficulty_configs() -> Tuple[DifficultyArgument, DifficultyArgument, DifficultyArgument]:
    """Create example difficulty configurations for testing."""
    
    # Create leaf generators
    threshold_gen = ThresholdArgumentGenerator(min_value=1, max_value=50, seed=42)
    comparison_gen = ComparisonArgumentGenerator(min_value=1, max_value=30, seed=42)
    modulo_gen = ModuloArgumentGenerator(min_value=1, max_value=50, max_divisor=8, seed=42)
    prime_gen = PrimeArgumentGenerator(min_value=2, max_value=50, seed=42)
    range_gen = RangeArgumentGenerator(min_value=1, max_value=50, seed=42)
    
    # Create combinator instances with different parameters
    # simple_and_combinator = SimpleCombinator(2, [], "AND", not_ratio=0.1)
    simple_combinator = SimpleCombinator(2, [], not_ratio=0.15)
    complex_combinator = SimpleCombinator(3, [], not_ratio=0.35)
    
    # Easy difficulty
    easy_config = DifficultyArgument(
        difficulty="easy",
        min_depth=1,
        max_depth=2,
        min_args=2,
        max_args=4,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen],
        combinators=[simple_combinator]
    )
    
    # Medium difficulty
    medium_config = DifficultyArgument(
        difficulty="medium",
        min_depth=2,
        max_depth=3,
        min_args=2,
        max_args=4,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen, prime_gen, range_gen],
        combinators=[simple_combinator, complex_combinator]
    )
    
    # Hard difficulty
    hard_config = DifficultyArgument(
        difficulty="hard",
        min_depth=2,
        max_depth=3,
        min_args=4,
        max_args=7,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen, prime_gen, range_gen],
        combinators=[complex_combinator, simple_combinator]
    )
    
    return easy_config, medium_config, hard_config


def main():
    """Example usage of the BBEH generator."""
    print("=== Boolean Expression Evaluation with Hierarchy (BBEH) Dataset ===")
    
    # Create difficulty configurations
    easy_config, medium_config, hard_config = create_example_difficulty_configs()
    
    # # Example 1: Using medium config for both root and inner nodes
    # print("\n--- Example 1: Medium config for all nodes ---")
    # generator1 = BBEHGenerator(
    #     root_config=medium_config,
    #     inner_config=None,  # Will use medium_config for all nodes
    #     seed=42
    # )
    
    # # Generate some example samples
    # print("\nGenerating example samples:")
    # print("-" * 80)
    
    # for i in range(3):
    #     sample = generator1.generate_sample(i)
    #     print(f"Sample {i}:")
    #     print(f"  Problem: {sample.problem}")
    #     print(f"  Answer: {sample.answer}")
    #     print(f"  Difficulty: {sample.metadata['difficulty']}")
    #     print(f"  Max Depth: {sample.metadata['max_depth']}")
    #     print()


    
    # Example 2: Using hard config for root, easy config for inner nodes
    print("\n--- Example 2: Hard root config, Easy inner config ---")
    inner_config = hard_config.model_copy()
    inner_config.min_args, inner_config.max_args = hard_config.min_args // 2, hard_config.max_args // 2

    generator2 = BBEHGenerator(
        root_config=hard_config,
        inner_config=inner_config,
        seed=42
    )
    
    for i in range(2):
        sample = generator2.generate_sample(i + 10)
        print(f"Sample {i + 10}:")
        print(f"  Problem: {sample.problem}")
        print(f"  Answer: {sample.answer}")
        print(f"  Difficulty: {sample.metadata['difficulty']}")
        print(f"  Max Depth: {sample.metadata['max_depth']}")
        print()
    
    # Create dataset configuration
    from create_dataset import DatasetConfig, TextDatasetGenerator
    
    config = DatasetConfig(
        num_samples=3000,
        dataset_name="bbeh_boolean_expressions",
        output_dir="datasets/bbeh_v2",
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42
    )
    
    # Generate full dataset using the first generator
    dataset_generator = TextDatasetGenerator(config, generator2)
    ds = dataset_generator.generate_dataset()
    dataset_generator.print_sample_examples(3)

    hist = {"true": 0, "false": 0}
    for sample in ds:
        hist[sample['answer']] += 1
    print("Answer distribution:", hist)
    
    # # Save splits
    filepaths = dataset_generator.save_split_datasets()
    print("Saved files:", filepaths)
    
    # Print statistics
    stats = dataset_generator.get_sample_statistics()
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
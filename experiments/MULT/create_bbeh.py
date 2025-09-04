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
            method: str = "AND"
            ):
        super().__init__(n_args, generators)
        if method not in ["AND", "OR"]:
            raise ValueError("Method must be 'AND' or 'OR'")
        self.method = method
    
    def combine_arguments(self, arguments: List[Argument]) -> Argument:
        combined_expr = f" {self.method} ".join([arg.argument for arg in arguments])
        combined_expr = f"({combined_expr})"
        if self.method == "AND":
            combined_value = all(arg.value for arg in arguments)
        else:  # OR
            combined_value = any(arg.value for arg in arguments)
        
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
    combinators: List[type[ArgumentCombinator]]  # List of combinator classes
    
    class Config:
        arbitrary_types_allowed = True

class BBEHGenerator(SampleGenerator):
    """Boolean Expression Evaluation with Hierarchy generator."""
    
    def __init__(
        self,
        difficulty_args: List[DifficultyArgument],
        difficulty_weights: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize BBEH generator.
        
        Args:
            difficulty_args: List of difficulty argument configurations
            difficulty_weights: Weights for sampling difficulty levels
            seed: Random seed for reproducibility
        """
        self.difficulty_args = difficulty_args
        self.difficulty_weights = difficulty_weights or [1.0] * len(difficulty_args)
        self.rng = random.Random(seed)
    
    def _create_argument_tree(self, depth: int, difficulty_arg: DifficultyArgument) -> Argument:
        """Recursively create a complex argument tree."""
        if depth <= 0:
            # Base case: use a leaf generator
            generator = self.rng.choice(difficulty_arg.leaf_generators)
            return generator.generate_argument()
        
        # Choose between creating a combinator or a leaf node
        # if depth == 1 or self.rng.random() < 0.3:  # 30% chance for leaf at any depth
        #     generator = self.rng.choice(difficulty_arg.leaf_generators)
        #     return generator.generate_argument()
        
        # Create a combinator
        combinator_class = self.rng.choice(difficulty_arg.combinators)
        n_args = self.rng.randint(difficulty_arg.min_args, difficulty_arg.max_args)
        
        # Create child arguments
        child_generators = []
        for _ in range(n_args):
            child_depth = depth - 1
            child_arg = self._create_argument_tree(child_depth, difficulty_arg)
            
            # Create a temporary generator that returns this specific argument
            class TempGenerator(ArgumentGenerator):
                def __init__(self, arg):
                    self.arg = arg
                def generate_argument(self):
                    return self.arg
            
            child_generators.append(TempGenerator(child_arg))
        
        # Create the combinator with the child generators
        combinator_class = self.rng.choice(difficulty_arg.combinators)
        
        # Create the combinator instance
        method = self.rng.choice(["AND", "OR"])
        combinator = combinator_class(n_args, child_generators, method) # only type currently is SimpleCombinator
        
        return combinator.generate_argument()
    
    def generate_sample(self, index: int) -> Sample:
        """Generate a boolean expression evaluation sample."""
        # Sample difficulty level
        difficulty_arg = self.rng.choices(
            self.difficulty_args, 
            weights=self.difficulty_weights
        )[0]
        
        # Generate the expression tree
        max_depth = self.rng.randint(difficulty_arg.min_depth, difficulty_arg.max_depth)
        main_argument = self._create_argument_tree(max_depth, difficulty_arg)
        
        # Create the problem statement
        problem = f"{main_argument.argument}"
        
        # The answer is the boolean value
        answer = str(main_argument.value).lower()  # "true" or "false"
        
        return Sample(
            index=index,
            problem=problem,
            answer=answer,
            metadata={
                "difficulty": difficulty_arg.difficulty,
                "expression": main_argument.argument,
                "max_depth": max_depth,
                "actual_value": main_argument.value
            }
        )


def create_example_difficulty_configs() -> List[DifficultyArgument]:
    """Create example difficulty configurations for testing."""
    
    # Create leaf generators
    threshold_gen = ThresholdArgumentGenerator(min_value=1, max_value=50, seed=42)
    comparison_gen = ComparisonArgumentGenerator(min_value=1, max_value=30, seed=42)
    modulo_gen = ModuloArgumentGenerator(min_value=1, max_value=50, max_divisor=8, seed=42)
    prime_gen = PrimeArgumentGenerator(min_value=2, max_value=50, seed=42)
    range_gen = RangeArgumentGenerator(min_value=1, max_value=50, seed=42)
    
    # Easy difficulty
    easy_config = DifficultyArgument(
        difficulty="easy",
        min_depth=1,
        max_depth=2,
        min_args=2,
        max_args=4,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen],
        combinators=[SimpleCombinator]
    )
    
    # Medium difficulty
    medium_config = DifficultyArgument(
        difficulty="medium",
        min_depth=2,
        max_depth=3,
        min_args=2,
        max_args=4,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen, prime_gen, range_gen],
        combinators=[SimpleCombinator]
    )
    
    # Hard difficulty
    hard_config = DifficultyArgument(
        difficulty="hard",
        min_depth=2,
        max_depth=3,
        min_args=3,
        max_args=5,
        leaf_generators=[threshold_gen, comparison_gen, modulo_gen, prime_gen, range_gen],
        combinators=[SimpleCombinator]
    )
    
    return [easy_config, medium_config, hard_config]


def main():
    """Example usage of the BBEH generator."""
    print("=== Boolean Expression Evaluation with Hierarchy (BBEH) Dataset ===")
    
    # Create difficulty configurations
    difficulty_configs = create_example_difficulty_configs()
    
    # Create BBEH generator
    generator = BBEHGenerator(
        difficulty_args=difficulty_configs,
        difficulty_weights=[0.3, 0.4, 0.3],  # Balanced distribution
        seed=42
    )
    
    # Generate some example samples
    print("\nGenerating example samples:")
    print("-" * 80)
    
    for i in range(5):
        sample = generator.generate_sample(i)
        print(f"Sample {i}:")
        print(f"  Problem: {sample.problem}")
        print(f"  Answer: {sample.answer}")
        print(f"  Difficulty: {sample.metadata['difficulty']}")
        print(f"  Expression: {sample.metadata['expression']}")
        print(f"  Max Depth: {sample.metadata['max_depth']}")
        print()
    
    # Create dataset configuration
    from create_dataset import DatasetConfig, TextDatasetGenerator
    
    config = DatasetConfig(
        num_samples=1000,
        dataset_name="bbeh_boolean_expressions",
        output_dir="datasets/bbeh",
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42
    )
    
    # Generate full dataset
    dataset_generator = TextDatasetGenerator(config, generator)
    dataset_generator.generate_dataset()
    dataset_generator.print_sample_examples(3)
    
    # Save splits
    filepaths = dataset_generator.save_split_datasets()
    print("Saved files:", filepaths)
    
    # Print statistics
    stats = dataset_generator.get_sample_statistics()
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script to demonstrate the new BBEH features:
1. NOT operator with configurable ratio
2. Combinator instances with different parameters 
3. Separate root and inner node configurations
"""

from create_bbeh import *

def test_not_functionality():
    """Test the NOT functionality with different ratios."""
    print("=== Testing NOT Functionality ===")
    
    # Create generators
    threshold_gen = ThresholdArgumentGenerator(min_value=1, max_value=10, seed=42)
    
    # Test different NOT ratios
    for not_ratio in [0.0, 0.5, 1.0]:
        print(f"\nTesting NOT ratio: {not_ratio}")
        combinator = SimpleCombinator(3, [threshold_gen, threshold_gen, threshold_gen], "OR", not_ratio=not_ratio)
        
        for i in range(3):
            result = combinator.generate_argument()
            print(f"  Expression: {result.argument}")
            print(f"  Value: {result.value}")

def test_combinator_configurations():
    """Test different combinator configurations."""
    print("\n=== Testing Combinator Configurations ===")
    
    # Create generators
    threshold_gen = ThresholdArgumentGenerator(min_value=1, max_value=20, seed=42)
    comparison_gen = ComparisonArgumentGenerator(min_value=1, max_value=15, seed=42)
    
    # Create different combinator instances
    combinators = [
        SimpleCombinator(2, [], "AND", not_ratio=0.0),  # No NOT
        SimpleCombinator(2, [], "OR", not_ratio=0.3),   # 30% NOT chance
        SimpleCombinator(3, [], "AND", not_ratio=0.6),  # 60% NOT chance
    ]
    
    config = DifficultyArgument(
        difficulty="test",
        min_depth=1,
        max_depth=2,
        min_args=2,
        max_args=3,
        leaf_generators=[threshold_gen, comparison_gen],
        combinators=combinators
    )
    
    generator = BBEHGenerator(root_config=config, seed=42)
    
    print("\nGenerated expressions with different combinator configs:")
    for i in range(5):
        sample = generator.generate_sample(i)
        print(f"  {sample.problem} -> {sample.answer}")

def test_root_vs_inner_configs():
    """Test different configurations for root vs inner nodes."""
    print("\n=== Testing Root vs Inner Configurations ===")
    
    # Create generators
    simple_gen = ThresholdArgumentGenerator(min_value=1, max_value=5, seed=42)
    complex_gen = ComparisonArgumentGenerator(min_value=10, max_value=20, seed=42)
    
    # Simple config for inner nodes (smaller numbers, no NOT)
    inner_config = DifficultyArgument(
        difficulty="simple_inner",
        min_depth=1,
        max_depth=1,
        min_args=2,
        max_args=2,
        leaf_generators=[simple_gen],
        combinators=[SimpleCombinator(2, [], "AND", not_ratio=0.0)]
    )
    
    # Complex config for root (larger numbers, high NOT ratio)
    root_config = DifficultyArgument(
        difficulty="complex_root",
        min_depth=2,
        max_depth=3,
        min_args=3,
        max_args=4,
        leaf_generators=[complex_gen],
        combinators=[SimpleCombinator(3, [], "OR", not_ratio=0.7)]
    )
    
    generator = BBEHGenerator(
        root_config=root_config,
        inner_config=inner_config,
        seed=42
    )
    
    print("\nExpressions with complex root, simple inner nodes:")
    for i in range(3):
        sample = generator.generate_sample(i)
        print(f"  {sample.problem}")
        print(f"    -> {sample.answer}")
        print()

if __name__ == "__main__":
    test_not_functionality()
    test_combinator_configurations()
    test_root_vs_inner_configs()

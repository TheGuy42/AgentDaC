#!/usr/bin/env python3
"""
Test script to demonstrate the separation of generator tree creation and argument generation.
"""

from create_bbeh import *

def test_generator_tree_separation():
    """Test that _create_generator_tree and _create_argument_tree work correctly."""
    print("=== Testing Generator Tree Separation ===")
    
    # Create a simple configuration
    threshold_gen = ThresholdArgumentGenerator(min_value=1, max_value=10, seed=42)
    comparison_gen = ComparisonArgumentGenerator(min_value=1, max_value=10, seed=42)
    
    simple_combinator = SimpleCombinator(2, [], "AND", not_ratio=0.3)
    
    config = DifficultyArgument(
        difficulty="test",
        min_depth=1,
        max_depth=2,
        min_args=2,
        max_args=3,
        leaf_generators=[threshold_gen, comparison_gen],
        combinators=[simple_combinator]
    )
    
    generator = BBEHGenerator(root_config=config, seed=42)
    
    print("\n--- Testing _create_generator_tree ---")
    
    # Test creating generator trees
    for depth in [0, 1, 2]:
        print(f"\nDepth {depth}:")
        gen_tree = generator._create_generator_tree(depth, is_root=True)
        print(f"  Generator type: {type(gen_tree).__name__}")
        
        if hasattr(gen_tree, 'generators'):
            print(f"  Number of child generators: {len(gen_tree.generators)}")
        
        # Generate multiple arguments to see variation
        print("  Generated arguments:")
        for i in range(3):
            arg = gen_tree.generate_argument()
            print(f"    {arg.argument} -> {arg.value}")
    
    print("\n--- Testing _create_argument_tree ---")
    
    # Test creating argument trees directly
    for depth in [0, 1, 2]:
        print(f"\nDepth {depth}:")
        print("  Generated arguments:")
        for i in range(3):
            arg = generator._create_argument_tree(depth, is_root=True)
            print(f"    {arg.argument} -> {arg.value}")
    
    print("\n--- Comparing Results ---")
    print("Both methods should produce similar complexity and structure:")
    
    # Create a generator tree and use it multiple times
    gen_tree = generator._create_generator_tree(2, is_root=True)
    print(f"\nUsing same generator tree (type: {type(gen_tree).__name__}):")
    for i in range(3):
        arg = gen_tree.generate_argument()
        print(f"  {arg.argument} -> {arg.value}")
    
    # Create argument trees directly
    print(f"\nCreating new argument trees each time:")
    for i in range(3):
        arg = generator._create_argument_tree(2, is_root=True)
        print(f"  {arg.argument} -> {arg.value}")

if __name__ == "__main__":
    test_generator_tree_separation()

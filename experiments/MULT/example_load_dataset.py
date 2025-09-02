#!/usr/bin/env python3
"""
Example script demonstrating how to load the generated multiplication datasets
as HuggingFace DatasetDict objects.
"""

from create_dataset import load_dataset_as_datasetdict, load_full_dataset_as_dataset, DATASETS_AVAILABLE

def main():
    """Demonstrate loading datasets."""
    if not DATASETS_AVAILABLE:
        print("Error: datasets library not available.")
        print("Please install it with: pip install datasets")
        return
    
    print("=== Loading Multiplication Datasets ===\n")
    
    try:
        # Load basic multiplication dataset as DatasetDict
        print("1. Loading basic multiplication dataset splits:")
        dataset_dict = load_dataset_as_datasetdict(
            dataset_dir="datasets/basic",
            dataset_name="basic_multiplication"
        )
        
        print(f"\nDatasetDict structure:")
        print(f"- Splits: {list(dataset_dict.keys())}")
        for split_name, split_dataset in dataset_dict.items():
            print(f"- {split_name}: {len(split_dataset)} samples")
            print(f"  Features: {split_dataset.features}")
        
        # Show example from train split
        if 'train' in dataset_dict:
            print(f"\nExample from train split:")
            example = dataset_dict['train'][0]
            print(f"  Problem: {example['problem']}")
            print(f"  Answer: {example['answer']}")
        
        print("\n" + "-"*60 + "\n")
        
        # Load single dataset file
        print("2. Loading single dataset file:")
        single_dataset = load_full_dataset_as_dataset(
            "datasets/basic/basic_multiplication_train.json"
        )
        
        print(f"\nSingle Dataset structure:")
        print(f"- Samples: {len(single_dataset)}")
        print(f"- Features: {single_dataset.features}")
        
        # Show first few examples
        print(f"\nFirst 3 examples:")
        for i in range(min(3, len(single_dataset))):
            example = single_dataset[i]
            print(f"  {i+1}. {example['problem']} = {example['answer']}")
        
        print("\n" + "-"*60 + "\n")
        
        # Try to load advanced dataset if it exists
        print("3. Loading advanced multiplication dataset (if available):")
        try:
            advanced_dataset_dict = load_dataset_as_datasetdict(
                dataset_dir="datasets/advanced",
                dataset_name="advanced_multiplication"
            )
            
            print(f"\nAdvanced DatasetDict structure:")
            print(f"- Splits: {list(advanced_dataset_dict.keys())}")
            
            # Show difficulty distribution
            if 'train' in advanced_dataset_dict:
                train_data = advanced_dataset_dict['train']
                difficulties = {}
                for example in train_data:
                    diff = example.get('difficulty', 'unknown')
                    difficulties[diff] = difficulties.get(diff, 0) + 1
                
                print(f"- Difficulty distribution in train set:")
                for diff, count in difficulties.items():
                    print(f"  {diff}: {count} samples")
                
                print(f"\nExample from each difficulty:")
                shown_difficulties = set()
                for example in train_data:
                    diff = example.get('difficulty')
                    if diff and diff not in shown_difficulties:
                        print(f"  {diff}: {example['problem']} = {example['answer']}")
                        shown_difficulties.add(diff)
                        if len(shown_difficulties) >= 3:
                            break
        
        except Exception as e:
            print(f"Advanced dataset not available: {e}")
    
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Make sure you've generated the datasets first by running create_dataset.py")

if __name__ == "__main__":
    main()

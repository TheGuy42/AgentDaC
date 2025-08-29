import json
from typing import List, Dict, Any, Optional
from pathlib import Path

class SATDataset:
    """
    A class to load and manage SAT (Boolean Satisfiability) problem datasets.
    
    The dataset contains SAT problems with the following structure:
    - id: unique identifier for the problem
    - n_sat: number of variables in each clause (n-SAT problem)
    - k: total number of distinct variables in the problem
    - solution: string of length k representing truth values (1 for true, 0 for false)
    - clause: string representation of the SAT formula
    - prompt: formatted prompt for solving the problem
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            file_path: Path to the JSONL file containing SAT problems
        """
        self.file_path = Path(file_path)
        self.data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self):
        """Load data from the JSONL file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data_point = json.loads(line)
                        self.data.append(data_point)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        print(f"Loaded {len(self.data)} SAT problems from {self.file_path}")
    
    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific problem by index."""
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]
    
    def get_problem_by_id(self, problem_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a problem by its ID.
        
        Args:
            problem_id: The ID of the problem to retrieve
            
        Returns:
            The problem dictionary if found, None otherwise
        """
        for problem in self.data:
            if problem.get('id') == problem_id:
                return problem
        return None
    
    def get_problems_by_k(self, k: int) -> List[Dict[str, Any]]:
        """
        Get all problems with a specific number of variables.
        
        Args:
            k: Number of variables
            
        Returns:
            List of problems with k variables
        """
        return [problem for problem in self.data if problem.get('k') == k]
    
    def get_problems_by_n_sat(self, n_sat: int) -> List[Dict[str, Any]]:
        """
        Get all problems with a specific n-SAT constraint.
        
        Args:
            n_sat: Number of variables in each clause
            
        Returns:
            List of problems with n_sat variables per clause
        """
        return [problem for problem in self.data if problem.get('n_sat') == n_sat]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.data:
            return {}
        
        k_values = [problem.get('k', 0) for problem in self.data]
        n_sat_values = [problem.get('n_sat', 0) for problem in self.data]
        
        stats = {
            'total_problems': len(self.data),
            'k_range': {
                'min': min(k_values),
                'max': max(k_values),
                'unique': sorted(list(set(k_values)))
            },
            'n_sat_range': {
                'min': min(n_sat_values),
                'max': max(n_sat_values),
                'unique': sorted(list(set(n_sat_values)))
            }
        }
        
        return stats
    
    def sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get a random sample of problems.
        
        Args:
            n: Number of problems to sample
            
        Returns:
            List of sampled problems
        """
        import random
        return random.sample(self.data, min(n, len(self.data)))
    
    def verify_solutions(self) -> Dict[str, int]:
        """
        Verify that all problems have valid solutions.
        
        Returns:
            Dictionary with counts of valid/invalid solutions
        """
        valid_count = 0
        invalid_count = 0
        
        for problem in self.data:
            solution = problem.get('solution', '')
            clause = problem.get('clause', '')
            true = self.calc_sat_value(clause, solution)
            
            if true:
                valid_count += 1
            else:
                invalid_count += 1
        
        return {
            'valid_solutions': valid_count,
            'invalid_solutions': invalid_count,
            'total': len(self.data)
        }
    
    def calc_sat_value(self, clause, solution):
        def parse_literals(clause_str):
            literals = []
            i = 0
            while i < len(clause_str):
                if clause_str[i] == '!':
                    literals.append(clause_str[i:i+2])
                    i += 2
                else:
                    literals.append(clause_str[i])
                    i += 1
            return literals
        
        for subclause in clause.split(' & '):
            satisfied = False
            for lit in parse_literals(subclause):
                neg = False
                if lit.startswith('!'):
                    var = lit[1]
                    neg = True
                else:
                    var = lit
                
                idx = ord(var) - ord('A')
                if idx >= len(solution):
                    val = '0'
                else:
                    val = solution[idx]
                
                if (neg and val == '0') or (not neg and val == '1'):
                    satisfied = True
                    break
            
            if not satisfied:
                return 0
        return 1

# Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset = SATDataset("./data/train/Train_prompt_3_5_5_270/train.jsonl")
    
    # Print basic info
    print(f"Dataset size: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Get a sample problem
    sample_problem = dataset[0]
    print(f"\nSample problem:")
    print(f"ID: {sample_problem['id']}")
    print(f"n_sat: {sample_problem['n_sat']}")
    print(f"k: {sample_problem['k']}")
    print(f"Solution: {sample_problem['solution']}")
    print(f"Clause: {sample_problem['clause']}")
    
    # Verify solutions
    verification = dataset.verify_solutions()
    print(f"\nSolution verification: {verification}")
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

class ARCExample:
    """Represents a single ARC AGI example with train and test samples"""
    def __init__(self, data: Dict[str, Any]):
        self.train_samples = data['train']
        self.test_samples = data['test']
        
    def get_train_inputs_outputs(self) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """Get all training input-output pairs"""
        return [(sample['input'], sample['output']) for sample in self.train_samples]
    
    def get_test_inputs(self) -> List[List[List[int]]]:
        """Get all test inputs"""
        return [sample['input'] for sample in self.test_samples]
    
    def get_test_outputs(self) -> List[List[List[int]]]:
        """Get all test outputs (for validation if available)"""
        return [sample['output'] for sample in self.test_samples if 'output' in sample]

if __name__ == "__main__":
    # Load all examples from the data directory
    data_dir = Path("ARC-AGI/data/training")
    examples = []
    # Get all JSON files in the training directory
    json_files = list(data_dir.glob("*.json"))
    
    if json_files:
        # Select a random JSON file
        import random
        random_file = random.choice(json_files)
        
        # Load the JSON data
        with open(random_file, 'r') as f:
            import json
            data = json.load(f)
            
        # Create an ARCExample instance
        example = ARCExample(data)
        examples.append(example)
        
        print(f"Loaded random example from: {random_file.name}")
        print(f"Number of training samples: {len(example.train_samples)}")
        print(f"Number of test samples: {len(example.test_samples)}")
    else:
        print("No JSON files found in the training directory")
    


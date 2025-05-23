#!/usr/bin/env python3
"""
Test Reinforcement Learning (TTRL) for ARC AGI

This implements a reinforcement learning approach where:
1. Load an ARC AGI example
2. Use all training samples minus 1 for training
3. RL training loop until reward reaches 1
4. Test on all training samples, then apply to test sample
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from code_executor import CodeExecutor
from loader import ARCExample
from llm_call import LLMInterface


class TestRL:
    """Main Test Reinforcement Learning class"""
    
    def __init__(self, arc_data_path: str = "ARC-AGI/data/training"):
        self.arc_data_path = Path(arc_data_path)
        self.llm = LLMInterface()
        self.code_executor = CodeExecutor()
        
    def load_random_example(self) -> ARCExample:
        """Load a random ARC AGI example"""
        json_files = list(self.arc_data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.arc_data_path}")
        
        random_file = random.choice(json_files)
        print(f"📂 Loading example: {random_file.name}")
        
        with open(random_file, 'r') as f:
            data = json.load(f)
        
        return ARCExample(data)
    
    def calculate_reward(self, code: str, train_examples: List[Tuple[List[List[int]], List[List[int]]]], 
                        held_out_example: Tuple[List[List[int]], List[List[int]]]) -> Tuple[float, str]:
        """
        Calculate reward based on code performance
        Returns (reward, explanation)
        """
        # First check if code can be extracted
        if code is None:
            return -1.0, "Code could not be extracted from response"
        
        extracted_code = self.code_executor.extract_code_from_response(code)
        if extracted_code is None:
            return -1.0, "No valid transform function found in code"
        
        # Test on the held-out example
        held_out_input, held_out_output = held_out_example
        success, predicted_output, error = self.code_executor.execute_code_safely(
            extracted_code, held_out_input
        )
        
        if not success:
            return -0.5, f"Code extracted but failed to execute: {error}"
        
        # Check if output matches expected
        if predicted_output == held_out_output:
            return 1.0, "Code works correctly on held-out example!"
        else:
            return -0.5, f"Code executed but output doesn't match. Expected: {held_out_output}, Got: {predicted_output}"
    
    def test_on_all_training_samples(self, code: str, all_train_examples: List[Tuple[List[List[int]], List[List[int]]]]) -> Tuple[bool, str]:
        """
        Test the code on all training samples
        Returns (success, explanation)
        """
        extracted_code = self.code_executor.extract_code_from_response(code)
        if extracted_code is None:
            return False, "Code could not be extracted"
        
        for i, (train_input, train_output) in enumerate(all_train_examples):
            success, predicted_output, error = self.code_executor.execute_code_safely(
                extracted_code, train_input
            )
            
            if not success:
                return False, f"Failed on training sample {i+1}: {error}"
            
            if predicted_output != train_output:
                return False, f"Wrong output on training sample {i+1}. Expected: {train_output}, Got: {predicted_output}"
        
        return True, "Code works on all training samples!"
    
    def apply_to_test_sample(self, code: str, test_input: List[List[int]]) -> Tuple[bool, Optional[List[List[int]]], str]:
        """
        Apply the trained code to a test sample
        Returns (success, output, explanation)
        """
        extracted_code = self.code_executor.extract_code_from_response(code)
        if extracted_code is None:
            return False, None, "Code could not be extracted"
        
        success, predicted_output, error = self.code_executor.execute_code_safely(
            extracted_code, test_input
        )
        
        if not success:
            return False, None, f"Failed to execute on test sample: {error}"
        
        return True, predicted_output, "Successfully applied to test sample"
    
    def run_single_rl_iteration(self, example: ARCExample) -> Dict[str, Any]:
        """
        Run a single RL iteration:
        1. Hold out one training sample
        2. Train on remaining samples until reward = 1
        3. Test on all training samples
        4. If successful, apply to test samples
        """
        train_examples = example.get_train_inputs_outputs()
        
        if len(train_examples) < 2:
            return {"success": False, "reason": "Need at least 2 training samples"}
        
        # Hold out one training sample randomly
        held_out_idx = random.randint(0, len(train_examples) - 1)
        held_out_example = train_examples[held_out_idx]
        training_examples = [ex for i, ex in enumerate(train_examples) if i != held_out_idx]
        
        print(f"🎯 Held out training sample {held_out_idx + 1}/{len(train_examples)}")
        print(f"📚 Training on {len(training_examples)} samples")
        
        # RL Training loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        best_code = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 RL Iteration {iteration}")
            
            # Generate code from LLM
            generated_response = self.llm.generate_code(training_examples)
            
            # Calculate reward
            reward, explanation = self.calculate_reward(
                generated_response, training_examples, held_out_example
            )
            
            print(f"🎖️  Reward: {reward} - {explanation}")
            
            if reward == 1.0:
                print("🎉 Achieved reward = 1! Stopping RL training.")
                best_code = generated_response
                break
        
        if best_code is None:
            return {
                "success": False, 
                "reason": f"Failed to achieve reward = 1 after {max_iterations} iterations",
                "held_out_idx": held_out_idx
            }
        
        # Test on all training samples
        print("\n✅ Testing on all training samples...")
        all_train_success, train_explanation = self.test_on_all_training_samples(best_code, train_examples)
        
        if not all_train_success:
            return {
                "success": False,
                "reason": f"Failed on full training set: {train_explanation}",
                "held_out_idx": held_out_idx,
                "achieved_reward_1": True
            }
        
        print(f"✅ {train_explanation}")
        
        # Apply to test samples
        test_inputs = example.get_test_inputs()
        test_results = []
        
        print(f"\n🧪 Applying to {len(test_inputs)} test samples...")
        
        for i, test_input in enumerate(test_inputs):
            success, test_output, explanation = self.apply_to_test_sample(best_code, test_input)
            test_results.append({
                "success": success,
                "output": test_output,
                "explanation": explanation
            })
            print(f"Test sample {i+1}: {explanation}")
            if success:
                print(f"Output: {test_output}")
        
        return {
            "success": True,
            "held_out_idx": held_out_idx,
            "training_iterations": iteration,
            "final_code": best_code,
            "test_results": test_results
        }
    
    def run_full_experiment(self, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Run the full TTRL experiment:
        - Try different held-out samples until success or max_attempts reached
        """
        print("🚀 Starting Test Reinforcement Learning Experiment")
        print("=" * 60)
        
        # Load a random example
        example = self.load_random_example()
        print(f"📋 Example has {len(example.train_samples)} training samples and {len(example.test_samples)} test samples")
        
        for attempt in range(max_attempts):
            print(f"\n🎲 Attempt {attempt + 1}/{max_attempts}")
            print("-" * 40)
            
            result = self.run_single_rl_iteration(example)
            
            if result["success"]:
                print("\n🎊 EXPERIMENT SUCCESSFUL!")
                return {
                    "overall_success": True,
                    "attempts_needed": attempt + 1,
                    "final_result": result
                }
            else:
                print(f"\n❌ Attempt {attempt + 1} failed: {result['reason']}")
                if attempt < max_attempts - 1:
                    print("🔄 Trying with a different held-out sample...")
        
        print(f"\n💥 EXPERIMENT FAILED after {max_attempts} attempts")
        return {
            "overall_success": False,
            "attempts_made": max_attempts
        }

def main():
    """Main function to run the TTRL experiment"""
    print("🧠 Test Reinforcement Learning for ARC AGI")
    print("=" * 50)
    
    # Initialize TTRL
    ttrl = TestRL()
    
    # Run the experiment
    result = ttrl.run_full_experiment(max_attempts=3)
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    
    if result["overall_success"]:
        print("✅ SUCCESS!")
        print(f"🎯 Solved in {result['attempts_needed']} attempt(s)")
        final_result = result["final_result"]
        print(f"🔄 Required {final_result['training_iterations']} RL iterations")
        print(f"📝 Held out training sample {final_result['held_out_idx'] + 1}")
        
        # Show test results
        test_results = final_result["test_results"]
        successful_tests = sum(1 for r in test_results if r["success"])
        print(f"🧪 Test samples: {successful_tests}/{len(test_results)} successful")
        
    else:
        print("❌ FAILED")
        print(f"🔄 Tried {result['attempts_made']} attempts")
    
    print("\n🏁 Experiment complete!")

if __name__ == "__main__":
    main() 
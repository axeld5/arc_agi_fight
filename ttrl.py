#!/usr/bin/env python3
"""
Test Reinforcement Learning (TTRL) for ARC AGI

This implements a reinforcement learning approach where:
1. Load an ARC AGI example
2. Use all training samples minus 1 for training
3. RL training loop using GRPO until reward reaches 1
4. Test on all training samples, then apply to test sample
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from code_executor import CodeExecutor
from loader import ARCExample
from llm_call import LLMInterface

#MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
#MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
#MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME = "Qwen/Qwen3-1.7B"

class TestRL:
    """Main Test Reinforcement Learning class"""
    
    def __init__(self, arc_data_path: str = "ARC-AGI/data/training", model_name: str = MODEL_NAME, use_lora: bool = True):
        self.arc_data_path = Path(arc_data_path)
        self.llm = LLMInterface(model_name=model_name)
        self.code_executor = CodeExecutor()
        self.model_name = model_name
        self.max_seq_length = 32768
        self.use_lora = use_lora
        
        # Initialize model and tokenizer for GRPO training
        print(f"🤖 Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto"
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize model embeddings to account for new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled
        if self.use_lora:
            print("🔧 Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,  # Rank of adaptation
                lora_alpha=32,  # LoRA scaling parameter
                lora_dropout=0.1,  # LoRA dropout
                # Target modules - adjust based on your model architecture
                target_modules="all-linear",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"✅ LoRA applied. Trainable parameters: {self.model.print_trainable_parameters()}")
        else:
            # Enable gradient checkpointing and input gradients for full fine-tuning
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        
        # Move model to device
        if not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to("cuda:0")
        
        # Store current training context for reward function
        self.current_training_examples = None
        self.current_held_out_example = None
        
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
    
    def grpo_reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function compatible with GRPO trainer
        """
        rewards = []
        for completion in completions:
            reward, _ = self.calculate_reward(
                completion[0]["content"], 
                self.current_training_examples, 
                self.current_held_out_example
            )
            rewards.append(reward)
        return rewards
    
    def create_training_dataset(self, training_examples: List[Tuple[List[List[int]], List[List[int]]]]) -> Dataset:
        """
        Create a dataset for GRPO training from ARC examples
        """
        # Create prompt from training examples
        prompt_text = "Given the following input-output examples, write a Python function called 'transform' that converts the input to the output:\n\n"
        
        for i, (input_grid, output_grid) in enumerate(training_examples):
            prompt_text += f"Example {i+1}:\n"
            prompt_text += f"Input: {input_grid}\n"
            prompt_text += f"Output: {output_grid}\n\n"
        
        prompt_text += "Provide the function and only the function. Now, provide the Python function that solves this pattern:"
        print(prompt_text)
        print("🤖 Generating initial code with LLM...")
        initial_answer = self.llm.generate_code(training_examples)
        
        # Debug: Check if initial_answer is None
        if initial_answer is None:
            print("⚠️ Warning: LLM returned None, using fallback answer")
            initial_answer = '''def transform(input_grid: List[List[int]]) -> List[List[int]]:
    # Simple fallback - return input unchanged
    return input_grid'''
        
        print(f"✅ Got initial answer: {len(initial_answer)} characters")
        
        rows = [{
            "question": prompt_text,
            "answer": initial_answer,
            "prompt": [{"role": "user", "content": prompt_text}]
        }]
        
        return Dataset.from_list(rows)
    
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
        Run a single RL iteration using GRPO:
        1. Hold out one training sample
        2. Train using GRPO with remaining samples
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
        
        # Set current context for reward function
        self.current_training_examples = training_examples
        self.current_held_out_example = held_out_example
        
        # Create training dataset
        dataset = self.create_training_dataset(training_examples)
        
        # Setup GRPO training with LoRA-friendly settings
        training_args = GRPOConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_generations=2,
            bf16=True,
            use_vllm=False,
            max_steps=50 if not self.use_lora else 100,  # Fewer steps for LoRA as it trains faster
            max_completion_length=self.max_seq_length,
            optim="adamw_torch",
            learning_rate=5e-4 if not self.use_lora else 1e-4,  # Lower learning rate for LoRA
            logging_steps=5,
            save_steps=50,
        )
        
        trainer = GRPOTrainer(
            self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self.grpo_reward_function],
            train_dataset=dataset,
            args=training_args,
        )
        
        print(f"🚀 Starting GRPO training with {'LoRA' if self.use_lora else 'full fine-tuning'}...")
        trainer.train()
        
        # Generate final code using trained model
        prompt_text = f"Given the following input-output examples, write a Python function called 'transform':\n\n"
        for i, (input_grid, output_grid) in enumerate(training_examples):
            prompt_text += f"Example {i+1}:\nInput: {input_grid}\nOutput: {output_grid}\n\n"
        prompt_text += "Output the code and only the code.Python function:"
        
        # Use the trained model to generate multiple candidate codes and pick the one with highest reward
        num_generations = 8
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_seq_length)
        
        # Handle device placement for LoRA models
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        candidate_codes = []
        rewards_and_explanations = []
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=num_generations
            )
        for i in range(num_generations):
            code = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            reward, explanation = self.calculate_reward(
                code, training_examples, held_out_example
            )
            candidate_codes.append(code)
            rewards_and_explanations.append((reward, explanation))
        # Pick the code with the highest reward
        best_idx = max(range(num_generations), key=lambda i: rewards_and_explanations[i][0])
        best_code = candidate_codes[best_idx]
        final_reward, explanation = rewards_and_explanations[best_idx]
        
        print(f"Final code:\n{best_code}")
        print(f"🎖️  Final Reward: {final_reward} - {explanation}")
        
        if final_reward < 1.0:
            return {
                "success": False,
                "reason": f"GRPO training completed but final reward = {final_reward}",
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
            "training_method": f"GRPO with {'LoRA' if self.use_lora else 'full fine-tuning'}",
            "final_code": best_code,
            "test_results": test_results
        }
    
    def run_full_experiment(self, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Run the full TTRL experiment:
        - Try different held-out samples until success or max_attempts reached
        """
        print("🚀 Starting Test Reinforcement Learning Experiment with GRPO")
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
    print("🧠 Test Reinforcement Learning for ARC AGI with GRPO + LoRA")
    print("=" * 50)
    
    # Initialize TTRL with LoRA enabled
    ttrl = TestRL(use_lora=True)
    
    # Run the experiment
    result = ttrl.run_full_experiment(max_attempts=3)
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    
    if result["overall_success"]:
        print("✅ SUCCESS!")
        print(f"🎯 Solved in {result['attempts_needed']} attempt(s)")
        final_result = result["final_result"]
        print(f"🤖 Training method: {final_result.get('training_method', 'GRPO with LoRA')}")
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
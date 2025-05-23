import json
import os
from typing import List, Tuple

class LLMInterface:
    """Interface for interacting with LLM to generate code"""
    
    def generate_code(self, train_examples: List[Tuple[List[List[int]], List[List[int]]]]) -> str:
        """
        Generate code based on training examples
        This is a placeholder - in practice you'd call your LLM here
        """
        # Create a prompt for the LLM
        prompt = self._create_prompt(train_examples)
        
        # For demonstration, return a simple template
        # In practice, you'd call your LLM API here
        return self._get_llm_response(prompt)
    
    def _create_prompt(self, train_examples: List[Tuple[List[List[int]], List[List[int]]]]) -> str:
        """Create a prompt for the LLM based on training examples"""
        prompt = """You are an expert at solving ARC AGI puzzles. 

Please analyze the following input-output pairs and write a Python function called 'transform' that takes an input grid and returns the corresponding output grid.

The function signature should be:
def transform(input_grid: List[List[int]]) -> List[List[int]]:

Training examples:
"""
        
        for i, (input_grid, output_grid) in enumerate(train_examples):
            prompt += f"\nExample {i+1}:\n"
            prompt += f"Input:\n{input_grid}\n"
            prompt += f"Output:\n{output_grid}\n"
        
        prompt += "\nPlease provide only the transform function implementation in Python. Focus on identifying the pattern and implementing it correctly."
        
        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM
        This is a placeholder - replace with actual LLM API call
        """        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"🤖 Loading Qwen-2.5-Coder-1.5B from HuggingFace...")
            
            # Load the model and tokenizer
            model_name = "Qwen/Qwen2.5-Coder-1.5B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            llm_response = response_text[len(prompt):].strip()
            
            print(f"✅ Got response from Qwen-2.5-Coder-1.5B")
            return llm_response
                
        except Exception as e:
            print(f"⚠️ Failed to call Qwen-2.5-Coder-1.5B: {e}")
            print("📝 Falling back to demo response...")
            
            # Fallback demo response
            response = '''```python
def transform(input_grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(input_grid), len(input_grid[0])
    # Create output grid - looks like we need to detect 2x2 blocks of 2s
    output_height = (rows + 1) // 2
    output_width = (cols + 1) // 2
    output = [[0 for _ in range(output_width)] for _ in range(output_height)]
    
    # Check for 2x2 blocks of 2s
    for i in range(0, rows - 1, 2):
        for j in range(0, cols - 1, 2):
            # Check if we have a 2x2 block of 2s
            if (i + 1 < rows and j + 1 < cols and
                input_grid[i][j] == 2 and input_grid[i][j+1] == 2 and
                input_grid[i+1][j] == 2 and input_grid[i+1][j+1] == 2):
                output[i//2][j//2] = 1
    return output
    '''    
        return response
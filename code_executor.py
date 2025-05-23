import json
import os
import re
import sys
from typing import List, Tuple, Optional
import tempfile
import subprocess

class CodeExecutor:
    """Handles safe execution of generated code"""
    
    @staticmethod
    def extract_code_from_response(response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        # Look for code blocks
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'def transform\(.*?\):(.*?)(?=\n\n|\n#|\nif __name__|$)',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if 'def transform(' in code:
                    return code
        
        # If no code block found, look for def transform function directly
        if 'def transform(' in response:
            lines = response.split('\n')
            in_function = False
            function_lines = []
            indent_level = 0
            
            for line in lines:
                if 'def transform(' in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    function_lines.append(line)
                elif in_function:
                    if line.strip() == '':
                        function_lines.append(line)
                    elif len(line) - len(line.lstrip()) > indent_level:
                        function_lines.append(line)
                    else:
                        break
            
            if function_lines:
                return '\n'.join(function_lines)
        
        return None
    
    @staticmethod
    def execute_code_safely(code: str, input_grid: List[List[int]]) -> Tuple[bool, Optional[List[List[int]]], str]:
        """
        Execute code safely and return (success, output, error_message)
        """
        temp_file = None
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                full_code = f"""
import sys
import json
from typing import List

{code}

if __name__ == "__main__":
    input_data = {input_grid}
    try:
        result = transform(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)
"""
                f.write(full_code)
                f.flush()
            
            # File is now closed, safe to execute on Windows
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    return True, output, ""
                except json.JSONDecodeError:
                    return False, None, f"Invalid JSON output: {result.stdout}"
            else:
                return False, None, result.stderr
                    
        except subprocess.TimeoutExpired:
            return False, None, "Code execution timed out"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
        finally:
            # Clean up the temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    # If we can't delete it, it's not critical
                    pass
        
if __name__ == "__main__":
    # Test the code executor
    code = """
def transform(input_data):
    return input_data
"""
    input_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    success, output, error_message = CodeExecutor.execute_code_safely(code, input_grid)
    print(f"Success: {success}")
    print(f"Output: {output}")
    print(f"Error: {error_message}")

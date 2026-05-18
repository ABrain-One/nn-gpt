"""
Code Extractor module.
Extracts Python code from LLM output.
"""

import re
from typing import Optional


def extract_code(response: str) -> Optional[str]:
    """
    Extract Python code from the LLM output.
    
    Args:
        response: LLM response text containing code blocks
        
    Returns:
        Extracted Python code or None if no code block found
    """
    # Try to extract code from ```python ... ``` blocks
    if '```python' in response:
        try:
            code = response.split('```python')[1].split('```')[0]
            return code.strip()
        except IndexError:
            pass
    
    # Try to extract code from ``` ... ``` blocks (without language specifier)
    if '```' in response:
        try:
            parts = response.split('```')
            if len(parts) >= 3:
                code = parts[1]
                # Remove language specifier if present at the start
                if code.startswith(('python', 'py')):
                    code = code[code.find('\n')+1:]
                return code.strip()
        except IndexError:
            pass
    
    return None


def extract_all_code_blocks(response: str) -> list[str]:
    """
    Extract all Python code blocks from the LLM output.
    
    Args:
        response: LLM response text containing code blocks
        
    Returns:
        List of extracted code blocks
    """
    code_blocks = []
    
    # Pattern to match code blocks
    pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for match in matches:
        code = match.strip()
        # Remove language specifier if present
        if code.startswith(('python', 'py')):
            code = code[code.find('\n')+1:].strip()
        if code:
            code_blocks.append(code)
    
    return code_blocks


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate that the code is syntactically correct Python.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # compile() is a Python built-in function, 
        # its core function is to compile string form of Python code into bytecode object (or AST abstract syntax tree), 
        # this process only does syntax checking, without actually executing the code.
        compile(code, '<string>', 'exec') 
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_net_class(code: str) -> tuple[bool, str]:
    """
    Check if the code contains a Net class that inherits from nn.Module.
    
    Args:
        code: Python code to check
        
    Returns:
        Tuple of (has_net_class, message)
    """
    # Check for class Net definition
    if 'class Net' not in code:
        return False, "Code does not contain a 'class Net' definition"
    
    # Check for nn.Module inheritance
    if 'nn.Module' not in code:
        return False, "Net class does not inherit from nn.Module"
    
    # Check for __init__ method
    if 'def __init__' not in code:
        return False, "Net class does not have __init__ method"
    
    # Check for forward method
    if 'def forward' not in code:
        return False, "Net class does not have forward method"
    
    return True, "Code structure is valid"


class CodeExtractor:
    """Extracts and validates code from LLM responses."""
    
    def extract(self, response: str) -> tuple[Optional[str], str]:
        """
        Extract and validate code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (code, message) where code is None if extraction failed
        """
        # Extract code
        code = extract_code(response)
        
        if code is None:
            return None, "Failed to extract code block from response"
        
        # Validate syntax
        is_valid, error_msg = validate_code(code)
        if not is_valid:
            return None, f"Code syntax error: {error_msg}"
        
        # Check Net class structure
        has_net, net_msg = check_net_class(code)
        if not has_net:
            return None, f"Code structure error: {net_msg}"
        
        return code, "Code extracted and validated successfully"


if __name__ == "__main__":
    # Test the code extractor
    test_response = """
Here is a simple CNN model:

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, parameters: dict):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

This model uses a simple convolutional layer followed by a fully connected layer.
"""
    
    extractor = CodeExtractor()
    code, message = extractor.extract(test_response)
    print(f"Message: {message}")
    print(f"Extracted code:\n{code}")

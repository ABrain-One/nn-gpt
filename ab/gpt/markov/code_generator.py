"""
Code Generator module.
Uses LLM to generate PyTorch vision model code for image classification.
"""

import torch
import random
import numpy as np
from typing import Optional
from ab.gpt.markov.llm_client import LLMClient

_PROMPT_COMMON_TAIL = """
## Reference Code (Best Implementation So Far)
{reference_code}

## Current Iteration Code (Accuracy: {current_accuracy})
{current_code}

## Improvement Suggestions (from previous iteration)
{improvement_suggestions}

## Output format
```python
class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        
        self.xxx = xxx # output shape: (xxx)
        ...

    def forward(self, *args, **kwargs):
        pass
```"""

IMAGENETTE_PROMPT_TEMPLATE = """## Role

You are a visionary deep learning architect renowned for designing breakthrough neural networks by drawing inspiration from meta principles in diverse scientific domains.

## Task

Generate a vision model that maximizes the accuracy on the ImageNette dataset for the image classification task.
ImageNette is a 10-class subset of ImageNet. The input images are RGB with shape 3x160x160 (channels x height x width), and the model should output logits for 10 classes.

## Requirements

- Don't use pre-trained models.
- Contain the implementation of the model, no other code.
- If reference code is provided, improve upon it based on the improvement suggestions.
- IMPORTANT: Calculate and annotate the output shape of each layer after the definition of the layer in __init__.
- The input tensor shape is (batch_size, 3, 160, 160).
""" + _PROMPT_COMMON_TAIL

CIFAR10_PROMPT_TEMPLATE = """## Role

You are a visionary deep learning architect renowned for designing breakthrough neural networks by drawing inspiration from meta principles in diverse scientific domains.

## Task

Generate a vision model that maximizes the accuracy on the CIFAR-10 dataset for the image classification task.

## Requirements

- Don't use pre-trained models.
- Contain the implementation of the model, no other code.
- If reference code is provided, improve upon it based on the improvement suggestions.
- IMPORTANT: Calculate and annotate the output shape of each layer after the definition of the layer in __init__.
""" + _PROMPT_COMMON_TAIL

CIFAR100_PROMPT_TEMPLATE = """## Role

You are an expert PyTorch developer. Write clean and working PyTorch code.

## Task

Write a PyTorch vision model for the CIFAR-100 dataset (Input shape: 3x32x32, Output classes: 100).

## STRICT Requirements

1. The class MUST be named `Net`.
2. ONLY output standard PyTorch code inside one ```python ... ``` block. NO explanations before or after the code.
3. Apply the **Improvement Suggestions** strictly.
4. IMPORTANT: Calculate and annotate the output shape as a comment `# shape: [B, C, H, W]` after every layer defined in `__init__`.
5. The model MUST output logits for 100 classes.
6. DO NOT use `__import__` or download any datasets inside the code.
""" + _PROMPT_COMMON_TAIL

# Default prompt template (kept for backward compatibility)
INITIAL_PROMPT_TEMPLATE = IMAGENETTE_PROMPT_TEMPLATE


def get_prompt_template(dataset: str) -> str:
    """Return the appropriate initial prompt template for the given dataset."""
    if dataset == 'cifar100':
        return CIFAR100_PROMPT_TEMPLATE
    elif dataset == 'cifar10':
        return CIFAR10_PROMPT_TEMPLATE
    return IMAGENETTE_PROMPT_TEMPLATE

# For first iteration when there's no reference code
NO_REFERENCE_CODE = "No reference code available. This is the first iteration."

# For first iteration when there's no improvement suggestions
NO_IMPROVEMENT_SUGGESTIONS = "No improvement suggestions yet. This is the first iteration."

# For first iteration when there's no current iteration code
NO_CURRENT_CODE = "No current iteration code available. This is the first iteration."


class CodeGenerator:
    """Generates vision model code using LLM."""
    
    def __init__(self, 
                llm_client: LLMClient, 
                initial_prompt_template: str = None, 
                *args, **kwargs):
        """
        Initialize the code generator.
        
        Args:
            llm_client: LLM client for text generation
            initial_prompt_template: Initial prompt template. If None, uses default.
        """
        self.llm_client = llm_client
        self.prompt_template = initial_prompt_template or INITIAL_PROMPT_TEMPLATE
        self.improvement_suggestions: Optional[str] = None
        self.current_code: Optional[str] = None
        self.current_accuracy: Optional[float] = None
        self.args = args
        self.kwargs = kwargs

    def generate(
        self, 
        prompt_template: str = None, 
        reference_code: str = None,
        improvement_suggestions: str = None,
        current_code: str = None,
        current_accuracy: float = None
    ) -> str:
        """
        Generate vision model code.
        
        Args:
            prompt_template: Optional custom prompt template. If None, uses stored template.
            reference_code: Optional reference code from previous iteration.
            improvement_suggestions: Optional improvement suggestions from previous iteration.
            current_code: Optional code from the current/previous iteration.
            current_accuracy: Optional accuracy from the current/previous iteration.
            
        Returns:
            LLM response containing the generated code
        """
        template = prompt_template 
        ref_code = reference_code or NO_REFERENCE_CODE
        suggestions = improvement_suggestions or NO_IMPROVEMENT_SUGGESTIONS
        c_code = current_code or NO_CURRENT_CODE
        if current_accuracy is not None:
            c_acc_str = f"{current_accuracy*100:.2f}%"
        else:
            c_acc_str = "Failed" if current_code else "N/A (first iteration)"
        
        # Format the prompt with reference code and improvement suggestions
        current_prompt = template.format(
            reference_code=ref_code,
            improvement_suggestions=suggestions,
            current_code=c_code,
            current_accuracy=c_acc_str
        )
        
        response = self.llm_client.generate(
            current_prompt,
            max_new_tokens=2048,
            temperature=0.7,
        )
        return response
    
    def update_prompt_template(self, new_template: str):
        """
        Update the generation prompt template.
        
        Args:
            new_template: New prompt template to use for code generation
        """
        self.prompt_template = new_template
    
    def update_improvement_suggestions(self, suggestions: str):
        """
        Update the improvement suggestions for next generation.
        
        Args:
            suggestions: Improvement suggestions from prompt improver
        """
        self.improvement_suggestions = suggestions
    
    def update_current_code_and_accuracy(self, code: str, accuracy: float):
        """
        Update the current code and accuracy for next generation.
        """
        self.current_code = code
        self.current_accuracy = accuracy
    
    def get_prompt_template(self) -> str:
        """Get the current prompt template."""
        return self.prompt_template
    
    def get_improvement_suggestions(self) -> Optional[str]:
        """Get the current improvement suggestions."""
        return self.improvement_suggestions
    
    def get_current_code(self) -> Optional[str]:
        """Get the current code."""
        return self.current_code
    
    def get_current_accuracy(self) -> Optional[float]:
        """Get the current accuracy."""
        return self.current_accuracy


if __name__ == "__main__":
    # Test the code generator
    client = LLMClient()
    generator = CodeGenerator(client)
    
    print("Generating code (first iteration, no reference)...")
    response = generator.generate()
    print(f"Generated response:\n{response}")

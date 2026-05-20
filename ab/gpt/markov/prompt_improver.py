"""
Prompt Improver module.
Uses LLM to improve the code generation prompt based on evaluation feedback.
"""

import re
import os
import json
from typing import Optional

from ab.gpt.markov.llm_client import LLMClient


# Prompt template from Prompt Improver.md
IMAGENETTE_IMPROVER_PROMPT_TEMPLATE = """## Role

You are an excellent vision model architect. You have much experience in designing vision models. You are good at improving the vision model code. You have a lot of experience of debugging the vision model code.

## Task

You are given:
1. The BEST performing code so far (reference implementation)
2. The CURRENT iteration's code and its evaluation result
3. History of previous improvement attempts and their results (learn from past experience!)

Analyze the problems and provide specific improvement suggestions for the next iteration.

- The dataset is **ImageNette** (a 10-class subset of ImageNet, input shape: 3x160x160, 10 output classes).
- The evaluation result is a float number between 0 and 1, meaning the accuracy of the vision model.
- Sometimes the evaluation maybe a message string, meaning the error or other questions of the vision model. In this case, you should provide suggestions to fix the error.
- Your suggestions should help improve upon the BEST code, not just fix the current iteration's issues.
- IMPORTANT: Learn from the improvement history! Avoid repeating suggestions that didn't work, and build upon ideas that showed improvement.

## Improvement History (Recent Iterations)
{improvement_history}

## Best Code (Reference - Accuracy: {best_accuracy})
{best_code}

## Current Iteration Code (Accuracy: {current_accuracy})
{current_code}

## Feedback from Evaluator
{message_from_evaluator}

## Tips

- **Learn from history**: Review what was tried before and what results it produced. Don't repeat failed approaches.
- Compare the current code with the best code to understand what changes led to performance differences.
- Analyze the possible reasons why the accuracy is low or why the error occurred.
- Provide specific, actionable improvement suggestions that build upon the BEST code.
- You have a lot of knowledge, so the inspiration can come from any subject, such as computer science, philosophy, economics, biology, etc.
- Focus on concrete architectural changes, hyperparameter suggestions, or bug fixes.

## Output format
```json
{{
    "reason": "The reason why the accuracy is low or the error occurred",
    "inspiration": "The inspiration from any subject, such as computer science, philosophy, economics, biology, etc.",
    "improvement_suggestions": "Specific, actionable suggestions on how to improve upon the best code. Be detailed and concrete."
}}
```
"""

CIFAR10_IMPROVER_PROMPT_TEMPLATE = """## Role

You are an excellent vision model architect. You have much experience in designing vision models. You are good at improving the vision model code. You have a lot of experience of debugging the vision model code.

## Task

You are given:
1. The BEST performing code so far (reference implementation)
2. The CURRENT iteration's code and its evaluation result
3. History of previous improvement attempts and their results (learn from past experience!)

Analyze the problems and provide specific improvement suggestions for the next iteration.

- The evaluation result is a float number between 0 and 1, meaning the accuracy of the vision model.
- Sometimes the evaluation maybe a message string, meaning the error or other questions of the vision model. In this case, you should provide suggestions to fix the error.
- Your suggestions should help improve upon the BEST code, not just fix the current iteration's issues.
- IMPORTANT: Learn from the improvement history! Avoid repeating suggestions that didn't work, and build upon ideas that showed improvement.

## Improvement History (Recent Iterations)
{improvement_history}

## Best Code (Reference - Accuracy: {best_accuracy})
{best_code}

## Current Iteration Code (Accuracy: {current_accuracy})
{current_code}

## Feedback from Evaluator
{message_from_evaluator}

## Tips

- **Learn from history**: Review what was tried before and what results it produced. Don't repeat failed approaches.
- Compare the current code with the best code to understand what changes led to performance differences.
- Analyze the possible reasons why the accuracy is low or why the error occurred.
- Provide specific, actionable improvement suggestions that build upon the BEST code.
- You have a lot of knowledge, so the inspiration can come from any subject, such as computer science, philosophy, economics, biology, etc.
- Focus on concrete architectural changes, hyperparameter suggestions, or bug fixes.

## Output format
```json
{{
    "reason": "The reason why the accuracy is low or the error occurred",
    "inspiration": "The inspiration from any subject, such as computer science, philosophy, economics, biology, etc.",
    "improvement_suggestions": "Specific, actionable suggestions on how to improve upon the best code. Be detailed and concrete."
}}
```
"""

CIFAR100_IMPROVER_PROMPT_TEMPLATE = """## Role

You are a PyTorch debugging and model optimization expert.

## Task

Analyze the current vision model's performance and provide specific, actionable code changes for the next iteration.

## Rules

- Dataset: CIFAR-100 (Input shape: 3x32x32, 100 output classes).
- If there is an Error Message, your ONLY priority is to fix the bug.
- Avoid repeating suggestions from the Improvement History that previously failed.
- Do NOT hallucinate complex theories. Suggest practical, standard deep learning improvements (e.g., adding BatchNorm, changing strides, adding Residual Connections, adjusting Dropout).
- The model MUST output logits for 100 classes.

## Improvement History (Recent Iterations)
{improvement_history}

## Best Code (Reference - Accuracy: {best_accuracy})
{best_code}

## Current Iteration Code (Accuracy: {current_accuracy})
{current_code}

## Feedback from Evaluator
{message_from_evaluator}

## Output Format
You must respond with valid JSON exactly matching this format:
```json
{{
    "analysis": "Briefly explain why the model failed or why accuracy is low.",
    "improvement_suggestions": "1. Change X to Y. 2. Add layer Z before layer W. (Be concrete and actionable for a coder)"
}}
```
"""

class PromptImprover:
    """Generates improvement suggestions based on evaluation feedback."""
    
    def __init__(self, llm_client: LLMClient, dataset: str = 'imagenette'):
        """
        Initialize the prompt improver.
        
        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client
        self.dataset = dataset
        if dataset == 'imagenette':
            self.IMPROVER_PROMPT_TEMPLATE = IMAGENETTE_IMPROVER_PROMPT_TEMPLATE
        elif dataset == 'cifar10':
            self.IMPROVER_PROMPT_TEMPLATE = CIFAR10_IMPROVER_PROMPT_TEMPLATE
        elif dataset == 'cifar100':
            self.IMPROVER_PROMPT_TEMPLATE = CIFAR100_IMPROVER_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    
    def improve(
        self,
        best_code: Optional[str],
        best_accuracy: float,
        current_code: str,
        current_accuracy: Optional[float],
        feedback: str,
        history: Optional[list[dict]] = None,
        output_dir: Optional[str] = None,
        current_iteration: Optional[int] = None
    ) -> dict:
        """
        Generate improvement suggestions based on evaluation feedback.
        
        Args:
            best_code: The best performing code so far (reference)
            best_accuracy: The accuracy of the best code
            current_code: The code from current iteration
            current_accuracy: The accuracy of current code (None if failed)
            feedback: Feedback from the evaluator
            history: List of previous improvement attempts with their results
            output_dir: The directory to save the prompt
            current_iteration: The current iteration number. We need this to save the prompt.
            dataset: The dataset being used ('imagenette' or 'cifar10')
        Returns:
            Dictionary with keys: reason, inspiration, improvement_suggestions
        """
        # Format accuracy strings
        best_acc_str = f"{best_accuracy*100:.2f}%" if best_accuracy > 0 else "N/A (first iteration)"
        current_acc_str = f"{current_accuracy*100:.2f}%" if current_accuracy is not None else "Failed"
        
        # Format history string
        history_str = self._format_history(history)
        
        improver_prompt = self.IMPROVER_PROMPT_TEMPLATE.format(
            best_code=best_code if best_code else "No best code yet (first iteration)",
            best_accuracy=best_acc_str,
            current_code=current_code if current_code else "No code was generated",
            current_accuracy=current_acc_str,
            message_from_evaluator=feedback,
            improvement_history=history_str
        )
        
        # DEBUG: Write the prompt to the directory of current iteration
        os.makedirs(os.path.join(output_dir, "prompts_of_improver"), exist_ok=True)
        with open(os.path.join(output_dir, "prompts_of_improver", f"prompts_{current_iteration}.md"), "w") as f:
            f.write(improver_prompt + "\n")
        
        # Generate improvement suggestions
        response = self.llm_client.generate(
            improver_prompt,
            max_new_tokens=2048,
            temperature=0.7
        )
        
        # Parse the JSON response
        result = self._parse_response(response)
        return result
    
    def _format_history(self, history: Optional[list[dict]]) -> str:
        """
        Format the improvement history for the prompt.
        
        Args:
            history: List of history entries, each containing:
                - iteration: int
                - problem: str (the identified problem/reason)
                - suggestion: str (the improvement suggestion)
                - result: str (the outcome - accuracy or error)
        
        Returns:
            Formatted string for the prompt
        """
        if not history:
            return "No previous history yet (first iteration)."
        
        lines = []
        for entry in history:
            lines.append(f"### Iteration {entry['iteration']}")
            lines.append(f"- **Problem identified**: {entry.get('problem', 'N/A')}")
            lines.append(f"- **Suggestion given**: {entry.get('suggestion', 'N/A')[:300]}...")  # Truncate long suggestions
            lines.append(f"- **Result after applying**: {entry.get('result', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse the JSON response from LLM.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed dictionary or default values if parsing fails
        """
        # Try to extract JSON from code block
        json_str = None
        
        if '```json' in response:
            try:
                json_str = response.split('```json')[1].split('```')[0].strip()
            except IndexError:
                pass
        elif '```' in response:
            try:
                parts = response.split('```')
                if len(parts) >= 3:
                    json_str = parts[1].strip()
                    if json_str.startswith('json'):
                        json_str = json_str[4:].strip()
            except IndexError:
                pass
        
        # Try to find JSON object directly
        if json_str is None:
            # Try to find JSON object pattern for analysis or reason
            match_analysis = re.search(r'\{[^{}]*"analysis"[^{}]*\}', response, re.DOTALL)
            match_reason = re.search(r'\{[^{}]*"reason"[^{}]*\}', response, re.DOTALL)
            if match_analysis:
                json_str = match_analysis.group()
            elif match_reason:
                json_str = match_reason.group()
        
        # Parse JSON
        if json_str:
            try:
                result = json.loads(json_str)
                # Extract improvement_suggestions (support both field names)
                suggestions = result.get('improvement_suggestions') or result.get('improvement_suggestion', 'No specific suggestions')
                return {
                    'reason': result.get('analysis', result.get('reason', 'Unknown')),
                    'inspiration': result.get('inspiration', 'None'),
                    'improvement_suggestions': suggestions
                }
            except json.JSONDecodeError:
                pass
        
        # If parsing fails, try to extract suggestions directly
        return self._extract_suggestions_fallback(response)
    
    def _extract_suggestions_fallback(self, response: str) -> dict:
        """
        Fallback method to extract suggestions from response.
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with extracted or default values
        """
        # Try to find "improvement_suggestions" field content
        suggestions_match = re.search(r'"improvement_suggestions?"\s*:\s*"([^"]+)"', response, re.DOTALL)
        if suggestions_match:
            return {
                'reason': 'Extracted from partial response',
                'inspiration': 'None',
                'improvement_suggestions': suggestions_match.group(1)
            }
        
        # Return empty result if nothing found
        return {
            'reason': 'Failed to parse LLM response',
            'inspiration': 'None',
            'improvement_suggestions': None
        }


if __name__ == "__main__":
    # Test the prompt improver JSON parsing
    test_response = '''
Here is my analysis:

```json
{
    "analysis": "The model lacks capacity to handle 100 classes with simple convolutions without batch normalization.",
    "improvement_suggestions": "1. Add batch normalization after each convolutional layer. 2. Increase base channels."
}
```
'''
    
    improver = PromptImprover(None)  # LLM client not needed for parsing test
    result = improver._parse_response(test_response)
    print(f"Parsed result: {json.dumps(result, indent=2)}")

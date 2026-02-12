#!/usr/bin/env python3
"""
Integrated MoE Pipeline - Generate, Test, Keep Only Successes
Simple workflow: Generate → Train 1 epoch → If works, save. If fails, discard.
"""

import argparse
import importlib.util
import json
import os
import py_compile
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM


class IntegratedMoEPipeline:
    """All-in-one: Generate → Validate → Train → Save (only successes)"""

    def __init__(self,
                 nn_dir: str,
                 output_dir: str,
                 model_name: str = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                 temperature: float = 0.2,  # Lower temp for more deterministic code generation
                 max_new_tokens: int = 32768,
                 batch_size: int = 128,
                 training_timeout: int = 600,
                 epoch: int = 0,
                 debug: bool = False,
                 max_retries: int = 3):
        """
        Args:
            nn_dir: Path to nn/ directory with expert architectures
            output_dir: Base output directory
            model_name: HuggingFace model for generation
            temperature: Sampling temperature
            max_new_tokens: Max generation length
            batch_size: Training batch size
            training_timeout: Max seconds per training (not used, but kept for future)
            epoch: Epoch number for directory structure
        """
        self.nn_dir = Path(nn_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.training_timeout = training_timeout
        self.epoch = epoch
        self.debug = debug
        self.max_retries = max_retries

        # Create epoch directory
        self.epoch_dir = self.output_dir / "epoch" / f"A{epoch}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Statistics
        self.attempted = 0
        self.successful = 0
        self.failed = 0
        self.success_counter = 0  # For B{i} numbering
        self.repaired = 0  # Models that needed repair
        self.results = []

        # Load LLM
        print(f"\n🤖 Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ LLM loaded on {self.device}")

        # Load CIFAR-10
        print(f"\n📦 Loading CIFAR-10...")
        self.train_loader, self.test_loader = self._load_cifar10()
        print(f"✅ Dataset loaded")

    def _load_cifar10(self):
        """Load CIFAR-10 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader

    def read_expert_file(self, expert_filename: str) -> str:
        """Read expert architecture code"""
        expert_path = self.nn_dir / expert_filename
        with open(expert_path, 'r') as f:
            return f.read()

    def create_prompt(self, expert_files: list, expert_names: list) -> str:
        """Create MoE generation prompt"""
        expert_codes = []
        for i, (filename, name) in enumerate(zip(expert_files, expert_names), 1):
            code = self.read_expert_file(filename)
            expert_codes.append(f"EXPERT {i}: {name}\n```python\n{code}\n```\n")

        experts_prompt = "\n\n".join(expert_codes)

        # Build dynamic examples using actual expert names
        num_experts = len(expert_names)
        expert_list_code = ',\n            '.join([f"{name}(in_shape, out_shape, prm, device)" for name in expert_names])
        expert_class_examples = '\n\n'.join([f"""class {name}(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        # Build {name} layers here - NO .to(device) calls!""" for name in expert_names])

        prompt = f"""CREATE a Heterogeneous Mixture-of-Experts (MoE) neural network model that combines the following expert architectures:

{experts_prompt}

⚠️ ⚠️ ⚠️ CRITICAL - DEVICE PLACEMENT (READ THIS FIRST) ⚠️ ⚠️ ⚠️

THIS IS THE #1 CAUSE OF FAILURES. You MUST follow these rules EXACTLY:

RULE 1 - ALL device placement happens in Net.__init__, NOWHERE ELSE:
```python
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        # Create experts (they stay on CPU initially)
        self.experts = nn.ModuleList([
            {expert_list_code}
        ]).to(self.device)  # ← MOVE ENTIRE ModuleList TO DEVICE!

        # Create gate
        self.gate = nn.Sequential(
            nn.Conv2d(in_shape[1], {num_experts}, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear({num_experts}, {num_experts})
        ).to(self.device)  # ← MOVE ENTIRE Sequential TO DEVICE!
```

RULE 2 - Inside each expert ({', '.join(expert_names)}), NO .to(device):
```python
class {expert_names[0]}(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        # Just create modules - Net will move them to device
        self.features = nn.Sequential(...)
        self.classifier = nn.Sequential(...)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # NO .to(device) calls here!
```

RULE 3 - NO .to(device) in forward(), train_setup(), or learn()

RULE 4 - If you use device= parameter, use it EVERYWHERE in that module:
```python
# Option A: Use .to(self.device) on Sequential
self.gate = nn.Sequential(...).to(self.device)

# Option B: Use device= on every layer
self.conv1 = nn.Conv2d(3, 64, 3, device=device)
self.linear = nn.Linear(256, 10, device=device)
```
NEVER mix both approaches in the same module.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERT CLASS REQUIREMENTS (CRITICAL):

ALL {num_experts} expert classes MUST be nn.Module subclasses with IDENTICAL signatures.
DO NOT define experts as factory functions! They MUST be classes.

```python
{expert_class_examples}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED API STRUCTURE (DO NOT DEVIATE):

```python
def supported_hyperparameters() -> set:
    # MODULE-LEVEL function (outside all classes)
    return {{'lr', 'momentum', 'dropout'}}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        # EXACTLY these 4 parameters
        super().__init__()
        self.device = device
        # Create expert instances (AlexNet, MobileNetV2, ConvNeXt, DenseNet)
        # and gate, then move to device (see RULE 1 above)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense MoE routing (see example below)
        gate_weights = self.gate(x)  # (batch, 4)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch, num_classes, 4)
        gate_weights = gate_weights.unsqueeze(1)  # (batch, 1, 4)
        return torch.sum(expert_outputs * gate_weights, dim=2)  # (batch, num_classes)

    def train_setup(self, prm: dict):
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTS & CODE ORDERING:

1. Import ALL needed symbols at the top:
   ```python
   import torch
   import torch.nn as nn
   from torch import Tensor
   from typing import List, Tuple, Optional, Callable, Dict, Any
   from functools import partial
   from collections import OrderedDict
   ```

2. Helper classes/functions MUST be defined BEFORE they are used:
   - _make_divisible (if needed)
   - Conv2dNormActivation
   - LayerNorm2d
   - CNBlockConfig
   - Other helpers
   - supported_hyperparameters()
   - Expert classes (AlexNet, MobileNetV2, etc.)
   - Net class (LAST)

3. NO external dependencies (no torchvision.models, no timm)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CIFAR-10 ADAPTATIONS:

- Input: (batch, 3, 32, 32)
- Output: (batch, 10)
- in_shape = (1, 3, 32, 32) where in_shape[1] = 3 (channels)
- Use in_shape[1] for first Conv2d input channels
- Each expert MUST end with: AdaptiveAvgPool2d((1,1)) → Flatten → Linear(channels, 10)
- AlexNet: Use kernel_size=3, stride=1 (NOT 11/4 for ImageNet)
- Conv2dNormActivation: ALWAYS include kernel_size parameter

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate COMPLETE working code inside <nn> tags (NO markdown, NO error messages, NO numbered lists).
DO NOT include placeholder ellipsis (...) - write FULL implementations.

Example structure (replace ... with actual code):
- Import ALL needed symbols at top
- Define _make_divisible, Conv2dNormActivation, LayerNorm2d helpers
- Define supported_hyperparameters() at module level
- Define all {num_experts} expert classes ({', '.join(expert_names)}) with full implementations
- Define Net class LAST with experts and gate moved to device

<nn>
</nn>"""

        return prompt

    def postprocess_code(self, code: str) -> str:
        """Auto-fix common issues in generated code before validation."""
        lines = code.split('\n')

        # 0. Check for truncated code (ends mid-statement)
        if lines:
            last_line = lines[-1].strip()
            # Reject if last line looks incomplete
            if last_line and (
                last_line.endswith('(') or  # Incomplete function call
                last_line.endswith(',') or  # Incomplete argument list
                last_line.endswith('=') or  # Incomplete assignment
                (last_line and not last_line[-1] in ':)}]"\'' and len(last_line.split()) == 1)  # Incomplete statement
            ):
                print(f"    ⚠️  Truncated code detected (last line: '{last_line[:50]}...'), rejecting")
                return None

        # 1. AGGRESSIVE markdown stripping
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip markdown/error message artifacts
            if stripped.startswith('```') or stripped.startswith('- For CIFAR') or stripped.startswith('- Input:'):
                continue
            if stripped.startswith('IMPORTANT CONSTRAINTS') or stripped.startswith('ERROR at'):
                continue
            # Skip numbered lists from error messages (e.g., "1. Added...", "9. Fixed...")
            if stripped and stripped[0].isdigit() and '. ' in stripped[:5]:
                continue
            # Skip lines that look like comments from repair messages
            if stripped.startswith('# Fixed:') or stripped.startswith('# Added:') or stripped.startswith('# Changes:'):
                continue
            cleaned_lines.append(line)
        lines = cleaned_lines

        # 1. Auto-add missing imports (basic + typing + common stdlib)
        code_body = '\n'.join(lines)

        # Basic torch imports (CRITICAL - repairs often strip these)
        basic_imports = [
            ('import torch', 'torch.'),
            ('import torch.nn as nn', 'nn.'),
            ('import torch.nn.functional as F', ' F.'),
        ]

        for import_line, usage_pattern in basic_imports:
            if usage_pattern in code_body and import_line not in code_body:
                # Insert at the very top
                lines.insert(0, import_line)

        code_body = '\n'.join(lines)  # Refresh

        # Typing imports (expanded to cover all common types)
        typing_needed = set()
        typing_map = {
            'Optional': 'Optional[',
            'Callable': 'Callable[',
            'Tuple': 'Tuple[',
            'List': 'List[',
            'Dict': 'Dict[',
            'Any': ': Any',
            'Union': 'Union[',  # NEW: Fix A2 combo 2 error
            'Type': 'Type[',    # NEW: Fix common type hint errors
            'Sequence': 'Sequence[',  # NEW: Common in nn code
            'Set': 'Set[',      # NEW
        }
        for name, pattern in typing_map.items():
            if pattern in code_body and name not in code_body.split('from typing import')[-1].split('\n')[0]:
                typing_needed.add(name)

        if typing_needed:
            for i, line in enumerate(lines):
                if line.strip().startswith('from typing import'):
                    existing = set(x.strip() for x in line.split('import')[1].split(','))
                    existing.update(typing_needed)
                    lines[i] = f"from typing import {', '.join(sorted(existing))}"
                    typing_needed.clear()
                    break
            if typing_needed:
                last_import = max((i for i, l in enumerate(lines) if l.strip().startswith(('import ', 'from '))), default=0)
                lines.insert(last_import + 1, f"from typing import {', '.join(sorted(typing_needed))}")

        # Other common imports
        code_body = '\n'.join(lines)  # Refresh after typing imports
        import_checks = {
            'partial': ('from functools import partial', 'partial('),
            'OrderedDict': ('from collections import OrderedDict', 'OrderedDict('),
            'Tensor': ('from torch import Tensor', ': Tensor'),  # Type hint usage
        }

        for import_name, (import_line, usage_pattern) in import_checks.items():
            if usage_pattern in code_body and import_line not in code_body:
                # Add missing import
                last_import = max((i for i, l in enumerate(lines) if l.strip().startswith(('import ', 'from '))), default=0)
                lines.insert(last_import + 1, import_line)

        # Fix missing private torchvision classes (EfficientNet, RegNet, etc.)
        code_body = '\n'.join(lines)  # Refresh after common imports

        # Define private classes that need to be created if missing
        private_class_definitions = {
            '_MBConvConfig': '''
# Auto-generated: Missing private class for EfficientNet/MobileNet
from dataclasses import dataclass

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: type = None

    def __init__(self, expand_ratio, kernel, stride, input_channels, out_channels, num_layers):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
''',
            '_log_api_usage_once': '''
# Auto-generated: Missing utility function
def _log_api_usage_once(obj):
    """Placeholder for torchvision._internally_replaced_utils._log_api_usage_once"""
    pass
''',
        }

        # Check if any private classes are used but not defined
        for class_name, definition in private_class_definitions.items():
            if class_name in code_body and f'class {class_name}' not in code_body and f'def {class_name}' not in code_body:
                # Insert definition after imports, before first class
                first_class_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('class '):
                        first_class_idx = i
                        break

                if first_class_idx is not None:
                    # Insert before first class
                    lines.insert(first_class_idx, definition.strip())
                else:
                    # No class found, append after imports
                    last_import = max((i for i, l in enumerate(lines) if l.strip().startswith(('import ', 'from '))), default=0)
                    lines.insert(last_import + 1, definition.strip())

        # 2. REMOVED: Conv2dNormActivation auto-fix (was causing syntax errors)
        # Let DeepSeek generate correct calls or repair will handle it

        # 3. Fix Net.__init__ signature ONLY if it's clearly wrong (missing all 4 params)
        code_body = '\n'.join(lines)
        if 'class Net(' in code_body:
            in_net, found_init, fixed_lines = False, False, []
            for i, line in enumerate(lines):
                if 'class Net(' in line:
                    in_net = True
                    found_init = False
                if in_net and 'def __init__' in line and not found_init:
                    found_init = True
                    # Only fix if ALL FOUR params are missing (clear error)
                    missing_count = sum(1 for p in ['in_shape', 'out_shape', 'prm', 'device'] if p not in line)
                    if missing_count >= 3:  # If 3+ params missing, it's wrong
                        indent = len(line) - len(line.lstrip())
                        fixed_lines.append(' ' * indent + 'def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:')
                        continue
                if in_net and line and not line[0].isspace() and ('class ' in line or 'def ' in line) and 'class Net' not in line:
                    in_net = False
                fixed_lines.append(line)
            lines = fixed_lines

        # 4. REMOVED broken device auto-fix - DeepSeek should handle this in repair
        # The auto-fix was corrupting code by creating incomplete lines

        final_code = '\n'.join(lines)

        # 5. Final sanity check: code MUST have Net class and supported_hyperparameters
        if 'class Net(' not in final_code or 'def supported_hyperparameters' not in final_code:
            print(f"    ⚠️  Code missing required elements (class Net or supported_hyperparameters), rejecting")
            return None

        return final_code

    def generate_code(self, prompt: str) -> str:
        """Generate MoE code using LLM"""
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)

            prompt_length = inputs.shape[1]  # Track prompt length to decode only new tokens

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # CRITICAL FIX: Decode only NEW tokens (skip prompt to avoid <nn> tag contamination)
            generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

            # Debug: save full LLM output
            if self.debug:
                debug_dir = self.epoch_dir / "debug" / "raw_generations"
                debug_dir.mkdir(parents=True, exist_ok=True)
                with open(debug_dir / f"gen_{int(time.time())}.txt", 'w') as f:
                    f.write(generated_text)

            # Extract code from <nn> tags - use LAST occurrence (after reasoning)
            if '<nn>' in generated_text and '</nn>' in generated_text:
                # Find ALL <nn> tags and use the LAST one (actual code, not reasoning examples)
                start_positions = []
                pos = 0
                while True:
                    pos = generated_text.find('<nn>', pos)
                    if pos == -1:
                        break
                    start_positions.append(pos)
                    pos += 4

                if start_positions:
                    # Use LAST <nn> tag
                    start = start_positions[-1] + 4
                    end = generated_text.find('</nn>', start)

                    if end != -1:
                        code = generated_text[start:end].strip()

                        # Remove any leading non-code text (preserve imports at start)
                        if code and not code.startswith(('import', 'from', '#')):
                            lines = code.split('\n')
                            for i, line in enumerate(lines):
                                stripped = line.strip()
                                if stripped.startswith(('import', 'from', '#', 'def', 'class')):
                                    code = '\n'.join(lines[i:])
                                    break

                        if code and len(code) > 100:  # Sanity check: code should be substantial
                            # Reject template code (check for PLACEHOLDER ellipsis, not type hints like Callable[..., nn.Module])
                            has_placeholder = (
                                '[Your complete implementation here]' in code or
                                '\n...\n' in code or  # Standalone ellipsis lines
                                code.strip() == '...' or
                                (code.count('\n...') > 2 and 'Callable[' not in code)  # Multiple ellipsis but not type hints
                            )
                            if not has_placeholder:
                                processed = self.postprocess_code(code)
                                if processed:  # Only return if postprocess succeeded
                                    return processed

            # Fallback 1: try ```python blocks (use LONGEST block, not first)
            if '```python' in generated_text:
                # Find all python blocks
                blocks = []
                pos = 0
                while True:
                    start = generated_text.find('```python', pos)
                    if start == -1:
                        break
                    start += 9
                    end = generated_text.find('```', start)
                    if end != -1:
                        blocks.append(generated_text[start:end].strip())
                        pos = end + 3
                    else:
                        break

                # Use LONGEST block (most likely to be actual code, not snippet)
                if blocks:
                    code = max(blocks, key=len)
                    has_placeholder = (
                        '[Your complete implementation here]' in code or
                        '\n...\n' in code or
                        code.strip() == '...' or
                        (code.count('\n...') > 2 and 'Callable[' not in code)
                    )
                    if code and len(code) > 100 and not has_placeholder:
                        processed = self.postprocess_code(code)
                        if processed:
                            return processed

            # Fallback 2: Extract code after </think> (DeepSeek-R1 pattern)
            if '</think>' in generated_text:
                # Find last </think>
                think_end = generated_text.rfind('</think>')
                if think_end != -1:
                    after_think = generated_text[think_end + 8:].strip()

                    # Look for first import statement (skip markdown/comments)
                    lines = after_think.split('\n')
                    code_start = None
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # Skip markdown code fences and non-code lines
                        if stripped.startswith('```'):
                            continue
                        if stripped.startswith(('import ', 'from ')):
                            code_start = i
                            break

                    if code_start is not None:
                        # Extract from first import to end
                        code_lines = lines[code_start:]

                        # Remove trailing non-code (markdown closing, etc.)
                        for i in range(len(code_lines) - 1, -1, -1):
                            line = code_lines[i].strip()
                            if line and not line.startswith(('```', '<', '#', '//')):
                                # Check if it's actual code (not just closing tags)
                                if any(c.isalpha() or c in '(){}[]' for c in line):
                                    code_lines = code_lines[:i+1]
                                    break

                        code = '\n'.join(code_lines).strip()

                        # Final cleanup: remove any leading/trailing markdown
                        while code.startswith('```'):
                            first_newline = code.find('\n')
                            if first_newline != -1:
                                code = code[first_newline+1:].strip()
                            else:
                                break

                        while code.endswith('```'):
                            last_newline = code.rfind('\n```')
                            if last_newline != -1:
                                code = code[:last_newline].strip()
                            else:
                                break

                        if code and len(code) > 500:  # Should be substantial
                            has_placeholder = (
                                '[Your complete implementation here]' in code or
                                '\n...\n' in code or
                                code.strip() == '...' or
                                (code.count('\n...') > 2 and 'Callable[' not in code)
                            )
                            if not has_placeholder:
                                processed = self.postprocess_code(code)
                                if processed:
                                    return processed

            return None

        except Exception as e:
            print(f"    ❌ Generation error: {e}")
            return None

    def validate_code(self, code: str, temp_dir: Path) -> dict:
        """Validate code through 5 stages. Returns None if valid, or error dict with details."""
        temp_file = temp_dir / "temp_nn.py"
        with open(temp_file, 'w') as f:
            f.write(code)

        code_lines = code.split('\n')

        def get_context(lineno, window=5):
            """Get numbered code lines around the error."""
            start = max(0, lineno - window - 1)
            end = min(len(code_lines), lineno + window)
            context_lines = []
            for i in range(start, end):
                marker = " >> " if i == lineno - 1 else "    "
                context_lines.append(f"{marker}{i+1:4d} | {code_lines[i]}")
            return '\n'.join(context_lines)

        # Stage 1: Syntax check
        try:
            py_compile.compile(str(temp_file), doraise=True)
        except SyntaxError as e:
            # Specific handling for IndentationError
            lineno = e.lineno if hasattr(e, 'lineno') else None
            error_type = 'IndentationError' if 'IndentationError' in str(type(e)) or 'indent' in str(e).lower() else 'SyntaxError'
            return {
                'stage': 'syntax',
                'error': error_type,
                'line': lineno,
                'message': str(e),
                'context': get_context(lineno) if lineno else ''
            }
        except py_compile.PyCompileError as e:
            # Extract line number from the error
            lineno = None
            msg = str(e)
            # PyCompileError includes line info like "(temp_nn.py, line 144)"
            if 'line ' in msg:
                try:
                    lineno = int(msg.split('line ')[1].split(')')[0].split(',')[0])
                except (ValueError, IndexError):
                    pass
            return {
                'stage': 'syntax',
                'error': 'SyntaxError',
                'line': lineno,
                'message': msg,
                'context': get_context(lineno) if lineno else ''
            }

        # Stage 2: Import check
        try:
            spec = importlib.util.spec_from_file_location("temp_validate", str(temp_file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            lineno = None
            tb = traceback.format_exc()
            # Try to extract line number from traceback
            for tb_line in reversed(tb.split('\n')):
                if 'line ' in tb_line and 'temp_nn.py' in tb_line:
                    try:
                        lineno = int(tb_line.split('line ')[1].split(',')[0].split('\n')[0])
                    except (ValueError, IndexError):
                        pass
                    break
            return {
                'stage': 'import',
                'error': type(e).__name__,
                'line': lineno,
                'message': str(e),
                'context': get_context(lineno) if lineno else '',
                'traceback': tb
            }

        # Stage 3: API check
        if not hasattr(module, 'Net'):
            return {
                'stage': 'api',
                'error': 'MissingClass',
                'line': None,
                'message': "Missing 'Net' class. The main model class MUST be named 'Net'.",
                'context': ''
            }
        if not hasattr(module, 'supported_hyperparameters'):
            return {
                'stage': 'api',
                'error': 'MissingFunction',
                'line': None,
                'message': "Missing 'supported_hyperparameters()' function at module level (outside Net class).",
                'context': ''
            }

        # Stage 4: Instantiation
        try:
            supported = module.supported_hyperparameters()
            prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.1,
                    'stochastic_depth_prob': 0.1, 'norm_eps': 1e-5, 'norm_std': 0.02}
            prm = {k: v for k, v in prm.items() if k in supported}
            model = module.Net(
                in_shape=(1, 3, 32, 32),
                out_shape=(10,),
                prm=prm,
                device=self.device
            )
        except Exception as e:
            lineno = None
            tb = traceback.format_exc()
            for tb_line in reversed(tb.split('\n')):
                if 'line ' in tb_line and 'temp_nn.py' in tb_line:
                    try:
                        lineno = int(tb_line.split('line ')[1].split(',')[0].split('\n')[0])
                    except (ValueError, IndexError):
                        pass
                    break
            return {
                'stage': 'instantiation',
                'error': type(e).__name__,
                'line': lineno,
                'message': str(e),
                'context': get_context(lineno) if lineno else '',
                'traceback': tb
            }

        # Stage 4.5: Device placement check (CRITICAL)
        try:
            # Check that all parameters are on the same device
            model_device = next(model.parameters()).device
            mismatched_modules = []

            for name, param in model.named_parameters():
                if param.device != model_device:
                    mismatched_modules.append(f"{name} is on {param.device}")

            if mismatched_modules:
                # Find the Net.__init__ method to give line context
                net_init_line = None
                for i, line in enumerate(code_lines):
                    if 'class Net(' in line:
                        for j in range(i, min(i+20, len(code_lines))):
                            if 'def __init__' in code_lines[j]:
                                net_init_line = j + 1
                                break
                        break

                return {
                    'stage': 'device_check',
                    'error': 'DeviceMismatch',
                    'line': net_init_line,
                    'message': f"Model has parameters on different devices! Expected all on {model_device}. "
                              f"Mismatched: {', '.join(mismatched_modules[:3])}. "
                              f"FIX: In Net.__init__, add .to(self.device) after creating self.experts and self.gate: "
                              f"self.experts = nn.ModuleList([...]).to(self.device) and self.gate = nn.Sequential(...).to(self.device)",
                    'context': get_context(net_init_line) if net_init_line else ''
                }
        except Exception:
            # If device check fails, continue (might be a different error that forward pass will catch)
            pass

        # Stage 5: Forward pass
        try:
            model.eval()
            with torch.no_grad():
                x = torch.randn(2, 3, 32, 32).to(self.device)
                out = model(x)
            if out.shape != (2, 10):
                return {
                    'stage': 'forward',
                    'error': 'ShapeMismatch',
                    'line': None,
                    'message': f"Expected output shape (2, 10), got {tuple(out.shape)}. Check forward() method and expert output dimensions.",
                    'context': ''
                }
        except Exception as e:
            lineno = None
            tb = traceback.format_exc()
            for tb_line in reversed(tb.split('\n')):
                if 'line ' in tb_line and 'temp_nn.py' in tb_line:
                    try:
                        lineno = int(tb_line.split('line ')[1].split(',')[0].split('\n')[0])
                    except (ValueError, IndexError):
                        pass
                    break
            return {
                'stage': 'forward',
                'error': type(e).__name__,
                'line': lineno,
                'message': str(e),
                'context': get_context(lineno) if lineno else '',
                'traceback': tb
            }
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # All stages passed
        return None

    def create_repair_prompt(self, code: str, error: dict, expert_names: list) -> str:
        """Build a focused repair prompt with error details for DeepSeek."""
        # Build expert examples using ACTUAL expert names from combination
        num_experts = len(expert_names)
        expert_list = ', '.join([f"{name}(...)" for name in expert_names])
        expert_mentions = ', '.join(expert_names)
        error_section = f"ERROR at stage '{error['stage']}': {error['error']}: {error['message']}"

        if error.get('line'):
            error_section += f"\nError at LINE {error['line']}"

        context_section = ""
        if error.get('context'):
            context_section = f"\n\nCODE CONTEXT around the error:\n{error['context']}"

        traceback_section = ""
        if error.get('traceback'):
            # Only include last few lines of traceback (most relevant)
            tb_lines = error['traceback'].strip().split('\n')
            tb_short = '\n'.join(tb_lines[-8:])
            traceback_section = f"\n\nTRACEBACK:\n{tb_short}"

        prompt = f"""The following Python neural network code has an error. Fix it and return the COMPLETE corrected code.

{error_section}{context_section}{traceback_section}

⚠️ CRITICAL DEVICE PLACEMENT RULES (MOST COMMON ERROR):

RULE 1 - In Net.__init__, after creating self.experts and self.gate, add .to(self.device):
  self.experts = nn.ModuleList([{expert_list}]).to(self.device)
  self.gate = nn.Sequential(Conv2d, ReLU, AdaptiveAvgPool2d, Flatten, Linear, output_size={num_experts}).to(self.device)

RULE 2 - NO .to(device) inside expert classes ({expert_mentions})
RULE 3 - NO .to(device) in forward(), train_setup(), or learn()
RULE 4 - If error is "Input type (cuda) and weight type (cpu) should be the same":
   → You forgot .to(self.device) on self.experts or self.gate in Net.__init__

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON ERRORS TO AVOID:

1. KeyError for hyperparameters: NEVER use prm['num_experts'] or any key without checking it exists
   → Either add it to supported_hyperparameters() OR hardcode the value (e.g., num_experts = 4)

2. IndentationError: Ensure ALL code blocks are properly indented (4 spaces per level)

3. Incomplete lines: NEVER leave statements incomplete (e.g., "model.load_state_dict(weight" without closing paren)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OTHER CONSTRAINTS:

- Input: (batch, 3, 32, 32) for CIFAR-10, Output: (batch, 10)
- in_shape parameter is (1, 3, 32, 32) where in_shape[1]=3 (channels)
- Net.__init__ signature: def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device)
- Each expert MUST end with: AdaptiveAvgPool2d((1,1)) → Flatten → Linear(channels, 10)
- supported_hyperparameters() is MODULE-LEVEL function (outside all classes)
- Helper classes/functions BEFORE they are used:
  * _make_divisible (if needed)
  * Conv2dNormActivation
  * LayerNorm2d, CNBlockConfig, etc.
  * supported_hyperparameters()
  * Expert classes
  * Net (LAST)
- Import ALL needed: from torch import Tensor, from functools import partial, from collections import OrderedDict
- Conv2dNormActivation MUST include kernel_size: Conv2dNormActivation(in, out, kernel_size=3, ...)
- Use layers.extend([a, b, c]) NOT layers.append(a, b, c)
- Gating network needs AdaptiveAvgPool2d((1,1)) before Flatten/Linear
- ONLY use torch/torch.nn/torch.nn.functional (NO torchvision.models, NO external deps)
- AlexNet for CIFAR-10: kernel_size=3, stride=1 (NOT 11/4)
- Dense MoE forward: gate_weights = self.gate(x); outputs = stack([e(x) for e in experts]); sum(outputs * weights)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 SHAPE DEBUGGING GUIDE (For tensor dimension errors):

If error mentions tensor shape mismatch, verify these shapes in Net.forward():
  1. Input x: (batch, 3, 32, 32)
  2. Gate output MUST be: (batch, {num_experts}) - NOT (batch, 10)!
     → gate = Sequential(..., AdaptiveAvgPool2d((1,1)), Flatten(), Linear(channels, {num_experts}))
  3. Expert output MUST be: (batch, 10) - the final classification
     → expert = Sequential(..., AdaptiveAvgPool2d((1,1)), Flatten(), Linear(channels, 10))
  4. After stacking experts: (batch, {num_experts}, 10)
     → expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
  5. Gate weights reshaped: (batch, {num_experts}, 1)
     → gate_weights = self.gate(x).unsqueeze(2)
  6. Final output: (batch, 10)
     → output = torch.sum(expert_outputs * gate_weights, dim=1)

Common shape errors:
  - "size of tensor a (10) must match size of tensor b (2)" → Gate outputting 10 instead of {num_experts}
  - "negative dimension -2" → Shape calculation error, check all Conv2d/Linear layer dimensions
  - "expected 3 channels, got 64" → Expert input expects (batch, 3, 32, 32), not feature maps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FULL CODE TO FIX:
<nn>
{code}
</nn>

Return the COMPLETE fixed code inside <nn> tags. DO NOT include numbered lists, comments about changes, or error messages - ONLY return the fixed code."""

        return prompt

    def repair_code(self, code: str, error: dict, expert_names: list) -> str:
        """Send error details to DeepSeek and get fixed code back."""
        repair_prompt = self.create_repair_prompt(code, error, expert_names)
        fixed_code = self.generate_code(repair_prompt)
        return fixed_code

    def validate_and_train(self, code: str, expert_names: list, temp_dir: Path) -> dict:
        """Try to train model for 1 epoch. Returns results dict or None"""
        try:
            # Save code to temp file
            temp_file = temp_dir / "temp_nn.py"
            with open(temp_file, 'w') as f:
                f.write(code)

            # Try to import
            spec = importlib.util.spec_from_file_location("temp_model", temp_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["temp_model"] = module
            spec.loader.exec_module(module)

            # Check required components
            if not hasattr(module, 'Net'):
                raise Exception("Missing Net class")
            if not hasattr(module, 'supported_hyperparameters'):
                raise Exception("Missing supported_hyperparameters()")

            # Get hyperparameters
            supported = module.supported_hyperparameters()
            prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.1,
                    'stochastic_depth_prob': 0.1, 'norm_eps': 1e-5, 'norm_std': 0.02}
            prm = {k: v for k, v in prm.items() if k in supported}

            # Instantiate model
            model = module.Net(
                in_shape=(1, 3, 32, 32),
                out_shape=(10,),
                prm=prm,
                device=self.device
            )
            model.train_setup(prm)

            num_params = sum(p.numel() for p in model.parameters())

            # Train for 1 epoch
            print(f"    🏋️  Training (1 epoch)...", end=" ", flush=True)
            train_start = time.time()

            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            if hasattr(model, 'learn'):
                # Use model's learn method
                loss = model.learn(self.train_loader)
                train_time = time.time() - train_start

                # Sample accuracy
                model.eval()
                with torch.no_grad():
                    for inputs, labels in self.train_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        if total >= 5000:
                            break
                train_acc = 100.0 * correct / total if total > 0 else 0.0
                train_loss = loss

            else:
                # Manual training loop
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    model.optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = model.criteria[0](outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 3)
                    model.optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                train_time = time.time() - train_start
                train_loss = total_loss / len(self.train_loader)
                train_acc = 100.0 * correct / total

            print(f"✅ ({train_time:.1f}s, Acc: {train_acc:.1f}%)")

            # Evaluate on test set
            print(f"    🧪 Testing...", end=" ", flush=True)
            test_start = time.time()

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_time = time.time() - test_start
            test_acc = 100.0 * correct / total
            test_loss = test_loss / len(self.test_loader)

            print(f"✅ Test Acc: {test_acc:.2f}%")

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'experts': expert_names,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'num_parameters': num_params,
                'train_time': train_time,
                'test_time': test_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ {error_msg}")

            # In debug mode, show first few lines of code
            if hasattr(self, 'debug') and self.debug:
                print(f"    📄 First 5 lines of generated code:")
                code_lines = code.split('\n')[:5] if code else []
                for line in code_lines:
                    print(f"       {line[:80]}")  # Truncate long lines

            # traceback.print_exc()  # Uncomment for full traceback
            return None

    def process_combination(self, combination: dict, index: int, total: int):
        """Process one expert combination"""
        expert_names = combination['names']
        expert_files = combination['files']

        print(f"\n[{index+1}/{total}] {' + '.join(expert_names)}")

        self.attempted += 1

        # Create temp directory
        temp_dir = self.epoch_dir / f"temp_{index}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Generate code
            print(f"    🔄 Generating...", end=" ", flush=True)
            gen_start = time.time()
            prompt = self.create_prompt(expert_files, expert_names)
            code = self.generate_code(prompt)
            gen_time = time.time() - gen_start

            if not code:
                print(f"❌ Failed to generate code")
                self.failed += 1
                shutil.rmtree(temp_dir)
                return

            print(f"✅ ({gen_time:.1f}s)")

            # Step 2: Validate → Repair loop (includes expert class check)
            results = None
            repair_attempts = 0

            for attempt in range(self.max_retries + 1):
                # First check: Do all expert classes exist?
                missing_experts = []
                for expert_name in expert_names:
                    if f"class {expert_name}(" not in code and f"class {expert_name}Expert(" not in code:
                        missing_experts.append(expert_name)

                if missing_experts:
                    error = {
                        'stage': 'api',
                        'error': 'MissingExpertClass',
                        'line': None,
                        'message': f"Generated code is missing expert classes: {', '.join(missing_experts)}. All {len(expert_names)} expert classes must be defined: {', '.join(expert_names)}.",
                        'context': ''
                    }
                else:
                    # Standard validation (syntax, import, instantiation, forward pass)
                    error = self.validate_code(code, temp_dir)

                if error is None:
                    # Code is valid! Now do full training
                    results = self.validate_and_train(code, expert_names, temp_dir)
                    if results:
                        break  # Success!
                    else:
                        # Training failed at runtime - can't easily repair
                        print(f"    ❌ Training failed at runtime")
                        break
                else:
                    # Code has errors
                    print(f"    ❌ Validation failed at stage '{error['stage']}': {error['error']}: {error['message'][:100]}")

                    if attempt < self.max_retries:
                        repair_attempts += 1

                        # Classify error: complex errors need regeneration, not patching
                        complex_errors = (
                            error['stage'] == 'syntax' or  # Syntax errors hard to patch
                            (error['stage'] == 'import' and 'NameError' in error['error'] and 'class' in error['message'].lower()) or  # Missing class definition
                            (error['stage'] == 'instantiation' and 'not defined' in error['message'])  # Missing class reference
                        )

                        if complex_errors and attempt == 0:
                            # For first attempt on complex errors, regenerate fresh
                            print(f"    🔄 Regenerating (complex error, fresh start)...", end=" ", flush=True)
                            repair_start = time.time()
                            code = self.generate_code(prompt)  # Fresh regeneration
                            repair_time = time.time() - repair_start
                            if code:
                                print(f"regenerated ({repair_time:.1f}s)")
                            else:
                                print(f"regeneration failed")
                                break
                        else:
                            # Try patching for simple errors or subsequent attempts
                            print(f"    🔧 Repair attempt {repair_attempts}/{self.max_retries}...", end=" ", flush=True)
                            repair_start = time.time()
                            fixed_code = self.repair_code(code, error, expert_names)
                            repair_time = time.time() - repair_start

                            if fixed_code:
                                # Validate repair is better (doesn't introduce new errors)
                                new_error = self.validate_code(fixed_code, temp_dir)

                                # Accept if: no error, or error is at later stage, or same stage but different error
                                stage_order = ['syntax', 'import', 'api', 'instantiation', 'device_check', 'forward']
                                current_stage_idx = stage_order.index(error['stage']) if error['stage'] in stage_order else -1
                                new_stage_idx = stage_order.index(new_error['stage']) if new_error and new_error['stage'] in stage_order else 999

                                if new_error is None or new_stage_idx > current_stage_idx:
                                    print(f"accepted ({repair_time:.1f}s)")
                                    code = fixed_code
                                else:
                                    print(f"rejected (no improvement) ({repair_time:.1f}s)")
                                    # Keep old code, will retry with different strategy next attempt
                            else:
                                print(f"repair generation failed")
                                break

                        # Save repair attempt in debug mode
                        if self.debug:
                            repair_debug_dir = self.epoch_dir / "debug" / f"repair_{index}"
                            repair_debug_dir.mkdir(parents=True, exist_ok=True)
                            with open(repair_debug_dir / f"attempt_{repair_attempts}.py", 'w') as f:
                                f.write(code)
                            with open(repair_debug_dir / f"error_{repair_attempts}.json", 'w') as f:
                                json.dump({k: v for k, v in error.items() if k != 'traceback'}, f, indent=2)
                        continue
                    else:
                        print(f"    ❌ Max retries ({self.max_retries}) exhausted")
                        break

            if results:
                # SUCCESS! Save everything
                success_dir = self.epoch_dir / f"B{self.success_counter}"
                success_dir.mkdir(exist_ok=True)

                # Save code
                code_file = success_dir / "new_nn.py"
                with open(code_file, 'w') as f:
                    f.write(code)

                # Save results
                results['model_id'] = f"B{self.success_counter}"
                results['generation_time'] = gen_time
                results['repair_attempts'] = repair_attempts

                results_file = success_dir / "results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

                if repair_attempts > 0:
                    print(f"    💾 Saved to B{self.success_counter}/ (repaired after {repair_attempts} attempt(s))")
                    self.repaired += 1
                else:
                    print(f"    💾 Saved to B{self.success_counter}/")

                self.results.append(results)
                self.success_counter += 1
                self.successful += 1

            else:
                # FAILED - discard (but save if debug mode)
                if self.debug:
                    debug_dir = self.epoch_dir / "debug" / f"failed_{index}"
                    debug_dir.mkdir(parents=True, exist_ok=True)

                    # Save failed code
                    with open(debug_dir / "generated_code.py", 'w') as f:
                        f.write(code)

                    # Save metadata
                    with open(debug_dir / "info.json", 'w') as f:
                        json.dump({
                            'experts': expert_names,
                            'index': index,
                            'generation_time': gen_time,
                            'repair_attempts': repair_attempts,
                            'last_error': {k: v for k, v in error.items() if k != 'traceback'} if error else None
                        }, f, indent=2)

                    print(f"    🗑️  Discarded after {repair_attempts} repair(s) (saved to debug/failed_{index}/)")
                else:
                    print(f"    🗑️  Discarded (failed after {repair_attempts} repair attempt(s))")

                self.failed += 1

        finally:
            # Clean up temp
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def save_aggregate_results(self):
        """Save aggregate CSV and summary"""
        if not self.results:
            print("\n⚠️  No successful models to save")
            return

        # Save CSV
        df = pd.DataFrame(self.results)
        csv_file = self.epoch_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Saved aggregate results to {csv_file}")

        # Save summary
        summary_file = self.epoch_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MoE PIPELINE SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Attempted:    {self.attempted}\n")
            f.write(f"Successful:   {self.successful}\n")
            f.write(f"Repaired:     {self.repaired}\n")
            f.write(f"Failed:       {self.failed}\n")
            f.write(f"Success Rate: {100.0*self.successful/self.attempted:.1f}%\n\n")

            if self.successful > 0:
                best_idx = df['test_acc'].idxmax()
                best = df.loc[best_idx]
                f.write(f"Best Model:   {best['model_id']}\n")
                f.write(f"Experts:      {' + '.join(best['experts'])}\n")
                f.write(f"Test Acc:     {best['test_acc']:.2f}%\n\n")

                f.write(f"Average Test Acc: {df['test_acc'].mean():.2f}% ± {df['test_acc'].std():.2f}%\n")
                f.write(f"Average Params:   {df['num_parameters'].mean():.0f}\n")

    def run(self, combinations: list):
        """Run complete pipeline"""
        print("\n" + "=" * 70)
        print("🔬 INTEGRATED MOE PIPELINE - SUCCESS ONLY MODE")
        print("=" * 70)
        print(f"📁 Output: {self.epoch_dir}")
        print(f"🎯 Combinations: {len(combinations)}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🔧 Auto-repair: up to {self.max_retries} retries per combination")
        print("=" * 70)

        # Process each combination
        for i, combo in enumerate(combinations):
            self.process_combination(combo, i, len(combinations))

        # Save results
        self.save_aggregate_results()

        # Final summary
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"Attempted:    {self.attempted}")
        print(f"Successful:   {self.successful}")
        print(f"Repaired:     {self.repaired}")
        print(f"Failed:       {self.failed}")
        print(f"Success Rate: {100.0*self.successful/self.attempted if self.attempted > 0 else 0:.1f}%")

        if self.successful > 0:
            df = pd.DataFrame(self.results)
            best_idx = df['test_acc'].idxmax()
            best = df.loc[best_idx]
            print(f"\nBest Model:   {best['model_id']} (Test Acc: {best['test_acc']:.2f}%)")
            print(f"Results:      {self.epoch_dir}/results.csv")

        print("=" * 70)


def generate_random_combinations(nn_dir: str, num_combinations: int = 10, num_experts: int = 2, filter_classification: bool = True) -> list:
    """
    Generate random expert combinations from available architectures in nn_dir.

    Args:
        nn_dir: Path to directory containing expert .py files
        num_combinations: Number of random combinations to generate
        num_experts: Number of experts per combination (default: 2)
        filter_classification: Only use classification architectures (default: True)

    Returns:
        List of dicts with format: [{"names": ["Expert1", "Expert2"], "files": ["path1.py", "path2.py"]}, ...]
    """
    import glob
    import random
    from pathlib import Path
    from collections import defaultdict

    print(f"🔍 Scanning {nn_dir} for available architectures...")

    # Classification-only whitelist (known good architectures for CIFAR-10)
    classification_whitelist = {
        # Simple & Reliable
        'AlexNet', 'VGG', 'SqueezeNet',
        # Residual Networks
        'ResNet', 'ResNeXt', 'WideResNet',
        # Efficient Networks
        'MobileNet', 'MobileNetV2', 'MobileNetV3', 'ShuffleNet', 'ShuffleNetV2',
        'EfficientNet', 'EfficientNetV2',
        # Dense & Inception
        'DenseNet', 'GoogLeNet', 'InceptionV3', 'InceptionV4',
        # Modern CNNs
        'RegNet', 'ConvNeXt', 'DarkNet', 'DPN68', 'DPN92', 'DPN98', 'DPN107', 'DPN131',
        # Specialty
        'AirNet', 'AirNext', 'BagNet',
        # Custom variants
        'C10C', 'C5C', 'C8C'
    }

    # Blacklist (non-classification tasks)
    blacklist = {
        # Segmentation
        'UNet', 'FCN', 'DeepLabV3',
        # Object Detection
        'SSD', 'FCOS', 'FasterRCNN', 'RetinaNet', 'YOLO',
        # Generative
        'VAE', 'Diffusion', 'DenoiseUNet', 'GAN', 'Diffuser',
        # Vision-Language
        'GIT', 'Blip2', 'CLIP',
        # Transformers (too complex)
        'Swin', 'ViT', 'DeiT', 'BEiT',
        # Other
        'RNN', 'LSTM', '__init__', 'MoEv2', 'Bayesian', 'Complex', 'Fractal'
    }

    # Get all .py files
    all_files = glob.glob(f"{nn_dir}/*.py")

    # Group files by architecture name (prefix before first dash or .py)
    arch_files = defaultdict(list)
    for file_path in all_files:
        filename = Path(file_path).stem  # Remove .py extension

        # Extract architecture name (before first dash or UUID)
        if '-' in filename:
            arch_name = filename.split('-')[0]
        else:
            arch_name = filename

        # Skip files with hash-like names (no clear architecture)
        if len(arch_name) == 32 or arch_name.startswith('alt'):
            continue

        # Apply filters
        if filter_classification:
            # Skip blacklisted architectures
            if any(bad in arch_name for bad in blacklist):
                continue
            # Only include whitelisted
            if arch_name not in classification_whitelist:
                continue

        arch_files[arch_name].append(file_path)

    # Filter architectures that have at least one implementation
    available_archs = {arch: files for arch, files in arch_files.items() if len(files) > 0}

    if filter_classification:
        print(f"✅ Found {len(available_archs)} classification architectures: {', '.join(sorted(available_archs.keys())[:10])}{'...' if len(available_archs) > 10 else ''}")
    else:
        print(f"✅ Found {len(available_archs)} architectures: {', '.join(sorted(available_archs.keys())[:10])}{'...' if len(available_archs) > 10 else ''}")
    print(f"📦 Total expert files: {sum(len(files) for files in available_archs.values())}")

    if len(available_archs) < num_experts:
        raise ValueError(f"Not enough architectures ({len(available_archs)}) to create {num_experts}-expert combinations")

    # Generate random combinations (with duplicate detection and HARD+HARD filtering)
    combinations = []
    seen_combinations = set()  # Track combinations to avoid duplicates
    arch_names = list(available_archs.keys())

    # Define HARD architectures (low success rate when paired together)
    hard_archs = {'GoogLeNet', 'InceptionV3', 'InceptionV4', 'EfficientNet', 'EfficientNetV2',
                  'DPN68', 'DPN92', 'DPN98', 'DPN107', 'DPN131', 'DarkNet', 'ConvNeXt', 'RegNet'}

    attempts = 0
    max_attempts = num_combinations * 100  # Prevent infinite loop

    while len(combinations) < num_combinations and attempts < max_attempts:
        attempts += 1

        # Randomly select architectures (without replacement within combination)
        selected_archs = random.sample(arch_names, num_experts)

        # Create a sorted tuple for duplicate detection (order-independent)
        combo_key = tuple(sorted(selected_archs))

        # Skip if duplicate
        if combo_key in seen_combinations:
            continue

        # Skip if both are HARD (low success rate) - only when filtering is enabled
        if filter_classification and all(arch in hard_archs for arch in selected_archs):
            continue

        # Accept this combination
        seen_combinations.add(combo_key)

        # For each architecture, randomly pick one file
        selected_files = [random.choice(available_archs[arch]) for arch in selected_archs]

        combinations.append({
            "names": selected_archs,
            "files": selected_files
        })

        print(f"  [{len(combinations)}/{num_combinations}] {' + '.join(selected_archs)}")

    if len(combinations) < num_combinations:
        print(f"\n⚠️  Warning: Only generated {len(combinations)} unique combinations (requested {num_combinations})")

    return combinations


def main():
    parser = argparse.ArgumentParser(description='Integrated MoE Pipeline')
    parser.add_argument('--combinations', type=str, required=False,
                       help='JSON file with expert combinations (mutually exclusive with --random)')
    parser.add_argument('--random', action='store_true',
                       help='Generate random combinations from nn_dir instead of using JSON file')
    parser.add_argument('--num_combinations', type=int, default=10,
                       help='Number of random combinations to generate (only with --random, default: 10)')
    parser.add_argument('--num_experts', type=int, default=2,
                       help='Number of experts per combination (only with --random, default: 2)')
    parser.add_argument('--nn_dir', type=str,
                       default='/home/yashkumarlukhi/cvpraktikumss25/nn-dataset/ab/nn/nn',
                       help='Path to nn/ directory')
    parser.add_argument('--output_dir', type=str,
                       default='out/nngpt/moe_combine',
                       help='Output directory')
    parser.add_argument('--model', type=str,
                       default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                       help='HuggingFace model')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Sampling temperature (lower = more deterministic, default: 0.2)')
    parser.add_argument('--max_tokens', type=int, default=32768,
                       help='Max tokens to generate (32K for DeepSeek-R1 reasoning)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--debug', action='store_true',
                       help='Save failed generations for debugging')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Max repair attempts per combination (DeepSeek fixes its own errors)')
    parser.add_argument('--filter_classification', action='store_true', default=True,
                       help='Filter to only classification architectures (default: True). Use --no-filter_classification to disable')
    parser.add_argument('--no-filter_classification', dest='filter_classification', action='store_false',
                       help='Disable classification filtering (allow all architectures)')

    args = parser.parse_args()

    # Validate arguments
    if args.random and args.combinations:
        parser.error("--random and --combinations are mutually exclusive. Choose one.")
    if not args.random and not args.combinations:
        parser.error("Either --combinations <file> or --random must be specified.")

    # Load or generate combinations
    if args.random:
        print(f"\n{'='*70}")
        print(f"🎲 RANDOM COMBINATION MODE {'(Classification Only)' if args.filter_classification else '(All Architectures)'}")
        print(f"{'='*70}\n")
        combinations = generate_random_combinations(
            nn_dir=args.nn_dir,
            num_combinations=args.num_combinations,
            num_experts=args.num_experts,
            filter_classification=args.filter_classification
        )
    else:
        print(f"\n{'='*70}")
        print(f"📁 LOADING COMBINATIONS FROM: {args.combinations}")
        print(f"{'='*70}\n")
        with open(args.combinations, 'r') as f:
            combinations = json.load(f)

    # Run pipeline
    pipeline = IntegratedMoEPipeline(
        nn_dir=args.nn_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        epoch=args.epoch,
        debug=args.debug,
        max_retries=args.max_retries
    )

    pipeline.run(combinations)


if __name__ == '__main__':
    main()

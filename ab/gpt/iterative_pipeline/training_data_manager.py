#!/usr/bin/env python3
"""
Training Data Manager for Iterative Fine-Tuning

Manages the augmentation of training data with successful generated models.
Converts model code to chat examples and maintains cumulative training datasets.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ab.nn.util.Const import out_dir


class TrainingDataManager:
    """Manage training data augmentation for iterative fine-tuning."""
    
    def __init__(self, base_data_dir: str):
        """
        Initialize training data manager.
        
        Args:
            base_data_dir: Path to curation_output/chat_data directory (train.jsonl, dev.jsonl, test.jsonl)
        """
        self.base_data_dir = Path(base_data_dir)
        
        # Standard filenames in curation_output/chat_data/
        self.train_file = self.base_data_dir / "train.jsonl"
        self.dev_file = self.base_data_dir / "dev.jsonl"
        self.test_file = self.base_data_dir / "test.jsonl"
        
        # Verify base files exist
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training data not found: {self.train_file}")
        
        # Load original training data
        self.original_train_data = self._load_jsonl(self.train_file)
        print(f"[INFO] Loaded {len(self.original_train_data)} original training examples from {self.train_file}")
        
        # Load test data to extract prompt template (used during generation)
        self.test_data = self._load_jsonl(self.test_file)
        if not self.test_data:
            raise FileNotFoundError(f"Test data not found: {self.test_file}. Cannot extract prompt template.")
        
        # Extract system and user prompt template from first test example
        first_test = self.test_data[0]
        if "messages" not in first_test or len(first_test["messages"]) < 2:
            raise ValueError(f"Invalid test data format in {self.test_file}. Expected 'messages' with at least 2 entries.")
        
        self.system_template = first_test["messages"][0]["content"]
        self.user_template = first_test["messages"][1]["content"]
        print(f"[INFO] Loaded prompt template from {self.test_file}")
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        if not path.exists():
            return data
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Failed to parse line {line_num} in {path}: {e}")
        
        return data
    
    def _save_jsonl(self, data: List[Dict[str, Any]], path: Path):
        """Save JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        def convert_paths(obj):
            """Recursively convert Path objects to strings for JSON serialization."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                # Convert any Path objects to strings before JSON serialization
                item_serializable = convert_paths(item)
                f.write(json.dumps(item_serializable, ensure_ascii=False) + '\n')
    
    def convert_code_to_chat_example(
        self,
        code: str,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert generated model code to a chat training example.
        
        Args:
            code: Python code string
            model_id: Unique identifier for this model
            metadata: Additional metadata (accuracy, params, etc.)
        
        Returns:
            Chat example in the same format as training data
        """
        # Extract system, user, and assistant messages from the code structure
        # The training data format is:
        # {
        #   "messages": [
        #     {"role": "system", "content": "<system policy>"},
        #     {"role": "user", "content": "<task description>"},
        #     {"role": "assistant", "content": "```python\n<code>\n```"}
        #   ]
        # }
        
        # Use the same system and user prompt template from test.jsonl
        # This ensures consistency with the prompts used during generation
        system_msg = self.system_template
        
        # Use the user template from test.jsonl, but enhance it with learned requirements
        accuracy = metadata.get("accuracy", 0.0)
        params = metadata.get("params", 0)
        params_limit = int(params * 1.5)
        
        import re
        # Replace params limit in the template
        user_content = re.sub(
            r'params\s*[≤<=]+\s*[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?',
            f'params ≤ {params_limit}',
            self.user_template
        )
        
        # Enhance the REQUIRED FORMAT section with additional requirements we've learned
        additional_requirements = (
            "\n"
            "- REQUIRED METHODS in Net class: `train_setup(self, prm)` and `learn(self, train_data)`\n"
            "- REQUIRED FUNCTION (outside class): `def supported_hyperparameters(): return {'lr', 'momentum'}`\n"
            "- REQUIRED IMPORTS (at the start): `import torch` and `import torch.nn as nn`\n"
            "- If using functional API (F.relu, F.max_pool2d, etc.), MUST import: `import torch.nn.functional as F`\n"
            "- PREFER module versions: `nn.ReLU()` instead of `F.relu()`, `nn.MaxPool2d()` instead of `F.max_pool2d()`, `nn.AdaptiveAvgPool2d((1, 1))` instead of `F.adaptive_avg_pool2d()`\n"
            "- NO helper classes (only Net class), NO duplicate class or function definitions"
        )
        
        # Insert additional requirements after the existing REQUIRED FORMAT section
        if "**REQUIRED FORMAT**" in user_content:
            # Find the end of REQUIRED FORMAT section (before PRIMARY OBJECTIVE or next section)
            required_format_end = user_content.find("**PRIMARY OBJECTIVE**")
            if required_format_end == -1:
                required_format_end = user_content.find("**UNIQUENESS REQUIREMENT**")
            if required_format_end == -1:
                required_format_end = user_content.find("**CRITICAL NOVELTY CONSTRAINT**")
            if required_format_end == -1:
                # If no section found, append at the end of REQUIRED FORMAT
                required_format_end = user_content.find("\n\n", user_content.find("**REQUIRED FORMAT**"))
            
            if required_format_end != -1:
                user_content = user_content[:required_format_end] + additional_requirements + "\n\n" + user_content[required_format_end:]
            else:
                # Fallback: append after REQUIRED FORMAT
                user_content = user_content.replace("**REQUIRED FORMAT**", "**REQUIRED FORMAT**" + additional_requirements)
        
        # Add reference accuracy if not already present
        if "Reference" not in user_content and "Similar models" not in user_content:
            # Insert after the goal line if it exists
            if "**PRIMARY OBJECTIVE**" in user_content:
                user_content = user_content.replace(
                    "**PRIMARY OBJECTIVE**",
                    f"**REFERENCE**: Similar models achieve {accuracy*100:.2f}%+ on first epoch.\n\n**PRIMARY OBJECTIVE**"
                )
        
        assistant_content = f"```python\\n{code.strip()}\\n```"
        
        # Match original format: use "id" and "meta" instead of "metadata"
        # Original format: {"id": "...", "messages": [...], "meta": {...}}
        chat_example = {
            "id": str(uuid.uuid4()),
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "meta": {
                "source_path": out_dir / 'iterative_cycles' / f"cycle_{metadata.get('cycle', 0)}/nneval/{model_id}/new_nn.py",
                "has_module": True,
                "family": "generated",
                "param_estimate": params,
                "class_names": ["Net"] if "class Net" in code else [],
                "calls": [],  # Can be populated if needed
                "type": "full",
                # Additional metadata for tracking
                "generated": True,
                "model_id": model_id,
                "cycle": metadata.get("cycle", 0),
                "parent_checkpoint": metadata.get("checkpoint", "unknown"),
                "first_epoch_accuracy": accuracy,
            }
        }
        
        return chat_example
    
    def augment_training_data(
        self,
        new_examples: List[Dict[str, Any]],
        cycle: int,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Augment training data with new successful models.
        
        Args:
            new_examples: List of chat examples to add
            cycle: Current cycle number
            output_dir: Output directory for augmented data
        
        Returns:
            Statistics about the augmentation
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing training data (cumulative from previous cycles)
        cycle_data_dir = output_dir / f"chat_data_cycle_{cycle}"
        cycle_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Start with original + all previous cycle additions
        if cycle == 1:
            # First cycle: start with original data
            augmented_train_data = self.original_train_data.copy()
        else:
            # Load from previous cycle
            prev_cycle_train = output_dir / f"chat_data_cycle_{cycle-1}" / "train.jsonl"
            if prev_cycle_train.exists():
                augmented_train_data = self._load_jsonl(prev_cycle_train)
            else:
                augmented_train_data = self.original_train_data.copy()
        
        # Add new examples
        original_count = len(augmented_train_data)
        augmented_train_data.extend(new_examples)
        
        # Save augmented train.jsonl
        train_output = cycle_data_dir / "train.jsonl"
        self._save_jsonl(augmented_train_data, train_output)
        
        # Copy dev.jsonl and test.jsonl unchanged
        if self.dev_file.exists():
            shutil.copy(self.dev_file, cycle_data_dir / "dev.jsonl")
        
        if self.test_file.exists():
            shutil.copy(self.test_file, cycle_data_dir / "test.jsonl")
        
        stats = {
            "cycle": cycle,
            "original_examples": len(self.original_train_data),
            "previous_total": original_count,
            "new_examples_added": len(new_examples),
            "total_examples": len(augmented_train_data),
            "growth_rate": len(new_examples) / max(1, original_count),
            "output_dir": str(cycle_data_dir),
        }
        
        print(f"[INFO] Cycle {cycle} training data:")
        print(f"  - Original examples: {stats['original_examples']}")
        print(f"  - Previous total: {stats['previous_total']}")
        print(f"  - New examples added: {stats['new_examples_added']}")
        print(f"  - Total examples: {stats['total_examples']}")
        print(f"  - Output directory: {stats['output_dir']}")
        
        return stats
    
    def get_training_data_dir(self, cycle: int, output_dir: Path) -> Path:
        """Get the training data directory for a specific cycle."""
        if cycle == 0:
            # Cycle 0 uses original data
            return self.base_data_dir
        else:
            # Cycle N uses augmented data from cycle N
            return Path(output_dir) / f"chat_data_cycle_{cycle}"
    
    def analyze_training_data(self, data_path: Path) -> Dict[str, Any]:
        """Analyze training data composition."""
        train_data = self._load_jsonl(data_path / "train.jsonl")
        
        sources = defaultdict(int)
        cycles = defaultdict(int)
        accuracies = []
        
        for example in train_data:
            metadata = example.get("metadata", {})
            source = metadata.get("source", "original")
            cycle = metadata.get("cycle", 0)
            accuracy = metadata.get("first_epoch_accuracy")
            
            sources[source] += 1
            cycles[cycle] += 1
            if accuracy is not None:
                accuracies.append(accuracy)
        
        return {
            "total_examples": len(train_data),
            "by_source": dict(sources),
            "by_cycle": dict(cycles),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "min_accuracy": min(accuracies) if accuracies else 0.0,
            "max_accuracy": max(accuracies) if accuracies else 0.0,
        }


def main():
    """Test training data manager."""
    # Initialize with base data
    manager = TrainingDataManager(out_dir / "curation_output/chat_data")
    
    # Create a test example
    test_code = """
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 10, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""
    
    metadata = {
        "accuracy": 0.35,
        "params": 50000,
        "cycle": 1,
        "checkpoint": "test-checkpoint"
    }
    
    chat_example = manager.convert_code_to_chat_example(test_code, "test_model_001", metadata)
    print("\\nTest chat example created:")
    print(json.dumps(chat_example, indent=2))
    
    # Test augmentation
    output_dir = Path("out/test_training_data")
    stats = manager.augment_training_data([chat_example], cycle=1, output_dir=output_dir)
    print(f"\\nAugmentation stats: {stats}")
    
    # Analyze
    analysis = manager.analyze_training_data(output_dir / "chat_data_cycle_1")
    print(f"\\nData analysis: {json.dumps(analysis, indent=2)}")


if __name__ == "__main__":
    main()




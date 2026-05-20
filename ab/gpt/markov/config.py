"""
Configuration for the Prompt Improvement Pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    
    # LLM settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_remote: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.7
    
    # Dataset settings
    dataset: str = "imagenette"  # "imagenette", "cifar10" or "cifar100"

    # Training settings
    train_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.01
    training_timeout: int = 1800  # 30 minutes
    data_dir: str = "./data"
    
    # Pipeline settings
    target_accuracy: float = 0.8
    max_iterations: int = 10
    
    # Logging settings
    results_log: str = "results.log"
    prompts_log: str = "prompts.log"
    
    # Output settings
    output_dir: str = "./output"
    save_models: bool = True
    
    # Ablation Study Flags
    use_prompt_improver: bool = True
    use_reference_code: bool = True
    use_history: bool = True
    history_size: int = 5

    # Unsloth flags
    use_unsloth: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dataset not in ("imagenette", "cifar10", "cifar100"):
            raise ValueError("dataset must be 'imagenette', 'cifar10' or 'cifar100'")
        if self.target_accuracy <= 0 or self.target_accuracy > 1:
            raise ValueError("target_accuracy must be between 0 and 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.train_epochs < 1:
            raise ValueError("train_epochs must be at least 1")


# Default configuration
DEFAULT_CONFIG = PipelineConfig()


# Quick test configuration (for debugging)
TEST_CONFIG = PipelineConfig(
    train_epochs=1,
    max_iterations=3,
    target_accuracy=0.5,
    training_timeout=600
)

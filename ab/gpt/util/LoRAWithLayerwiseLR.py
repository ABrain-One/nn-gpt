"""
LoRA Fine-tuning with Layerwise Learning Rates

Extension to LoRA class that supports layerwise learning rates for transformer models.
This enables more efficient fine-tuning by using different learning rates for different layers.
"""

import torch
from transformers import TrainingArguments, get_scheduler
from peft import get_peft_model
from typing import Optional

from ab.gpt.util.LayerwiseLR import (
    LayerwiseLRConfig,
    assign_layerwise_lr,
    print_layerwise_lr
)


def create_lora_optimizer_with_layerwise_lr(
    peft_model,
    config: LayerwiseLRConfig,
    optimizer_name: str = "adamw_torch",
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Create optimizer for LoRA fine-tuning with layerwise learning rates.

    This function creates an optimizer that assigns different learning rates
    to different layers of the model, which can improve fine-tuning efficiency.

    Args:
        peft_model: PEFT model (after get_peft_model)
        config: LayerwiseLR configuration
        optimizer_name: Optimizer type (adamw_torch, sgd, etc.)
        weight_decay: Weight decay for regularization

    Returns:
        Optimizer with layerwise learning rates
    """
    # Get parameter groups with layerwise LR
    param_groups = assign_layerwise_lr(peft_model, config)

    # Create optimizer based on name
    if optimizer_name == "adamw_torch":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "adamw_8bit" or optimizer_name == "paged_adamw_8bit":
        # Try to import bitsandbytes for 8-bit optimizer
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(param_groups, weight_decay=weight_decay)
        except ImportError:
            print("[WARN] bitsandbytes not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, weight_decay=weight_decay, momentum=0.9)
    else:
        # Default to AdamW
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    return optimizer


def get_layerwise_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "linear"
):
    """
    Create learning rate scheduler for layerwise LR optimizer.

    Args:
        optimizer: Optimizer with layerwise LR
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler (linear, cosine, etc.)

    Returns:
        Learning rate scheduler
    """
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler


def apply_layerwise_lr_to_trainer(
    trainer,
    config: LayerwiseLRConfig,
    optimizer_name: str = "adamw_torch",
    weight_decay: float = 0.01,
    scheduler_type: str = "linear"
):
    """
    Apply layerwise learning rates to an existing Trainer.

    This function modifies a Trainer to use layerwise learning rates.
    Call this BEFORE trainer.train().

    Args:
        trainer: transformers.Trainer instance
        config: LayerwiseLR configuration
        optimizer_name: Optimizer type
        weight_decay: Weight decay
        scheduler_type: LR scheduler type

    Example:
        ```python
        # In LoRA.train() method:
        trainer = SFTTrainer(...)

        # Apply layerwise LR
        if use_layerwise_lr:
            from ab.gpt.util.LoRAWithLayerwiseLR import apply_layerwise_lr_to_trainer
            apply_layerwise_lr_to_trainer(
                trainer,
                LayerwiseLRConfig(base_lr=1e-5, strategy="linear_decay")
            )

        # Then train
        trainer.train()
        ```
    """
    # Create optimizer with layerwise LR
    optimizer = create_lora_optimizer_with_layerwise_lr(
        trainer.model,
        config,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay
    )

    # Calculate training steps for scheduler
    num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * trainer.args.num_train_epochs

    # Handle warmup
    if trainer.args.warmup_steps > 0:
        num_warmup_steps = trainer.args.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * trainer.args.warmup_ratio)

    # Create scheduler
    scheduler = get_layerwise_lr_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        scheduler_type=scheduler_type
    )

    # Inject optimizer and scheduler into trainer
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler

    # Print layerwise LR for debugging
    print("\n[INFO] Layerwise Learning Rates Applied:")
    print_layerwise_lr(optimizer)

    return trainer


# ============================================================================
# LLM-Enhanced Layerwise LR (For future integration with TuneHyperparameters)
# ============================================================================

def generate_llm_lora_learning_rates(
    peft_model,
    base_lr: float = 1e-5,
    llm_hyperparameter_predictor=None
) -> LayerwiseLRConfig:
    """
    Use LLM to generate optimal layerwise learning rates for LoRA fine-tuning.

    This function will integrate with TuneHyperparameters.py to:
    1. Analyze the LoRA adapter configuration
    2. Send model architecture info to LLM
    3. Get suggested learning rates per layer/module
    4. Return LayerwiseLRConfig with LLM suggestions

    Args:
        peft_model: PEFT model with LoRA adapters
        base_lr: Base learning rate
        llm_hyperparameter_predictor: Trained LLM for hyperparameter prediction

    Returns:
        LayerwiseLRConfig with LLM-suggested rates
    """
    # TODO: Integrate with TuneHyperparameters.py
    # For now, use heuristic approach

    from ab.gpt.util.LayerwiseLR import generate_llm_learning_rates

    # Generate heuristic-based rates
    llm_suggested_rates = generate_llm_learning_rates(
        peft_model,
        base_lr=base_lr,
        model_arch="LoRA-Transformer",
        task="code_generation"
    )

    config = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.CUSTOM,
        llm_suggested_rates=llm_suggested_rates
    )

    return config


# ============================================================================
# Example Integration Pattern
# ============================================================================

"""
Example: How to use this in TuneNNGen.py

In TuneNNGen.py, add a new parameter:

```python
parser.add_argument('--use_layerwise_lr', action='store_true', default=False,
                    help='Use layerwise learning rates for LoRA fine-tuning')
parser.add_argument('--layerwise_lr_strategy', type=str, default='linear_decay',
                    help='Layerwise LR strategy (uniform, linear_decay, exponential_decay, discriminative, custom)')
```

Then in the main() function:

```python
# After creating training_args and peft_config...

if args.use_layerwise_lr:
    from ab.gpt.util.LoRAWithLayerwiseLR import generate_llm_lora_learning_rates

    # Option 1: Use LLM-suggested rates
    layerwise_config = generate_llm_lora_learning_rates(
        model,  # or peft_model
        base_lr=learning_rate
    )

    # Option 2: Use predefined strategy
    from ab.gpt.util.LayerwiseLR import LayerwiseLRConfig
    layerwise_config = LayerwiseLRConfig(
        base_lr=learning_rate,
        strategy=args.layerwise_lr_strategy
    )

    # Pass config to LoRA tuner (need to modify LoRA.train() to accept it)
    lora_tuner.train(dataset, tokenizer, out_path, layerwise_lr_config=layerwise_config)
```

Then in LoRA.train() method (ab/gpt/util/LoRA.py):

```python
def train(self, dataset, tokenizer, output_dir, layerwise_lr_config=None, ...):
    # ... existing code ...

    trainer = SFTTrainer(...)

    # Apply layerwise LR if configured
    if layerwise_lr_config is not None:
        from ab.gpt.util.LoRAWithLayerwiseLR import apply_layerwise_lr_to_trainer
        trainer = apply_layerwise_lr_to_trainer(
            trainer,
            layerwise_lr_config,
            optimizer_name=self.training_args.optim,
            weight_decay=self.training_args.weight_decay
        )

    # Train
    trainer.train()
    # ... rest of the code ...
```
"""


if __name__ == "__main__":
    print("LoRA Layerwise LR Integration Module")
    print("="*80)
    print("\nThis module provides utilities for using layerwise learning rates")
    print("with LoRA fine-tuning in the TuneNNGen pipeline.")
    print("\nKey functions:")
    print("  - create_lora_optimizer_with_layerwise_lr()")
    print("  - apply_layerwise_lr_to_trainer()")
    print("  - generate_llm_lora_learning_rates()")
    print("\nSee module docstring for integration examples.")

"""
Layerwise Learning Rate Utility

Provides simple, configurable layerwise learning rates for any PyTorch model.
Can be enhanced with LLM-based learning rate suggestions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
import re
import json
import warnings


class LayerwiseLRConfig:
    """Configuration for layerwise learning rates."""

    UNIFORM = "uniform"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    DISCRIMINATIVE = "discriminative"
    CUSTOM = "custom"

    def __init__(
        self,
        base_lr: float = 1e-3,
        strategy: str = LINEAR_DECAY,
        decay_factor: float = 0.95,
        layer_lr_map: Optional[Dict[str, float]] = None,
        llm_suggested_rates: Optional[Dict[str, float]] = None
    ):
        self.base_lr = base_lr
        self.strategy = strategy
        self.decay_factor = decay_factor
        self.layer_lr_map = layer_lr_map or {}
        self.llm_suggested_rates = llm_suggested_rates or {}


def get_layer_groups(model: nn.Module) -> List[tuple]:
    """Extract layer groups from a model for layerwise LR assignment."""
    layer_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_groups.append((name, param))
    return layer_groups


def get_layer_info(model: nn.Module) -> List[Dict]:
    """Extract structured layer information for LLM prompt building."""
    info = []
    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = get_layer_groups(model)
    num_layers = len(layers)
    for idx, (name, param) in enumerate(layers):
        info.append({
            "name": name,
            "shape": list(param.shape),
            "num_params": param.numel(),
            "position": round(idx / max(num_layers - 1, 1), 3),
            "pct_of_total": round(100.0 * param.numel() / max(num_params_total, 1), 2),
        })
    return info


def assign_layerwise_lr(
    model: nn.Module,
    config: LayerwiseLRConfig
) -> List[Dict[str, Union[List[nn.Parameter], float]]]:
    """Assign layerwise learning rates based on configuration."""
    layer_groups = get_layer_groups(model)
    num_layers = len(layer_groups)

    param_groups = []

    for idx, (layer_name, param) in enumerate(layer_groups):
        if config.strategy == LayerwiseLRConfig.UNIFORM:
            lr = config.base_lr

        elif config.strategy == LayerwiseLRConfig.LINEAR_DECAY:
            decay = 1.0 - (idx / max(num_layers - 1, 1)) * (1.0 - config.decay_factor)
            lr = config.base_lr * decay

        elif config.strategy == LayerwiseLRConfig.EXPONENTIAL_DECAY:
            lr = config.base_lr * (config.decay_factor ** idx)

        elif config.strategy == LayerwiseLRConfig.DISCRIMINATIVE:
            lr = config.base_lr
            for pattern, multiplier in config.layer_lr_map.items():
                if re.search(pattern, layer_name):
                    lr = config.base_lr * multiplier
                    break

        elif config.strategy == LayerwiseLRConfig.CUSTOM:
            lr = config.base_lr
            if layer_name in config.llm_suggested_rates:
                lr = config.llm_suggested_rates[layer_name]
            else:
                for pattern, rate in config.layer_lr_map.items():
                    if re.search(pattern, layer_name):
                        lr = rate
                        break
        else:
            lr = config.base_lr

        param_groups.append({
            'params': [param],
            'lr': lr,
            'name': layer_name
        })

    return param_groups


def create_optimizer_with_layerwise_lr(
    model: nn.Module,
    config: LayerwiseLRConfig,
    optimizer_class=torch.optim.SGD,
    **optimizer_kwargs
) -> torch.optim.Optimizer:
    """Create optimizer with layerwise learning rates."""
    param_groups = assign_layerwise_lr(model, config)
    optimizer = optimizer_class(param_groups, **optimizer_kwargs)
    return optimizer


# ============================================================================
# LLM-Based Learning Rate Generation
# ============================================================================

def _build_llm_prompt(layer_info: List[Dict], base_lr: float, model_arch: str, task: str) -> str:
    """Build a structured prompt for the LLM to suggest per-layer LR multipliers."""
    layer_desc = json.dumps(layer_info, indent=2)
    return (
        f"You are an expert deep learning engineer. Given a {model_arch} neural network "
        f"being trained for {task}, suggest optimal per-layer learning rate multipliers.\n\n"
        f"Base learning rate: {base_lr}\n\n"
        f"Layer information:\n{layer_desc}\n\n"
        f"Guidelines:\n"
        f"- Early/low-level feature layers should have smaller multipliers (0.01-0.3)\n"
        f"- Middle layers should have medium multipliers (0.3-0.8)\n"
        f"- Later/classifier layers should have higher multipliers (0.8-2.0)\n"
        f"- Batch norm layers can match their parent layer\n"
        f"- Bias terms can be slightly higher than their corresponding weights\n\n"
        f"Return ONLY a JSON object mapping each layer name to its LR multiplier. Example:\n"
        f'{{"layer1.weight": 0.1, "layer1.bias": 0.12, "fc.weight": 1.0}}\n'
    )


def _parse_llm_response(response: str, layer_names: List[str], base_lr: float) -> Optional[Dict[str, float]]:
    """Parse LLM response to extract per-layer LR multipliers."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if not json_match:
        return None

    try:
        multipliers = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    # Validate: all values should be positive numbers
    if not all(isinstance(v, (int, float)) and v > 0 for v in multipliers.values()):
        return None

    # Convert multipliers to absolute LRs
    rates = {}
    for name in layer_names:
        if name in multipliers:
            rates[name] = base_lr * float(multipliers[name])
        else:
            # Try partial match
            for key, mult in multipliers.items():
                if key in name or name in key:
                    rates[name] = base_lr * float(mult)
                    break

    return rates if len(rates) > 0 else None


def generate_llm_learning_rates(
    model: nn.Module,
    base_lr: float = 1e-3,
    model_arch: str = "unknown",
    task: str = "classification",
    llm=None,
    chatbot=None,
    llm_model=None,
    llm_tokenizer=None
) -> Dict[str, float]:
    """
    Use LLM to suggest optimal learning rates for each layer.

    Args:
        model: PyTorch model
        base_lr: Base learning rate as reference
        model_arch: Model architecture name
        task: Training task (classification, segmentation, etc.)
        llm: An ab.gpt.util.LLM.LLM instance (has .model and .tokenizer)
        chatbot: An ab.gpt.util.Chatbot.ChatBot instance (preferred over raw llm)
        llm_model: Pre-loaded HF model (legacy parameter)
        llm_tokenizer: HF tokenizer (legacy parameter)

    Returns:
        Dictionary mapping layer names to suggested learning rates
    """
    layer_groups = get_layer_groups(model)
    layer_names = [name for name, _ in layer_groups]

    # Resolve model/tokenizer from llm object if provided
    if llm is not None and llm_model is None:
        llm_model = llm.model
        llm_tokenizer = llm.tokenizer

    if llm_model is None or llm_tokenizer is None:
        if chatbot is None:
            warnings.warn("No LLM available, falling back to heuristic-based LR assignment.")
            return _heuristic_layer_rates(layer_groups, base_lr)

    # Build prompt
    layer_info = get_layer_info(model)
    prompt = _build_llm_prompt(layer_info, base_lr, model_arch, task)

    try:
        if chatbot is not None:
            # Use ChatBot interface
            _, _, _, raw_response = chatbot.chat(prompt, max_new_tokens=2048, engineer_prompt=False)
        else:
            # Direct generation with model + tokenizer
            from ab.gpt.util.Chatbot import ChatBot
            bot = ChatBot(llm_model, llm_tokenizer)
            _, _, _, raw_response = bot.chat(prompt, max_new_tokens=2048, engineer_prompt=False)

        if raw_response:
            parsed = _parse_llm_response(raw_response, layer_names, base_lr)
            if parsed is not None:
                print(f"[LayerwiseLR] Successfully parsed LLM suggestions for {len(parsed)}/{len(layer_names)} layers.")
                # Fill missing layers with heuristic
                heuristic = _heuristic_layer_rates(layer_groups, base_lr)
                for name in layer_names:
                    if name not in parsed:
                        parsed[name] = heuristic[name]
                return parsed

        warnings.warn("LLM response could not be parsed, falling back to heuristic.")
    except Exception as e:
        warnings.warn(f"LLM generation failed: {e}. Falling back to heuristic.")

    return _heuristic_layer_rates(layer_groups, base_lr)


def create_optimizer_with_llm_lr(
    model: nn.Module,
    base_lr: float = 1e-3,
    model_arch: str = "unknown",
    task: str = "classification",
    llm=None,
    chatbot=None,
    optimizer_class=torch.optim.SGD,
    **optimizer_kwargs
) -> torch.optim.Optimizer:
    """
    Convenience function: get LLM-suggested LRs and create an optimizer in one call.

    Args:
        model: PyTorch model
        base_lr: Base learning rate
        model_arch: Architecture name for the LLM prompt
        task: Task description for the LLM prompt
        llm: An ab.gpt.util.LLM.LLM instance
        chatbot: An ab.gpt.util.Chatbot.ChatBot instance
        optimizer_class: Optimizer class (default: SGD)
        **optimizer_kwargs: Additional optimizer args (momentum, weight_decay, etc.)

    Returns:
        Optimizer with LLM-suggested layerwise learning rates
    """
    rates = generate_llm_learning_rates(
        model, base_lr=base_lr, model_arch=model_arch, task=task,
        llm=llm, chatbot=chatbot
    )
    config = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.CUSTOM,
        llm_suggested_rates=rates
    )
    return create_optimizer_with_layerwise_lr(model, config, optimizer_class, **optimizer_kwargs)


def _heuristic_layer_rates(layer_groups: List[tuple], base_lr: float) -> Dict[str, float]:
    """Heuristic-based learning rate assignment."""
    layer_rates = {}
    num_layers = len(layer_groups)

    for idx, (layer_name, _) in enumerate(layer_groups):
        position = idx / max(num_layers - 1, 1)

        # Gradually increase LR from early to late layers
        lr_multiplier = 0.1 + 0.9 * position

        if 'bn' in layer_name.lower() or 'batch_norm' in layer_name.lower():
            lr_multiplier *= 1.2
        elif 'bias' in layer_name.lower():
            lr_multiplier *= 1.1
        elif 'fc' in layer_name.lower() or 'linear' in layer_name.lower() or 'classifier' in layer_name.lower():
            lr_multiplier = max(lr_multiplier, 1.0)

        layer_rates[layer_name] = base_lr * lr_multiplier

    return layer_rates


# ============================================================================
# Helper Functions
# ============================================================================

def print_layerwise_lr(optimizer: torch.optim.Optimizer) -> None:
    """Print learning rates for each parameter group."""
    print("\n" + "=" * 80)
    print("Layerwise Learning Rates")
    print("=" * 80)
    for idx, group in enumerate(optimizer.param_groups):
        layer_name = group.get('name', f'group_{idx}')
        lr = group['lr']
        print(f"{layer_name:60s} | LR: {lr:.6e}")
    print("=" * 80 + "\n")


def update_layerwise_lr(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    new_base_lr: Optional[float] = None
) -> None:
    """Update learning rates while maintaining layerwise ratios."""
    if lr_scheduler is not None:
        lr_scheduler.step()
    elif new_base_lr is not None:
        if len(optimizer.param_groups) > 0:
            current_base = optimizer.param_groups[-1]['lr']
            ratio = new_base_lr / current_base
            for group in optimizer.param_groups:
                group['lr'] *= ratio

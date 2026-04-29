"""Shared GRPO trainer construction helpers for TuneRL pipelines."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.utils.checkpoint


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(tokenizer_source: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_causal_lm(
    *,
    model_source: str,
    precision: dict,
    train_device: str,
    use_deepspeed: bool,
):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision["torch_dtype"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        "torch_dtype": precision["torch_dtype"],
    }
    if not use_deepspeed:
        model_load_kwargs["device_map"] = {"": train_device}
    return AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_load_kwargs,
    )


def maybe_merge_initial_adapter(
    model,
    *,
    enabled: bool,
    adapter_path: str,
    label: str,
    empty_adapter_message: Optional[str] = None,
    missing_adapter_message: Optional[str] = None,
    load_message: Optional[str] = None,
):
    if not enabled:
        return model
    if not adapter_path:
        raise ValueError(empty_adapter_message or f"{label} adapter path is empty.")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(missing_adapter_message or f"{label} adapter not found: {adapter_path}")

    from peft import PeftModel

    print(load_message or f"Loading initial {label} adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    return model.merge_and_unload()


def build_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_or_resume_lora(
    model,
    *,
    peft_config,
    stage_adapter_dir,
    log_prefix: str,
    missing_adapter_message: Optional[str] = None,
    load_message: Optional[str] = None,
):
    from peft import PeftModel, get_peft_model

    if stage_adapter_dir is not None:
        adapter_dir = Path(stage_adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(missing_adapter_message or f"Missing adapter directory: {adapter_dir}")
        print(load_message or f"{log_prefix} Loading stage checkpoint adapter from {adapter_dir}...")
        return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    return get_peft_model(model, peft_config)


def _iter_gradient_checkpoint_roots(model: Any):
    stack = [model]
    seen: Set[int] = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        for attr in ("module", "model", "base_model", "pretrained_model"):
            try:
                child = getattr(current, attr, None)
            except Exception:
                child = None
            if child is not None:
                stack.append(child)


def enforce_non_reentrant_gradient_checkpointing(model: Any) -> Dict[str, int]:
    if model is None:
        return {"roots": 0, "modules": 0}

    checkpoint_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    patched_roots = 0
    patched_modules = 0
    seen_modules: Set[int] = set()

    for root in _iter_gradient_checkpoint_roots(model):
        original_enable = getattr(root, "_nngpt_original_gradient_checkpointing_enable", None)
        if original_enable is None:
            bound_enable = getattr(root, "gradient_checkpointing_enable", None)
            if callable(bound_enable):
                def _wrapped_gradient_checkpointing_enable(*args, _bound_enable=bound_enable, _root=root, **kwargs):
                    gradient_checkpointing_kwargs = dict(kwargs.get("gradient_checkpointing_kwargs") or {})
                    gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
                    kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs
                    result = _bound_enable(*args, **kwargs)
                    for module in _root.modules():
                        if hasattr(module, "_gradient_checkpointing_func"):
                            module._gradient_checkpointing_func = checkpoint_func
                    return result

                setattr(root, "_nngpt_original_gradient_checkpointing_enable", bound_enable)
                setattr(root, "gradient_checkpointing_enable", _wrapped_gradient_checkpointing_enable)
        patched_roots += 1
        try:
            root.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except Exception:
            pass
        try:
            root.enable_input_require_grads()
        except Exception:
            pass
        for module in root.modules():
            module_id = id(module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            if hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = checkpoint_func
                patched_modules += 1

    return {"roots": patched_roots, "modules": patched_modules}


def enable_non_reentrant_gradient_checkpointing(
    model,
    *,
    log_prefix: str,
) -> Dict[str, int]:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    gc_patch_stats = enforce_non_reentrant_gradient_checkpointing(model)
    print(
        f"{log_prefix} Gradient checkpointing enforcement: "
        f"roots={gc_patch_stats['roots']} modules={gc_patch_stats['modules']} use_reentrant=False"
    )
    return gc_patch_stats


def train_grpo(
    *,
    trainer,
    trainer_checkpoint,
    log_prefix: str,
) -> None:
    if trainer_checkpoint is not None:
        print(f"{log_prefix} Resuming trainer state from {trainer_checkpoint}...")
        trainer.train(resume_from_checkpoint=str(trainer_checkpoint))
    else:
        trainer.train()

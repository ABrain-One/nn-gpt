"""
TuneHeadersNNGen.py - Header-Only Fine-tuning Module

This module specializes in fine-tuning only the header/output layers (lm_head) of the model,
as opposed to fine-tuning the full transformer stack. This is useful for:
1. Reducing training time and memory footprint
2. Focusing on output quality and format
3. Preserving internal representations while adapting the output layer

Key differences from standard TuneNNGen.py:
- target_modules only includes 'lm_head' (output layer)
- Reduced LoRA rank and parameters for efficiency
- Faster training cycles with lower memory requirements
- Suitable for fine-tuning model output behavior (headers, formatting, etc.)
"""

import argparse
import os
import json
import sys
from typing import Literal

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from ab.gpt.NNEval import NN_TRAIN_EPOCHS
from ab.gpt.util.Const import nngpt_dir, new_out_file

# --- Header-Only Fine-tuning Defaults ---
# These defaults are optimized for header-only training (lm_head layer)

START_LAYER = 0
END_LAYER = 1  # Only the output layer
TUNE_LAYERS = range(START_LAYER, END_LAYER)

R = 16  # Smaller dimension for header-only (vs 32 for full)
LORA_ALPHA = 16  # Smaller alpha for header
LORA_DROPOUT = 0.05
TARGET_MODULES = ('lm_head',)  # ONLY lm_head for header-only training
TASK_TYPE = 'CAUSAL_LM'
BiasType = Literal['none', 'all', 'lora_only']
BIAS: BiasType = 'none'

LEARNING_RATE = 5e-6  # Smaller learning rate for sensitive output layer
MAX_GRAD_NORM = 1.0
PEFT = None
SKIP_EPOCHES = -1

NUM_TRAIN_EPOCHS = 2  # Shorter training for header (vs 3 for full)
LR_SCHEDULER = 'cosine'
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4  # Smaller accumulation for header
WARMUP_RATIO = 0.05
TEST_NN = 5  # Fewer test samples for header
LOGGING_STEPS = 48  # Less frequent logging
OPTIMIZER = 'paged_adamw_8bit'
LLM_TUNE_CONF = 'NN_gen.json'
NN_GEN_CONF = 'NN_gen.json'
NN_GEN_CONF_ID = 'improve_classification_only'
LLM_CONF = 'ds_coder_7b_olympic.json'
MAX_PROMPTS = 2 * 1024  # Smaller for header (vs 4*1024 for full)
# CRITICAL: max_new_tokens MUST be large enough to generate COMPLETE code
# The model needs room to generate all required sections:
# <hp>...</hp>, <tr>...</tr>, <nn>...</nn> (with complete class definitions)
# We use 16k tokens for generation, then truncate at boundaries after generation completes
MAX_NEW_TOKENS = 16 * 1024  # Keep at 16k to ensure COMPLETE generation (same as full training)
SAVE_LLM_OUTPUT = True
USE_DEEPSPEED = False
NN_NAME_PREFIX = 'header'
TEMPERATURE = 0.7  # Slightly lower for more deterministic output
TOP_K = 50
TOP_P = 0.9
TEST_METRIC = None
ONNX_RUN = False
TRANS_MODE = False

# Import the actual tune function from the main module
if ONNX_RUN:
    from ab.gpt.util.Tune_Onnx import tune, ds_conf
else:
    from ab.gpt.util.Tune import tune, ds_conf


def _best_dtype_args():
    """Detect best mixed precision dtype based on hardware support."""
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16_ok, "fp16": not bf16_ok}


def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=NN_GEN_CONF, 
         nn_gen_conf_id=NN_GEN_CONF_ID, llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES, 
         per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
         warmup_ratio=WARMUP_RATIO, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER, max_prompts=MAX_PROMPTS, 
         save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=MAX_NEW_TOKENS, use_deepspeed=USE_DEEPSPEED, 
         nn_name_prefix=NN_NAME_PREFIX, nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K, 
         top_p=TOP_P, data_dir=None, evaluation_strategy=None, eval_steps=None, save_strategy=None, save_steps=None,
         save_total_limit=None, load_best_model_at_end=False, metric_for_best_model=None, warmup_steps=None, 
         weight_decay=None, per_device_eval_batch_size=None, onnx_run=ONNX_RUN, trans_mode=TRANS_MODE):
    
    print(f'''Header-Only Fine-tuning Configuration:
num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler}, max_grad_norm={max_grad_norm}, 
r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, target_modules={target_modules}, 
learning_rate={learning_rate}, max_new_tokens={max_new_tokens}, temperature={temperature}''')

    test_prm = {
        'metric_for_best_model': test_metric,
        'greater_is_better': True,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 3,
        'load_best_model_at_end': False
    } if test_metric else {}
    
    dtype_flags = _best_dtype_args()
    
    if evaluation_strategy is not None:
        # PIPELINE MODE
        training_kwargs = {
            'report_to': None,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'learning_rate': learning_rate,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs_headers',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            'max_grad_norm': max_grad_norm,
            'num_train_epochs': num_train_epochs,
            **dtype_flags,
        }
        
        if warmup_steps is not None:
            training_kwargs['warmup_steps'] = warmup_steps
        else:
            training_kwargs['warmup_ratio'] = warmup_ratio
        
        if weight_decay is not None:
            training_kwargs['weight_decay'] = weight_decay
        
        training_kwargs['eval_strategy'] = evaluation_strategy
        if eval_steps is not None:
            training_kwargs['eval_steps'] = eval_steps
        if per_device_eval_batch_size is not None:
            training_kwargs['per_device_eval_batch_size'] = per_device_eval_batch_size
        if save_strategy is not None:
            training_kwargs['save_strategy'] = save_strategy
            if save_steps is not None:
                training_kwargs['save_steps'] = save_steps
            if save_total_limit is not None:
                training_kwargs['save_total_limit'] = save_total_limit
        if load_best_model_at_end:
            training_kwargs['load_best_model_at_end'] = True
            if metric_for_best_model is not None:
                training_kwargs['metric_for_best_model'] = metric_for_best_model
    else:
        # STANDALONE MODE
        training_kwargs = {
            'num_train_epochs': num_train_epochs,
            'lr_scheduler_type': lr_scheduler,
            'max_grad_norm': max_grad_norm,
            'report_to': None,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_ratio': warmup_ratio,
            'learning_rate': learning_rate,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs_headers',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            **dtype_flags,
            **test_prm
        }
    
    training_args = TrainingArguments(**training_kwargs)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=list(tune_layers),
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type)

    print(f"[Header-Only LoRA Config] target_modules={target_modules}")
    print(f"[Header-Only LoRA Config] layers_to_transform={list(tune_layers)}")
    print(f"[Header-Only LoRA Config] r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")

    tune(test_nn, nn_train_epochs, skip_epoches, peft, llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf, 
         training_args, peft_config, max_prompts=max_prompts, save_llm_output=save_llm_output, 
         max_new_tokens=max_new_tokens, nn_name_prefix=nn_name_prefix, temperature=temperature, top_k=top_k, 
         top_p=top_p, onnx_run=onnx_run, trans_mode=trans_mode)
    
    print("\n" + "="*70)
    print("HEADER-ONLY FINE-TUNING SUMMARY")
    print("="*70)
    print(f"✓ Target Modules: {target_modules} (header/output layer only)")
    print(f"✓ LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"✓ Training: epochs={num_train_epochs}, scheduler={lr_scheduler}, lr={learning_rate}")
    print(f"✓ Batch: {per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size*gradient_accumulation_steps}")
    print(f"✓ Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_new_tokens}")
    print(f"✓ Output Directory: nngpt_dir/outputs_headers")
    print("="*70)


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TARGET_MODULES)
    parser = argparse.ArgumentParser(description='Header-Only Fine-tuning of Neural Networks generated by NNAlter.py.')
    
    # Header-specific parameters
    parser.add_argument('-ne', '--num_train_epochs', type=int, default=NUM_TRAIN_EPOCHS,
                        help=f'Number of header fine-tuning epochs (default: {NUM_TRAIN_EPOCHS}).')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER,
                        help=f'Name of learning rate scheduler (default: {LR_SCHEDULER}).')
    parser.add_argument('-g', '--max_grad_norm', type=float, default=MAX_GRAD_NORM,
                        help=f'Upper limit on gradients (default: {MAX_GRAD_NORM}).')
    parser.add_argument('--test_metric', type=str, default=TEST_METRIC,
                        help=f'Test metric for evaluation (default: {TEST_METRIC}).')
    
    # LoRA configuration (header-optimized defaults)
    parser.add_argument('-r', '--r', type=int, default=R,
                        help=f'LoRA rank - smaller for header (default: {R}).')
    parser.add_argument('-a', '--lora_alpha', type=float, default=LORA_ALPHA,
                        help=f'LoRA alpha parameter (default: {LORA_ALPHA}).')
    parser.add_argument('-d', '--lora_dropout', type=float, default=LORA_DROPOUT,
                        help=f'LoRA dropout probability (default: {LORA_DROPOUT}).')
    parser.add_argument('-t', '--target_modules', type=lambda s: tuple(s.split(',')), default=TARGET_MODULES,
                        help=f'Target modules (default: {TARGET_MODULES_STR})')
    parser.add_argument('-l', '--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE}).')
    parser.add_argument('-y', '--task_type', type=str, default=TASK_TYPE,
                        help=f'LLM task type (default: {TASK_TYPE}).')
    parser.add_argument('-b', '--bias', type=str, default=BIAS,
                        help=f'Bias type (default: {BIAS}).')
    
    # Config files
    parser.add_argument('--llm_tune_conf', type=str, default=LLM_TUNE_CONF,
                        help=f'Config with prompt for fine-tuning (default: {LLM_TUNE_CONF}).')
    parser.add_argument('--nn_gen_conf', type=str, default=NN_GEN_CONF,
                        help=f'Config with prompt for NN generation (default: {NN_GEN_CONF}).')
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID,
                        help=f'Prompt ID in config (default: {NN_GEN_CONF_ID}).')
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF,
                        help=f'LLM config file (default: {LLM_CONF}).')
    
    # Training parameters (header-optimized)
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN,
                        help=f'Neural networks to test before/between epochs (default: {TEST_NN}).')
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f'Training epochs for generated NNs (default: {NN_TRAIN_EPOCHS}).')
    parser.add_argument('-m', '--max_prompts', type=int, default=MAX_PROMPTS,
                        help=f'Max prompts for fine-tuning (default: {MAX_PROMPTS}).')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS,
                        help=f'Max output tokens - enforced to prevent truncation (default: {MAX_NEW_TOKENS}).')
    parser.add_argument('--save_llm_output', type=bool, default=SAVE_LLM_OUTPUT,
                        help=f'Save full LLM output (default: {SAVE_LLM_OUTPUT}).')
    parser.add_argument('--use_deepspeed', type=bool, default=USE_DEEPSPEED,
                        help=f'Use DeepSpeed optimization (default: {USE_DEEPSPEED}).')
    parser.add_argument('--per_device_train_batch_size', type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f'Per device train batch size (default: {PER_DEVICE_TRAIN_BATCH_SIZE}).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help=f'Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}).')
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO,
                        help=f'Warmup ratio (default: {WARMUP_RATIO}).')
    parser.add_argument('--logging_steps', type=int, default=LOGGING_STEPS,
                        help=f'Logging frequency (default: {LOGGING_STEPS}).')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help=f'Optimizer (default: {OPTIMIZER}).')
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES,
                        help='Number of epochs to skip NN generation.')
    parser.add_argument('--peft', type=str, default=None, help='Path to saved LoRA layers.')
    parser.add_argument("--data_dir", type=str, default=None,
        help="Folder with train.jsonl/dev.jsonl/test.jsonl.")
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f'Neural network name prefix (default: {NN_NAME_PREFIX}).')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'Generation temperature (default: {TEMPERATURE}).')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help=f'Top-k for generation (default: {TOP_K}).')
    parser.add_argument('--top_p', type=float, default=TOP_P,
                        help=f'Top-p for generation (default: {TOP_P}).')
    
    # Pipeline overrides
    parser.add_argument('--evaluation_strategy', type=str, default=None,
                        help="Evaluation strategy during training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None,
                        help="Per device eval batch size.")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="Evaluate every N steps.")
    parser.add_argument('--save_strategy', type=str, default=None,
                        help="Save strategy during training.")
    parser.add_argument('--save_steps', type=int, default=None,
                        help="Save checkpoint every N steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help="Maximum number of checkpoints to keep.")
    parser.add_argument('--load_best_model_at_end', action=argparse.BooleanOptionalAction, default=False,
                        help="Load best model at end of training.")
    parser.add_argument('--metric_for_best_model', type=str, default=None,
                        help="Metric for selecting best model.")
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help="Warmup steps override.")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="Weight decay for regularization.")
    parser.add_argument('--onnx_run', type=int, choices=[0, 1], default=0,
                    help="Enable ONNX (1) or disable (0, default)")
    parser.add_argument('--trans_mode', type=float, default=TRANS_MODE,
                        help="Transform fine-tuning only mode.")

    args = parser.parse_args()

    # Run header-only fine-tuning
    main(num_train_epochs=args.num_train_epochs,
         lr_scheduler=args.lr_scheduler,
         max_grad_norm=args.max_grad_norm,
         tune_layers=range(0, 1),  # Only output layer for headers
         r=args.r,
         lora_alpha=args.lora_alpha,
         lora_dropout=args.lora_dropout,
         task_type=args.task_type,
         bias=args.bias,
         target_modules=args.target_modules,
         learning_rate=args.learning_rate,
         llm_tune_conf=args.llm_tune_conf,
         nn_gen_conf=args.nn_gen_conf,
         nn_gen_conf_id=args.nn_gen_conf_id,
         llm_conf=args.llm_conf,
         test_nn=args.test_nn,
         per_device_train_batch_size=args.per_device_train_batch_size,
         gradient_accumulation_steps=args.gradient_accumulation_steps,
         warmup_ratio=args.warmup_ratio,
         logging_steps=args.logging_steps,
         optimizer=args.optimizer,
         peft=args.peft,
         skip_epoches=args.skip_epoches,
         max_prompts=args.max_prompts,
         max_new_tokens=args.max_new_tokens,
         use_deepspeed=args.use_deepspeed,
         save_llm_output=args.save_llm_output,
         nn_name_prefix=args.nn_name_prefix,
         nn_train_epochs=args.nn_train_epochs,
         temperature=args.temperature,
         top_k=args.top_k,
         top_p=args.top_p,
         test_metric=args.test_metric,
         data_dir=args.data_dir,
         evaluation_strategy=args.evaluation_strategy,
         eval_steps=args.eval_steps,
         per_device_eval_batch_size=args.per_device_eval_batch_size,
         save_strategy=args.save_strategy,
         save_steps=args.save_steps,
         save_total_limit=args.save_total_limit,
         load_best_model_at_end=args.load_best_model_at_end,
         metric_for_best_model=args.metric_for_best_model,
         warmup_steps=args.warmup_steps,
         weight_decay=args.weight_decay,
         onnx_run=args.onnx_run
    )

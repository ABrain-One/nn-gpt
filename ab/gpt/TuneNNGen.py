import argparse
import os
import json
from typing import Literal

from peft import LoraConfig
from transformers import TrainingArguments

from ab.gpt.NNEval import NN_TRAIN_EPOCHS
from ab.gpt.util.Const import nngpt_dir, new_out_file
from ab.gpt.util.Tune import tune, ds_conf

# --- Default Evaluation Parameters ---
# These will be used as defaults for argparse arguments
START_LAYER = 0
END_LAYER = 24
TUNE_LAYERS = range(START_LAYER, END_LAYER)
R = 32  # dimension of the updated matrices
LORA_ALPHA = 32  # parameter for scaling
LORA_DROPOUT = 0.05  # dropout probability for layers
TARGET_MODULES = ('q_proj', 'v_proj', 'k_proj', 'o_proj')  # Reordered for standalone default
TASK_TYPE = 'CAUSAL_LM'
BiasType = Literal['none', 'all', 'lora_only']
BIAS: BiasType = 'none'

LEARNING_RATE = 1e-6  # Conservative default for standalone
MAX_GRAD_NORM = 1.0  # Gradient clipping

PEFT = None
SKIP_EPOCHES = -1

NUM_TRAIN_EPOCHS = 3  # Standalone default
LR_SCHEDULER = 'cosine'  # Learning rate scheduler
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Increased for better stability
WARMUP_RATIO = 0.05  # Warmup as ratio of total steps
TEST_NN = 10
LOGGING_STEPS = 96  # Less frequent logging
OPTIMIZER = 'paged_adamw_8bit'
LLM_TUNE_CONF = 'NN_gen.json'
NN_GEN_CONF = 'NN_gen.json'
NN_GEN_CONF_ID = 'improve_classification_only'
LLM_CONF = 'ds_coder_7b_olympic.json'
MAX_PROMPTS = 4 * 1024  # Increased
MAX_NEW_TOKENS = 16 * 1024
SAVE_LLM_OUTPUT = True
USE_DEEPSPEED = False
NN_NAME_PREFIX = None
TEMPERATURE = 0.8
TOP_K = 70
TOP_P = 0.9
TEST_METRIC = None  # 'bleu' or other metric for evaluation

def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=NN_GEN_CONF, nn_gen_conf_id=NN_GEN_CONF_ID,
         llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES, per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, warmup_ratio=WARMUP_RATIO, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER,
         max_prompts=MAX_PROMPTS, save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=MAX_NEW_TOKENS, use_deepspeed=USE_DEEPSPEED, nn_name_prefix=NN_NAME_PREFIX,
         nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, data_dir=None, 
         # Pipeline-specific overrides (for backward compatibility with iterative_finetune.py)
         evaluation_strategy=None, eval_steps=None, save_strategy=None, save_steps=None, 
         save_total_limit=None, load_best_model_at_end=False, metric_for_best_model=None, warmup_steps=None, weight_decay=None, per_device_eval_batch_size=None):
    print(f'''All hyperparameters: 
num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler}, max_grad_norm={max_grad_norm}, tune_layers={tune_layers}, test_metric={test_metric}, 
r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, target_modules={target_modules}, task_type={task_type}, bias={bias}, 
learning_rate={learning_rate}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf}, nn_gen_conf_id={nn_gen_conf_id},
llm_conf={llm_conf}, test_nn={test_nn}, nn_train_epochs={nn_train_epochs}, peft={peft}, skip_epoches={skip_epoches}, 
per_device_train_batch_size={per_device_train_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, warmup_ratio={warmup_ratio}, 
logging_steps={logging_steps}, optimizer={optimizer}, max_prompts={max_prompts}, save_llm_output={save_llm_output}, max_new_tokens={max_new_tokens}, 
use_deepspeed={use_deepspeed}, nn_name_prefix={nn_name_prefix}, temperature={temperature}, top_k={top_k}, top_p={top_p} ''')

    # Build test_prm for standalone mode (epoch-based evaluation)
    # Pipeline mode will override with step-based evaluation via evaluation_strategy
    test_prm = {
        'metric_for_best_model': test_metric,
        'greater_is_better': True,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 3,
        'load_best_model_at_end': False
    } if test_metric else {}
    
    # Build TrainingArguments kwargs
    # Pipeline mode (when evaluation_strategy is passed): use step-based eval with pipeline overrides
    # Standalone mode: use epoch-based eval with test_metric
    if evaluation_strategy is not None:
        # PIPELINE MODE: Use pipeline-specific settings
        training_kwargs = {
            'report_to': None,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'learning_rate': learning_rate,
            'fp16': True,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            'max_grad_norm': max_grad_norm,
            'num_train_epochs': num_train_epochs,  # Use parameter from command-line or default
        }
        
        # Add warmup - pipeline may pass warmup_steps (override) or use warmup_ratio
        if warmup_steps is not None:
            training_kwargs['warmup_steps'] = warmup_steps
        else:
            training_kwargs['warmup_ratio'] = warmup_ratio
        
        # Add weight_decay if provided by pipeline
        if weight_decay is not None:
            training_kwargs['weight_decay'] = weight_decay
        
        # Pipeline evaluation settings
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
        # STANDALONE MODE: Use new defaults with epoch-based training
        training_kwargs = {
            'num_train_epochs': num_train_epochs,
            'lr_scheduler_type': lr_scheduler,
            'max_grad_norm': max_grad_norm,
            'report_to': None,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_ratio': warmup_ratio,
            'learning_rate': learning_rate,
            'fp16': True,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            **test_prm  # Add test metric configuration if provided
        }
    
    # Create TrainingArguments with all parameters at once
    training_args = TrainingArguments(**training_kwargs)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=list(tune_layers),
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type)

    # Log effective LoRA targets to catch mismatched names early
    print(f"[LoRA Config] target_modules={target_modules}")
    print(f"[LoRA Config] layers_to_transform={list(tune_layers)}")
    print(f"[LoRA Config] r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")

    # (Optional) sanity check to surface wrong target_modules/layer indices early
    try:
        from peft import get_peft_model
        import copy
        # Validate peft_config structure - check that target_modules is not empty
        if not target_modules or len(target_modules) == 0:
            raise ValueError("target_modules cannot be empty")
        # Check that layers_to_transform makes sense
        layer_list = list(tune_layers)
        if layer_list and (min(layer_list) < 0):
            raise ValueError(f"Invalid layer indices: layers must be >= 0, got {layer_list}")
        # If we had access to model.num_hidden_layers, we could validate upper bound too
        # For now, just ensure the config object is well-formed
        _probe = copy.deepcopy(peft_config)
        # Basic validation passed
        print(f"[VALID] peft_config: r={r}, target_modules={target_modules}, layers={layer_list}")
    except Exception as e:
        print(f"[WARN] peft_config validation warning: {e}")
        # Don't fail, just warn - actual validation happens when model is loaded

    tune(test_nn, nn_train_epochs, skip_epoches, peft, llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf, training_args, peft_config,
         max_prompts=max_prompts, save_llm_output=save_llm_output, max_new_tokens=max_new_tokens, nn_name_prefix=nn_name_prefix, 
         temperature=temperature, top_k=top_k, top_p=top_p)
    
    print("\n" + "="*70)
    print("FINE-TUNING CONFIGURATION SUMMARY")
    print("="*70)
    print(f"✓ LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target={target_modules}")
    
    # Show warmup based on what was actually used
    if evaluation_strategy is not None:
        # Pipeline mode
        warmup_display = f"warmup_steps={warmup_steps}" if warmup_steps is not None else f"warmup_ratio={warmup_ratio}"
        wd_display = f", wd={weight_decay}" if weight_decay is not None else ""
        print(f"✓ Training (PIPELINE): {warmup_display}, lr={learning_rate}{wd_display}, batch={per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size*gradient_accumulation_steps}")
        print(f"✓ Evaluation: strategy={evaluation_strategy}, steps={eval_steps}")
        print(f"✓ Checkpointing: save_strategy={save_strategy}, save_steps={save_steps}, total_limit={save_total_limit}")
        if load_best_model_at_end:
            print(f"✓ Best model selection: metric={metric_for_best_model}")
    else:
        # Standalone mode
        print(f"✓ Training (STANDALONE): epochs={num_train_epochs}, scheduler={lr_scheduler}, warmup_ratio={warmup_ratio}, lr={learning_rate}, batch={per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size*gradient_accumulation_steps}")
        if test_metric:
            print(f"✓ Test metric: {test_metric} (epoch-based evaluation)")
    
    print(f"✓ Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_new_tokens}")
    print(f"✓ Data: Completion-only training, NO packing (unique code generation)")
    print("="*70)


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TARGET_MODULES)
    parser = argparse.ArgumentParser(description='Evaluate Neural Networks generated by NNAlter.py.')
    
    # Standalone-specific parameters
    parser.add_argument('-ne', '--num_train_epochs', type=int, default=NUM_TRAIN_EPOCHS,
                        help=f'Number of LLM fine-tuning epochs (default: {NUM_TRAIN_EPOCHS}).')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER,
                        help=f'Name of learning rate scheduler for LLM fine-tuning (default: {LR_SCHEDULER}).')
    parser.add_argument('-g', '--max_grad_norm', type=float, default=MAX_GRAD_NORM,
                        help=f'Upper limit on the backpropagation gradients for LLM fine-tuning (default: {MAX_GRAD_NORM}).')
    parser.add_argument('--test_metric', type=str, default=TEST_METRIC,
                        help=f'Test metric for LLM fine-tuning implemented in transformers package (default: {TEST_METRIC}).')
    
    # LoRA configuration
    parser.add_argument('-s', '--start_layer', type=int, default=START_LAYER,
                        help=f'Index of the first fine-tuned layer in the LLM (default: {START_LAYER}).')
    parser.add_argument('-e', '--end_layer', type=int, default=END_LAYER,
                        help=f'Index of the last fine-tuned layer in the LLM (default: {END_LAYER}).')
    parser.add_argument('-r', '--r', type=int, default=R,
                        help=f'Dimension of the updated matrices (default: {R}).')
    parser.add_argument('-a', '--lora_alpha', type=float, default=LORA_ALPHA,
                        help=f'LoRA alpha parameter for scaling (default: {LORA_ALPHA}).')
    parser.add_argument('-d', '--lora_dropout', type=float, default=LORA_DROPOUT,
                        help=f'LoRA dropout probability for layers (default: {LORA_DROPOUT}).')
    parser.add_argument('-t', '--target_modules', type=lambda s: s.split(','), default=TARGET_MODULES,
                        help=f'Target modules separated by comma (default: {TARGET_MODULES_STR})')
    parser.add_argument('-l', '--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE}).')
    parser.add_argument('-y', '--task_type', type=str, default=TASK_TYPE,
                        help=f'LLM task type (default: {TASK_TYPE}).')
    parser.add_argument('-b', '--bias', type=str, default=BIAS,
                        help=f'Bias type (default: {BIAS}).')
    
    # Config files
    parser.add_argument('--llm_tune_conf', type=str, default=LLM_TUNE_CONF,
                        help=f'Config with a prompt for LLM fine-tuning (default: {LLM_TUNE_CONF}).')
    parser.add_argument('--nn_gen_conf', type=str, default=NN_GEN_CONF,
                        help=f'Config with a prompt for generation of neural networks by LLM (default: {NN_GEN_CONF}).')
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID,
                        help=f'Specifies prompt in the config for neural network generation by LLM (default: {NN_GEN_CONF_ID}).')
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF,
                        help=f'Config of LLM (default: {LLM_CONF}).')
    
    # Training configuration
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN,
                        help=f'Count of neural networks generated or modified by the LLM before and between fine-tuning epochs to monitor training progress (default: {TEST_NN}).')
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f'Number of training epochs for the generated neural network (default: {NN_TRAIN_EPOCHS}).')
    parser.add_argument('-m', '--max_prompts', type=int, default=MAX_PROMPTS,
                        help=f'Max prompts for LLM fine-tuning; excess is truncated (default: {MAX_PROMPTS}).')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS,
                        help=f'Max number of tokens in LLM output (default: {MAX_NEW_TOKENS}).')
    parser.add_argument('--save_llm_output', type=bool, default=SAVE_LLM_OUTPUT,
                        help=f'Save full output of LLM in the file {new_out_file} (default: {SAVE_LLM_OUTPUT}).')
    parser.add_argument('--use_deepspeed', type=bool, default=USE_DEEPSPEED,
                        help=f'Utilize DeepSpeed optimizations for LLM fine-tuning (default: {USE_DEEPSPEED}).')
    parser.add_argument('--per_device_train_batch_size', type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f'Per device train batch size (default: {PER_DEVICE_TRAIN_BATCH_SIZE}).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help=f'Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}).')
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO,
                        help=f'Warmup step ratio for one fine-tuning epoch (default: {WARMUP_RATIO}).')
    parser.add_argument('--logging_steps', type=int, default=LOGGING_STEPS,
                        help=f'Logging steps (default: {LOGGING_STEPS}).')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help=f'Optimizer for LLM fine-tuning (default: {OPTIMIZER}).')
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES,
                        help='Number of epoches to skip the neural network generation.')
    parser.add_argument('--peft', type=str, default=None, help='Path to saved LoRA layers.')
    parser.add_argument("--data_dir", type=str, default=None,
        help="Folder with train.jsonl/dev.jsonl/test.jsonl (produced by chat prep).")
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f'Neural network name prefix (default: {NN_NAME_PREFIX}).')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'LLM temperature controls randomness in output generation (default: {TEMPERATURE}).')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help=f'LLM top_k limits token selection in output generation (default: {TOP_K}).')
    parser.add_argument('--top_p', type=float, default=TOP_P,
                        help=f'LLM top_p controls token diversity in output generation (default: {TOP_P}).')
    
    # Pipeline-specific overrides (optional - for backward compatibility)
    parser.add_argument('--evaluation_strategy', type=str, default=None,
                        help=f"[Pipeline] Evaluation strategy during training (default: None, uses epoch-based for standalone).")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None,
                        help=f"[Pipeline] Per device eval batch size (default: None, uses same as train batch size).")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help=f"[Pipeline] Evaluate every N steps (default: None).")
    parser.add_argument('--save_strategy', type=str, default=None,
                        help=f"[Pipeline] Save strategy during training (default: None).")
    parser.add_argument('--save_steps', type=int, default=None,
                        help=f"[Pipeline] Save checkpoint every N steps (default: None).")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help=f"[Pipeline] Maximum number of checkpoints to keep (default: None).")
    parser.add_argument('--load_best_model_at_end', action=argparse.BooleanOptionalAction, default=False,
                        help=f"[Pipeline] Load best model at end of training (default: False).")
    parser.add_argument('--metric_for_best_model', type=str, default=None,
                        help=f"[Pipeline] Metric to use for selecting best model (default: None).")
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help=f"[Pipeline] Warmup steps override (default: None, uses warmup_ratio for standalone).")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help=f"[Pipeline] Weight decay for regularization (default: None).")

    args = parser.parse_args()
    main(num_train_epochs=args.num_train_epochs,
         lr_scheduler=args.lr_scheduler,
         max_grad_norm=args.max_grad_norm,
         tune_layers=range(args.start_layer, args.end_layer),
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
         # Pipeline overrides (optional)
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

         )

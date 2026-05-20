import argparse
from typing import Literal

import torch
from ab.gpt.util.Const import nngpt_dir, NN_TRAIN_EPOCHS

from ab.nn.util.Const import out_dir

import json
from ab.gpt.util.Const import conf_llm_dir

RUN_META = out_dir / 'nngpt' / 'run_config.json'


def persist_llm_conf(llm_conf, enable_merge=False):
    """Save run configuration including llm_conf, base_model_name, and enable_merge."""

    RUN_META.parent.mkdir(parents=True, exist_ok=True)

    # Read base_model_name from llm_conf
    base_model_name = None
    llm_conf_path = conf_llm_dir / llm_conf

    if llm_conf_path.exists():
        try:
            with open(llm_conf_path) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name")
        except Exception as e:
            print(f"Failed to read base_model_name from {llm_conf}: {e}")

    # Save llm_conf, base_model_name, and enable_merge
    run_config = {
        "llm_conf": llm_conf,
        "enable_merge": enable_merge
    }
    if base_model_name:
        run_config["base_model_name"] = base_model_name

    with open(RUN_META, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Run config saved: {RUN_META}")
# --- Default Evaluation Parameters ---
START_LAYER = 0
END_LAYER = 24
TUNE_LAYERS = range(START_LAYER, END_LAYER)
R = 32  # dimension of the updated matrices
LORA_ALPHA = 32  # parameter for scaling
LORA_DROPOUT = 0.05  # dropout probability for layers
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj')  # , 'lm_head'
TASK_TYPE = 'CAUSAL_LM'
BiasType = Literal['none', 'all', 'lora_only']
BIAS: BiasType = 'none'
LEARNING_RATE = 1e-6  # Conservative default for standalone
MAX_GRAD_NORM = 1.0  # Gradient clipping
ENABLE_MERGE = False
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
LLM_TUNE_CONF = 'NN_gen.json'  # 'Transform_gen.json' for transform fine-tune
NN_GEN_CONF = 'NN_gen.json'  # 'Transform_gen.json'
NN_GEN_CONF_ID = 'improve_classification_only'
LLM_CONF = 'ds_coder_7b_olympic.json'
MAX_PROMPTS = 4 * 1024
MAX_NEW_TOKENS = 16 * 1024
SAVE_LLM_OUTPUT = True
USE_DEEPSPEED = False
NN_NAME_PREFIX = None
TEMPERATURE = 0.8
TOP_K = 70
TOP_P = 0.9
TEST_METRIC = None  # 'bleu' or other metric for evaluation
ONNX_RUN = False
UNSLOTH_OPT = False
TRANS_MODE = False  # only transform fine-tuning
PROMPT_BATCH = 2

# --- LangGraph Agent Defaults ---
USE_AGENTS = True
USE_PREDICTOR = False

# --- Pipeline-Optimized Defaults (for iterative_finetune.py) ---
# These defaults are optimized for multi-cycle iterative fine-tuning
PIPELINE_LEARNING_RATE = 1e-5  # Conservative for stability (vs standalone 1e-6)
PIPELINE_WEIGHT_DECAY = 0.01  # Regularization
PIPELINE_WARMUP_STEPS = 20  # ~2% of samples for stable start
PIPELINE_NUM_TRAIN_EPOCHS = 5  # More epochs per cycle (vs standalone 3)
PIPELINE_LOGGING_STEPS = 10  # More frequent logging (vs standalone 96)
PIPELINE_TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj')  # Extended to include MLP
PIPELINE_MAX_NEW_TOKENS = 8192  # Balanced length (vs standalone 16*1024)
PIPELINE_TEMPERATURE = 0.2  # More deterministic (vs standalone 0.8)
PIPELINE_TOP_K = 50  # Less randomness (vs standalone 70)
PIPELINE_EVAL_STEPS = 100  # Less frequent evaluation to reduce memory spikes
PIPELINE_SAVE_STEPS = 100  # Match eval_steps for consistency
PIPELINE_SAVE_TOTAL_LIMIT = 3  # Keep last 3 checkpoints
PIPELINE_PER_DEVICE_EVAL_BATCH_SIZE = 1  # Reduce eval batch size to save memory
PIPELINE_EVALUATION_STRATEGY = 'steps'
PIPELINE_LOAD_BEST_MODEL_AT_END = True
PIPELINE_METRIC_FOR_BEST_MODEL = 'eval_loss'


def get_pipeline_defaults():
    return {
        'learning_rate': PIPELINE_LEARNING_RATE,
        'weight_decay': PIPELINE_WEIGHT_DECAY,
        'warmup_steps': PIPELINE_WARMUP_STEPS,
        'num_train_epochs': PIPELINE_NUM_TRAIN_EPOCHS,
        'logging_steps': PIPELINE_LOGGING_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,  # Same as standalone
        'target_modules': ','.join(PIPELINE_TARGET_MODULES),  # Convert tuple to comma-separated string
        'max_new_tokens': PIPELINE_MAX_NEW_TOKENS,
        'temperature': PIPELINE_TEMPERATURE,
        'top_k': PIPELINE_TOP_K,
        'evaluation_strategy': PIPELINE_EVALUATION_STRATEGY,
        'eval_steps': PIPELINE_EVAL_STEPS,
        'per_device_eval_batch_size': PIPELINE_PER_DEVICE_EVAL_BATCH_SIZE,
        'save_strategy': PIPELINE_EVALUATION_STRATEGY,  # Same as evaluation_strategy
        'save_steps': PIPELINE_SAVE_STEPS,
        'save_total_limit': PIPELINE_SAVE_TOTAL_LIMIT,
        'load_best_model_at_end': PIPELINE_LOAD_BEST_MODEL_AT_END,
        'metric_for_best_model': PIPELINE_METRIC_FOR_BEST_MODEL,
    }


def _best_dtype_args():
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16_ok, "fp16": not bf16_ok}


def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=NN_GEN_CONF, nn_gen_conf_id=NN_GEN_CONF_ID,
         llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES, per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, warmup_ratio=WARMUP_RATIO, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER,
         max_prompts=MAX_PROMPTS, save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=MAX_NEW_TOKENS, use_deepspeed=USE_DEEPSPEED, nn_name_prefix=NN_NAME_PREFIX,
         nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, data_dir=None,base_data_dir=None,output_dir=None,
         # Pipeline-specific overrides (for backward compatibility with iterative_finetune.py)
         evaluation_strategy=None, eval_steps=None, save_strategy=None, save_steps=None,
         save_total_limit=None, load_best_model_at_end=False, metric_for_best_model=None, warmup_steps=None, weight_decay=None,
         per_device_eval_batch_size=None, onnx_run=ONNX_RUN, unsloth_opt=UNSLOTH_OPT, trans_mode=TRANS_MODE,
         prompt_batch=PROMPT_BATCH, enable_merge=False,
         # --- Pipeline Hyperparameters ---
         run_iterative_pipeline=False, cycles=5, models_per_cycle=150, samples_per_prompt=1, accuracy_threshold=0.40,
         min_selected_k=15, fallback_threshold=0.35, adaptive_threshold=False,
         novelty_check=True, resume_from_cycle=None, max_retries=3, use_optimized_training=True,
         use_agents=USE_AGENTS, use_predictor=USE_PREDICTOR, use_backbone=False,
         classification_mode=False):
    persist_llm_conf(llm_conf, enable_merge)
    # --- Pipeline mode intercept ---
    if run_iterative_pipeline:
        print("--- Initiating Iterative Fine-Tuning Pipeline ---")
        from ab.gpt.iterative_finetune import IterativeFinetuner
        pipeline = IterativeFinetuner(
            llm_conf=llm_conf,
            cycles=cycles,
            models_per_cycle=models_per_cycle,
            samples_per_prompt=samples_per_prompt,
            accuracy_threshold=accuracy_threshold,
            min_selected_k=min_selected_k,
            fallback_threshold=fallback_threshold,
            adaptive_threshold=adaptive_threshold,
            novelty_check=novelty_check,
            resume_from_cycle=resume_from_cycle,
            max_retries=max_retries,
            use_optimized_training=use_optimized_training,
            num_train_epochs=num_train_epochs,
        )
        pipeline.run()
        return  # Skip standalone training

    UNSLOTH_AVAILABLE = False
    if unsloth_opt:
        try:
            import unsloth
            UNSLOTH_AVAILABLE = True
        except:
            pass

    from peft import LoraConfig
    from transformers import TrainingArguments

    if onnx_run:
        from ab.gpt.util.Tune_Onnx import tune, ds_conf
    else:
        from ab.gpt.util.Tune import tune, ds_conf

    print(f'''All hyperparameters:
num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler}, max_grad_norm={max_grad_norm}, tune_layers={tune_layers}, test_metric={test_metric},
r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, target_modules={target_modules}, task_type={task_type}, bias={bias},
learning_rate={learning_rate}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf}, nn_gen_conf_id={nn_gen_conf_id},
llm_conf={llm_conf}, test_nn={test_nn}, nn_train_epochs={nn_train_epochs}, peft={peft}, skip_epoches={skip_epoches},
per_device_train_batch_size={per_device_train_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, warmup_ratio={warmup_ratio},
logging_steps={logging_steps}, optimizer={optimizer}, max_prompts={max_prompts}, save_llm_output={save_llm_output}, max_new_tokens={max_new_tokens},
use_deepspeed={use_deepspeed}, nn_name_prefix={nn_name_prefix}, temperature={temperature}, top_k={top_k}, top_p={top_p}, onnx_run={onnx_run},
unsloth_opt={unsloth_opt}, trans_mode={trans_mode}, prompt_batch={prompt_batch}, use_agents={use_agents}, use_predictor={use_predictor}''')

    test_prm = {
        'metric_for_best_model': test_metric,
        'greater_is_better': True,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 3,
        'load_best_model_at_end': False
    } if test_metric else {}

    # Detect best mixed precision dtype (aligns with model loading which uses bfloat16)
    # This fixes the mismatch where model is loaded in bfloat16 but training used fp16,
    # which caused: NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
    dtype_flags = _best_dtype_args()

    # Build TrainingArguments kwargs
    # Pipeline mode (when evaluation_strategy is passed): use step-based eval with pipeline overrides
    # Standalone mode: use epoch-based eval with test_metric
    if evaluation_strategy is not None:
        training_kwargs = {
            'report_to': [],
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'learning_rate': learning_rate,
            'bf16': True,  # Use bf16 to match Unsloth's bfloat16 compute dtype
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            'max_grad_norm': max_grad_norm,
            'num_train_epochs': num_train_epochs,
            **dtype_flags,
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
        training_kwargs = {
            'num_train_epochs': num_train_epochs,
            'lr_scheduler_type': lr_scheduler,
            'max_grad_norm': max_grad_norm,
            'report_to': [],
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_ratio': warmup_ratio,
            'learning_rate': learning_rate,
            'bf16': True,  # Use bf16 to match Unsloth's bfloat16 compute dtype
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            **dtype_flags,
            **test_prm
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

    print(f"[LoRA Config] target_modules={target_modules}")
    print(f"[LoRA Config] layers_to_transform={list(tune_layers)}")
    print(f"[LoRA Config] r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")

    try:
        from peft import get_peft_model
        import copy
        if not target_modules or len(target_modules) == 0:
            raise ValueError("target_modules cannot be empty")
        layer_list = list(tune_layers)
        if layer_list and (min(layer_list) < 0):
            raise ValueError(f"Invalid layer indices: layers must be >= 0, got {layer_list}")
        _probe = copy.deepcopy(peft_config)
        print(f"[VALID] peft_config: r={r}, target_modules={target_modules}, layers={layer_list}")
    except Exception as e:
        print(f"[WARN] peft_config validation warning: {e}")

    print(f"\n[DEBUG] === TUNENNGEN MAIN START ===")
    print(f"[DEBUG] llm_conf: {llm_conf}")
    print(f"[DEBUG] enable_merge: {enable_merge}")
    print(f"[DEBUG] nngpt_dir: {nngpt_dir}")

    # Show what was written to config
    run_config_path = out_dir / 'nngpt' / 'run_config.json'
    if run_config_path.exists():
        with open(run_config_path) as f:
            cfg = json.load(f)
        print(f"[CONFIG] run_config.json current state:")
        print(f"[CONFIG]   base_model_name: {cfg.get('base_model_name')}")
        print(f"[CONFIG]   enable_merge: {cfg.get('enable_merge')}")
    print(f"[DEBUG] === ===\n")
    try:
        tune(
            test_nn, nn_train_epochs, skip_epoches, peft,
            llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf,
            training_args, peft_config,
            max_prompts=max_prompts,
            save_llm_output=save_llm_output,
            max_new_tokens=max_new_tokens,
            nn_name_prefix=nn_name_prefix,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            onnx_run=onnx_run,
            trans_mode=trans_mode,
            prompt_batch=prompt_batch,
            use_agents=use_agents,
            use_predictor=use_predictor,
            enable_merge=enable_merge,
            use_unsloth=unsloth_opt,
            classification_mode=classification_mode,
        )

        # Normal completion - auto merge best
        if enable_merge:
            print("\n[DEBUG] === NORMAL COMPLETION MERGE ===")
            print(f"[DEBUG] enable_merge is True")
            print(f"[DEBUG] About to import Merge module")
            print("\n[MERGE] Training complete - running auto merge...\n")
            try:
                print(f"[DEBUG] Importing from ab.gpt.util.Merge")
                from ab.gpt.util.MergeLLM import rebuild_from_lineage
                print(f"[DEBUG] ✓ Import successful")
                print(f"[DEBUG] Calling rebuild_from_lineage()")
                rebuild_from_lineage()
                print(f"[DEBUG] ✓ rebuild_from_lineage() completed")
            except Exception as e:
                print(f"[MERGE] Auto merge failed: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training stopped by user")

    finally:
        # Interrupted case - still try to merge best from available epochs
        if enable_merge:
            print("\n[DEBUG] === FINALLY BLOCK MERGE ===")
            print(f"[DEBUG] enable_merge is True in finally block")
            print("\n[MERGE] Running emergency merge (interrupted)...\n")
            try:
                print(f"[DEBUG] Importing from ab.gpt.util.Merge in finally")
                from ab.gpt.util.MergeLLM import rebuild_from_lineage
                print(f"[DEBUG] ✓ Import successful in finally")
                print(f"[DEBUG] Calling rebuild_from_lineage() from finally")
                rebuild_from_lineage()
                print(f"[DEBUG] ✓ rebuild_from_lineage() completed in finally")
            except Exception as e:
                print(f"[MERGE] Emergency merge failed: {e}")
                import traceback
                traceback.print_exc()

    print("FINE-TUNING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"✓ LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target={target_modules}")

    # Show warmup based on what was actually used
    if evaluation_strategy is not None:
        warmup_display = f"warmup_steps={warmup_steps}" if warmup_steps is not None else f"warmup_ratio={warmup_ratio}"
        wd_display = f", wd={weight_decay}" if weight_decay is not None else ""
        print(
            f"✓ Training (PIPELINE): {warmup_display}, lr={learning_rate}{wd_display}, batch={per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size * gradient_accumulation_steps}")
        print(f"✓ Evaluation: strategy={evaluation_strategy}, steps={eval_steps}")
        print(f"✓ Checkpointing: save_strategy={save_strategy}, save_steps={save_steps}, total_limit={save_total_limit}")
        if load_best_model_at_end:
            print(f"✓ Best model selection: metric={metric_for_best_model}")
    else:
        # Standalone mode
        print(
            f"✓ Training (STANDALONE): epochs={num_train_epochs}, scheduler={lr_scheduler}, warmup_ratio={warmup_ratio}, lr={learning_rate}, batch={per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size * gradient_accumulation_steps}")
        if test_metric:
            print(f"✓ Test metric: {test_metric} (epoch-based evaluation)")

    print(f"✓ Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_new_tokens}")
    print(f"✓ Data: Completion-only training, NO packing (unique code generation)")
    print("=" * 70)


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TARGET_MODULES)
    parser = argparse.ArgumentParser(description='Evaluate Neural Networks generated by NNAlter.py.')

    # Iterative pipeline mode flag
    parser.add_argument('--run_iterative_pipeline', action='store_true', default=False,
                        help='Run the full iterative fine-tuning pipeline instead of standalone fine-tuning')

    # Iterative pipeline-specific arguments (only used when --run_iterative_pipeline is set)
    parser.add_argument("--base_data_dir", type=str, default=None,
                        help="[Pipeline] Path to original chat_data directory (required when --run_iterative_pipeline is set)")
    parser.add_argument("--output_dir", type=str, default="out/iterative_cycles",
                        help="[Pipeline] Output directory for all cycle results (default: out/iterative_cycles)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="[Pipeline] Number of fine-tuning cycles to run (default: 5)")
    parser.add_argument("--models_per_cycle", type=int, default=150,
                        help="[Pipeline] Number of models to generate per cycle (default: 150)")
    parser.add_argument("--samples_per_prompt", type=int, default=1,
                        help="[Pipeline] Number of models to generate per prompt (default: 1). ")
    parser.add_argument("--accuracy_threshold", type=float, default=0.40,
                        help="[Pipeline] Minimum first-epoch accuracy to select models (0.0-1.0, default: 0.40)")
    parser.add_argument("--min_selected_k", type=int, default=15,
                        help="[Pipeline] Minimum models to select via fallback (default: 15)")
    parser.add_argument("--fallback_threshold", type=float, default=0.35,
                        help="[Pipeline] Lower bound accuracy for fallback selection (default: 0.35)")
    parser.add_argument("--adaptive_threshold", action="store_true", default=False,
                        help="[Pipeline] Enable adaptive threshold (60-70th percentile)")
    parser.add_argument("--novelty_check", action="store_true", default=True,
                        help="[Pipeline] Enable novelty checking (default: True)")
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false",
                        help="[Pipeline] Disable novelty checking")
    parser.add_argument("--resume_from_cycle", type=int, default=None,
                        help="[Pipeline] Resume pipeline from a specific cycle (1-cycles)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="[Pipeline] Maximum retry attempts for transient failures (default: 3)")
    parser.add_argument("--use_optimized_training", action="store_true", default=True,
                        help="[Pipeline] Use optimized training hyperparameters for stability and quality (default: True)")
    parser.add_argument("--no_optimized_training", dest="use_optimized_training", action="store_false",
                        help="[Pipeline] Use original default training hyperparameters")
    # Note: --num_train_epochs is already defined below for standalone mode, but pipeline can override it

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
                        help=f'Index of the first fine-tuned layer (default: {START_LAYER}).')
    parser.add_argument('-e', '--end_layer', type=int, default=END_LAYER,
                        help=f'Index of the last fine-tuned layer (default: {END_LAYER}).')
    parser.add_argument('-r', '--r', type=int, default=R,
                        help=f'Dimension of the updated matrices (default: {R}).')
    parser.add_argument('-a', '--lora_alpha', type=float, default=LORA_ALPHA,
                        help=f'LoRA alpha parameter (default: {LORA_ALPHA}).')
    parser.add_argument('-d', '--lora_dropout', type=float, default=LORA_DROPOUT,
                        help=f'LoRA dropout (default: {LORA_DROPOUT}).')
    parser.add_argument('-t', '--target_modules', type=lambda s: s.split(','), default=TARGET_MODULES,
                        help=f'Target modules (default: {TARGET_MODULES_STR})')
    parser.add_argument('-l', '--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE}).')
    parser.add_argument('-y', '--task_type', type=str, default=TASK_TYPE,
                        help=f'LLM task type (default: {TASK_TYPE}).')
    parser.add_argument('-b', '--bias', type=str, default=BIAS,
                        help=f'Bias type (default: {BIAS}).')

    # Config files
    parser.add_argument('--llm_tune_conf', type=str, default=LLM_TUNE_CONF,
                        help=f'Config for LLM fine-tuning (default: {LLM_TUNE_CONF}).')
    parser.add_argument('--nn_gen_conf', type=str, default=NN_GEN_CONF,
                        help=f'Config for NN generation (default: {NN_GEN_CONF}).')
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID,
                        help=f'Prompt key for NN generation (default: {NN_GEN_CONF_ID}).')
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF,
                        help=f'Config of LLM (default: {LLM_CONF}).')

    # Training configuration
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN,
                        help=f'Count of NNs to generate (default: {TEST_NN}).')
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f'Training epochs for generated NN (default: {NN_TRAIN_EPOCHS}).')
    parser.add_argument('-m', '--max_prompts', type=int, default=MAX_PROMPTS,
                        help=f'Max prompts for fine-tuning (default: {MAX_PROMPTS}).')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS,
                        help=f'Max tokens in LLM output (default: {MAX_NEW_TOKENS}).')
    parser.add_argument('--use_deepspeed', action='store_true',
                        help=f'Use DeepSpeed (default: {USE_DEEPSPEED}).')
    parser.add_argument('--per_device_train_batch_size', type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f'Per device train batch size (default: {PER_DEVICE_TRAIN_BATCH_SIZE}).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help=f'Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}).')
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO,
                        help=f'Warmup ratio (default: {WARMUP_RATIO}).')
    parser.add_argument('--logging_steps', type=int, default=LOGGING_STEPS,
                        help=f'Logging steps (default: {LOGGING_STEPS}).')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help=f'Optimizer (default: {OPTIMIZER}).')
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES,
                        help='Epochs to skip NN generation.')
    parser.add_argument('--peft', type=str, default=None, help='Path to saved LoRA layers.')
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Folder with train.jsonl/dev.jsonl/test.jsonl (produced by chat prep).")
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f'NN name prefix (default: {NN_NAME_PREFIX}).')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'LLM temperature (default: {TEMPERATURE}).')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help=f'LLM top_k (default: {TOP_K}).')
    parser.add_argument('--top_p', type=float, default=TOP_P,
                        help=f'LLM top_p controls token diversity in output generation (default: {TOP_P}).')

    # Pipeline-specific overrides (optional - for backward compatibility)
    parser.add_argument('--evaluation_strategy', type=str, default=None,
                        help="[Pipeline] Evaluation strategy (default: None).")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None,
                        help="[Pipeline] Per device eval batch size (default: None).")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="[Pipeline] Evaluate every N steps (default: None).")
    parser.add_argument('--save_strategy', type=str, default=None,
                        help="[Pipeline] Save strategy (default: None).")
    parser.add_argument('--save_steps', type=int, default=None,
                        help="[Pipeline] Save checkpoint every N steps (default: None).")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help="[Pipeline] Max checkpoints to keep (default: None).")
    parser.add_argument('--load_best_model_at_end', action=argparse.BooleanOptionalAction, default=False,
                        help="[Pipeline] Load best model at end (default: False).")
    parser.add_argument('--metric_for_best_model', type=str, default=None,
                        help="[Pipeline] Metric for best model (default: None).")
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help="[Pipeline] Warmup steps override (default: None).")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="[Pipeline] Weight decay (default: None).")
    parser.add_argument('--trans_mode', action='store_true',
                        help=f"Transform mode only (default: {TRANS_MODE}).")
    parser.add_argument('--onnx_run', action='store_true',
                        help=f"ONNX format (default: {ONNX_RUN}).")
    parser.add_argument('--unsloth_opt', action='store_true',
                        help=f"Use Unsloth optimizations (default: {UNSLOTH_OPT}).")
    parser.add_argument("--enable_merge", action="store_true", default=False, help="Enable automatic merge decision after fine-tuning.")
    parser.add_argument('--prompt_batch', type=int, default=PROMPT_BATCH,
                        help=f"Prompt batch size (default: {PROMPT_BATCH}).")
    parser.add_argument('--use_agents', action='store_false', default=USE_AGENTS,
                        help='Enable LangGraph multi-agent workflow (default: False).')
    parser.add_argument('--use_predictor', action='store_true', default=USE_PREDICTOR,
                        help='Enable predictor agent (requires --use_agents) (default: False).')

    args = parser.parse_args()

    # Convert start_layer/end_layer → tune_layers (main() expects a range, not two ints)
    kwargs = vars(args)
    kwargs['tune_layers'] = range(kwargs.pop('start_layer'), kwargs.pop('end_layer'))

    main(**kwargs)
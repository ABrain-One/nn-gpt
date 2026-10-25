import argparse
import os
import sys
from typing import Literal

from ab.gpt.util.Caches import setup_caches
setup_caches()

import torch
from ab.gpt.util.Const import nngpt_dir, NN_TRAIN_EPOCHS, DEFAULT_DATASET, DEFAULT_NN_PREFIXES

from ab.nn.util.Const import out_dir

import json
from datetime import datetime
from pathlib import Path
from ab.gpt.util.Const import conf_llm_dir

RUN_META = out_dir / 'nngpt' / 'run_config.json'


def persist_llm_conf(llm_conf, enable_merge=False, params=None,
                     nn_gen_conf=None, nn_gen_conf_id=None, llm_tune_conf=None):
    """Save run configuration including llm_conf, base_model_name, and enable_merge.

    ``params`` carries the fine-tuning settings so the plot can state which model
    and which hyper-parameters produced it. Statistics without the configuration
    that produced them cannot be compared between runs.

    ``nn_gen_conf`` names the generation prompt config, which is where the task
    and the dataset are recorded. They are copied in under the keys the dataset
    itself uses, so the figure can be matched against the database.

    The two prompt configs are recorded by name as well: ``llm_tune_conf`` is
    what the model was fine-tuned on and ``nn_gen_conf`` is what it was asked
    to generate from. Two runs of identical hyper-parameters on different
    prompts produce different curves, and nothing else distinguishes them.

    ``script`` is the entry point this run was started from. The repository
    carries a script per model and per experiment, and two of them can produce
    the same run_config.json while differing in everything the config does not
    record, so the figure names the one that ran.

    ``started_at`` is stamped here, at the one moment that is the start of the
    run for certain. The epoch rows carry the time each epoch finished, so the
    two together are what the figure states the run cost; without the stamp the
    first epoch -- generation, training and evaluation -- is uncounted.
    """

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
        "enable_merge": enable_merge,
        "started_at": datetime.now().isoformat(),
    }
    if base_model_name:
        run_config["base_model_name"] = base_model_name
    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    if script and not script.startswith("-"):      # -c, -m: no script to name
        run_config["script"] = script
    prompts = {"train": llm_tune_conf, "test": nn_gen_conf,
               "test_id": nn_gen_conf_id}
    prompts = {k: v for k, v in prompts.items() if v}
    if prompts:
        run_config["prompts"] = prompts
    if params:
        run_config["training"] = {k: v for k, v in params.items() if v is not None}
    if nn_gen_conf:
        try:
            from ab.gpt.util.Const import conf_test_dir
            with open(conf_test_dir / nn_gen_conf) as f:
                gen_cfg = json.load(f)
            key = nn_gen_conf_id if isinstance(nn_gen_conf_id, str) else None
            entry = gen_cfg.get(key) if key else next(iter(gen_cfg.values()), None)
            if isinstance(entry, dict):
                for field in ("task", "dataset"):
                    if entry.get(field):
                        run_config[field] = entry[field]
        except Exception as e:
            print(f"Failed to read task/dataset from {nn_gen_conf}: {e}")

    with open(RUN_META, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Run config saved: {RUN_META}")
# --- Default Evaluation Parameters ---
# None now means "adapt every layer"; pass -s/-e to restrict it.
START_LAYER = None
END_LAYER = None
TUNE_LAYERS = None
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
WARMUP_STEPS = 5  # Warmup as ratio of total steps


def _warmup_steps_arg(value):
    """--warmup_steps: whole numbers stay int, because transformers 4.x TrainingArguments rejects a float
    warmup_steps ("must be of type int"); other values stay float (a ratio for transformers 5.x)."""
    number = float(value)
    return int(number) if number.is_integer() else number
TEST_NN = 10
SFT_MAX_LENGTH = 4096  # Packing/truncation budget for supervised fine-tuning
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
USE_UNSLOTH = False
TRANS_MODE = False  # only transform fine-tuning
PROMPT_BATCH = 2

# --- LangGraph Agent Defaults ---
USE_AGENTS = True
USE_PREDICTOR = False

# --- Run-mode defaults ---
# Delete the previous run's cycle_results*.json / epoch_tracker.json /
# epoch_statistics.png before starting, so one run's statistics cannot be read
# as another's. Ignored when resuming.
RESET_PREVIOUS_RESULTS = True
USE_VLLM = True
from ab.gpt.util.eval.Seed import PIN_SEEDS
SUPPRESS_THINKING = False
SKIP_LM_FINETUNE = False
USE_BACKBONE = False
CLASSIFICATION_MODE = False
ONLY_BEST_ACCURACY = False
LOAD_IN_4BIT = True
LOAD_BEST_MODEL_AT_END = False

# --- Iterative pipeline defaults (--run_iterative_pipeline) ---
RUN_ITERATIVE_PIPELINE = False
CYCLES = 5
MODELS_PER_CYCLE = 150
SAMPLES_PER_PROMPT = 1
ACCURACY_THRESHOLD = 0.40
MIN_SELECTED_K = 15
FALLBACK_THRESHOLD = 0.35
ADAPTIVE_THRESHOLD = False
NOVELTY_CHECK = True
RESUME_FROM_CYCLE = None
MAX_RETRIES = 3
USE_OPTIMIZED_TRAINING = True

# --- LLMatic MAP-Elites seed selection ---
LLMATIC = False
LLMATIC_CROSSOVER_EVERY = 3

# --- Mobile deployment defaults ---
MOBILE_DEPLOYMENT = False
MOBILE_REVAL_ONLY = False
MOBILE_MIN_QUANTIZED_ACCURACY = None
MOBILE_MAX_DURATION_MS = None
MOBILE_SCORE_TOLERANCE = 0.99
MOBILE_MIN_VALID_MODELS = 5
MOBILE_DELEGATE_PRIORITY = 'npu,gpu,cpu'

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
    """Detect the best mixed precision dtype based on hardware support."""
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16_ok:
        return {"bf16": True}
    else:
        return {"fp16": True}


def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=None, nn_gen_conf_id=NN_GEN_CONF_ID,
         llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES, per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, warmup_steps=WARMUP_STEPS, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER,
         report_to=None, logging_dir=None,
         max_prompts=MAX_PROMPTS, save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=None, use_deepspeed=USE_DEEPSPEED, nn_name_prefix=NN_NAME_PREFIX,
         nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, data_dir=None,
         num_cycles=None, epoch_root=None,
         suppress_thinking=SUPPRESS_THINKING, budget_tokens=None, skip_lm_finetune=SKIP_LM_FINETUNE,
         # Pipeline-specific overrides (for backward compatibility with iterative_finetune.py)
         evaluation_strategy=None, eval_steps=None, save_strategy=None, save_steps=None,
         save_total_limit=None, load_best_model_at_end=LOAD_BEST_MODEL_AT_END, metric_for_best_model=None, weight_decay=None,
         per_device_eval_batch_size=None, onnx_run=ONNX_RUN, use_unsloth=USE_UNSLOTH, trans_mode=TRANS_MODE,
         prompt_batch=PROMPT_BATCH, enable_merge=ENABLE_MERGE,
         # --- Pipeline Hyperparameters ---
         run_iterative_pipeline=RUN_ITERATIVE_PIPELINE, cycles=CYCLES, models_per_cycle=MODELS_PER_CYCLE,
         samples_per_prompt=SAMPLES_PER_PROMPT, accuracy_threshold=ACCURACY_THRESHOLD,
         min_selected_k=MIN_SELECTED_K, fallback_threshold=FALLBACK_THRESHOLD, adaptive_threshold=ADAPTIVE_THRESHOLD,
         novelty_check=NOVELTY_CHECK, resume_from_cycle=RESUME_FROM_CYCLE, max_retries=MAX_RETRIES,
         use_optimized_training=USE_OPTIMIZED_TRAINING,
         use_agents=USE_AGENTS, use_predictor=USE_PREDICTOR, use_backbone=USE_BACKBONE,
         classification_mode=CLASSIFICATION_MODE, context_length=None, max_input_length=None,
         only_best_accuracy=ONLY_BEST_ACCURACY, load_in_4bit=LOAD_IN_4BIT, sft_max_length=SFT_MAX_LENGTH,
         reset_previous_results=RESET_PREVIOUS_RESULTS, pin_seeds=PIN_SEEDS,
         use_vllm=USE_VLLM,
         mobile_deployment=MOBILE_DEPLOYMENT, mobile_reval_only=MOBILE_REVAL_ONLY,
         mobile_min_quantized_accuracy=MOBILE_MIN_QUANTIZED_ACCURACY, mobile_max_duration_ms=MOBILE_MAX_DURATION_MS,
         mobile_score_tolerance=MOBILE_SCORE_TOLERANCE, mobile_min_valid_models=MOBILE_MIN_VALID_MODELS,
         mobile_delegate_priority=MOBILE_DELEGATE_PRIORITY,
         # --- Corpus filters for the iterative pipeline's LEMUR curation ---
         dataset=DEFAULT_DATASET, nn_prefixes=DEFAULT_NN_PREFIXES,
         # --- LLMatic MAP-Elites seed selection ---
         llmatic=LLMATIC, llmatic_crossover_every=LLMATIC_CROSSOVER_EVERY):

    from ab.gpt.util.Logs import start_run_log
    start_run_log()

    if num_cycles is not None:
        cycles = int(num_cycles)

    import os
    os.environ["NNGPT_VLLM_SLEEP"] = "1" if use_vllm else "0"

    run_params = {
        'learning_rate': learning_rate,
        'lr_scheduler': lr_scheduler,
        'num_train_epochs': num_train_epochs,
        'cycles': cycles,
        'max_prompts': max_prompts,
        'lora_r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'optimizer': optimizer,
        'warmup_steps': warmup_steps,
        'per_device_train_batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'context_length': context_length,
        'sft_max_length': sft_max_length,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'test_nn': test_nn,
        'load_in_4bit': load_in_4bit,
        'unsloth': use_unsloth,
        'vllm_sleep': use_vllm,
    }
    persist_llm_conf(llm_conf, enable_merge, params=run_params,
                     nn_gen_conf=nn_gen_conf or NN_GEN_CONF,
                     nn_gen_conf_id=nn_gen_conf_id,
                     llm_tune_conf=llm_tune_conf)

    if run_iterative_pipeline:
        print("--- Initiating Iterative Fine-Tuning Pipeline ---")
        try:
            if mobile_deployment:
                from ab.gpt.act.edge.iterative_finetune import MobileDeploymentFinetuner as IterativeFinetuner
            else:
                from ab.gpt.act.tune.iterative.finetune import IterativeFinetuner
        except ImportError as e:
            print(f"[ERROR] Pipeline mode requires ab.gpt.act.tune.iterative.finetune: {e}")
            sys.exit(1)
        pipeline_kwargs = dict(
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
            dataset=dataset,
            nn_prefixes=nn_prefixes,
            llmatic=llmatic,
            llmatic_crossover_every=llmatic_crossover_every,
        )
        # Forward per-cycle overrides only when given explicitly
        if nn_gen_conf is not None:
            pipeline_kwargs["nn_gen_conf"] = nn_gen_conf
        if max_new_tokens is not None:
            pipeline_kwargs["max_new_tokens"] = max_new_tokens
        if mobile_deployment:
            pipeline_kwargs["mobile_min_quantized_accuracy"] = mobile_min_quantized_accuracy
            pipeline_kwargs["mobile_max_duration_ms"] = mobile_max_duration_ms
            pipeline_kwargs["mobile_score_tolerance"] = mobile_score_tolerance
            pipeline_kwargs["mobile_min_valid_models"] = mobile_min_valid_models
            pipeline_kwargs["mobile_delegate_priority"] = mobile_delegate_priority
            if mobile_reval_only:
                pipeline_kwargs["skip_mobile_seed_prep"] = True
        pipeline = IterativeFinetuner(**pipeline_kwargs)
        if mobile_deployment and mobile_reval_only:
            cycle = resume_from_cycle or 1
            result = pipeline.run_mobile_reval_cycle(cycle, clean=True)
            if not result.get("success", False):
                print(f"[ERROR] Mobile re-eval failed for cycle {cycle}: {result.get('error')}")
                sys.exit(1)
            print(f"Mobile re-eval cycle {cycle} complete.")
            return
        if pipeline.run() is False:
            sys.exit(1)
        return  # Skip standalone training

    if nn_gen_conf is None:
        nn_gen_conf = NN_GEN_CONF
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    UNSLOTH_AVAILABLE = False
    if use_unsloth:
        try:
            import unsloth
            UNSLOTH_AVAILABLE = True
        except ImportError:
            print("[WARN] Unsloth requested but not installed. Falling back to standard PEFT.")

    # Guard against broken flash_attn binary in the K8s image
    import sys, types
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError):
        import importlib.machinery
        _fa = types.ModuleType("flash_attn")
        _fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
        _fa.__version__ = "0.0.0"
        _fa.flash_attn_func = None
        _fa.flash_attn_varlen_func = None
        _fa.flash_attn_with_kvcache = None
        sys.modules["flash_attn"] = _fa
        _bp = types.ModuleType("flash_attn.bert_padding")
        _bp.index_first_axis = lambda x, i: x[i]
        _bp.pad_input = lambda *a, **k: None
        _bp.unpad_input = lambda *a, **k: (None, None, None, None)
        sys.modules["flash_attn.bert_padding"] = _bp
        sys.modules["flash_attn.layers"] = types.ModuleType("flash_attn.layers")
        _rot = types.ModuleType("flash_attn.layers.rotary")
        _rot.apply_rotary_emb = lambda *a, **k: a[0]
        sys.modules["flash_attn.layers.rotary"] = _rot

    # Guard against torchvision::nms operator mismatch (torch version in user site
    # doesn't match torchvision in venv)
    try:
        import torchvision
    except (ImportError, RuntimeError):
        pass  # torchvision not needed for fine-tuning, ignore if broken

    from peft import LoraConfig
    from transformers import TrainingArguments

    if onnx_run:
        try:
            from ab.gpt.util.tune.Tune_Onnx import tune, ds_conf
        except ImportError as e:
            print(f"[ERROR] ONNX mode requires ab.gpt.util.tune.Tune_Onnx: {e}")
            sys.exit(1)
    else:
        try:
            from ab.gpt.util.tune.Tune import tune, ds_conf
        except ImportError as e:
            print(f"[ERROR] Failed to import ab.gpt.util.tune.Tune: {e}")
            sys.exit(1)

    print(f'''All hyperparameters:
num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler}, max_grad_norm={max_grad_norm}, tune_layers={tune_layers}, test_metric={test_metric},
r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, target_modules={target_modules}, task_type={task_type}, bias={bias},
learning_rate={learning_rate}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf}, nn_gen_conf_id={nn_gen_conf_id},
llm_conf={llm_conf}, test_nn={test_nn}, nn_train_epochs={nn_train_epochs}, peft={peft}, skip_epoches={skip_epoches},
per_device_train_batch_size={per_device_train_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, warmup_steps={warmup_steps},
logging_steps={logging_steps}, optimizer={optimizer}, max_prompts={max_prompts}, save_llm_output={save_llm_output}, max_new_tokens={max_new_tokens},
use_deepspeed={use_deepspeed}, nn_name_prefix={nn_name_prefix}, temperature={temperature}, top_k={top_k}, top_p={top_p}, onnx_run={onnx_run},
use_unsloth={use_unsloth}, trans_mode={trans_mode}, prompt_batch={prompt_batch}, use_agents={use_agents}, use_predictor={use_predictor},
use_backbone={use_backbone}, enable_merge={enable_merge}, classification_mode={classification_mode}, pin_seeds={pin_seeds}''')

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
    # report_to defaults to [] (console print only, via logging_steps) unless the
    # caller opts in with --report_to (e.g. 'tensorboard') for persisted/visualizable
    # logs. logging_dir only matters when report_to is non-empty.
    report_to_list = [r.strip() for r in report_to.split(',') if r.strip()] if report_to else []
    logging_dir_path = logging_dir or (str(nngpt_dir / 'outputs' / 'runs') if report_to_list else None)

    # Pipeline mode (when evaluation_strategy is passed): use step-based eval with pipeline overrides
    # Standalone mode: use epoch-based eval with test_metric
    if evaluation_strategy is not None:
        training_kwargs = {
            'report_to': report_to_list,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'learning_rate': learning_rate,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            'max_grad_norm': max_grad_norm,
            'num_train_epochs': num_train_epochs,
            **dtype_flags,
        }
        if logging_dir_path:
            training_kwargs['logging_dir'] = logging_dir_path

        # Add warmup - pipeline may pass warmup_steps (override) or use warmup_steps
        if warmup_steps is not None:
            training_kwargs['warmup_steps'] = warmup_steps
        else:
            training_kwargs['warmup_steps'] = warmup_steps

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
            'report_to': report_to_list,
            'per_device_train_batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_steps': warmup_steps,
            'learning_rate': learning_rate,
            'logging_steps': logging_steps,
            'output_dir': nngpt_dir / 'outputs',
            'optim': optimizer,
            'deepspeed': ds_conf if use_deepspeed else None,
            'gradient_checkpointing': True,
            **dtype_flags,
            **test_prm
        }
        if logging_dir_path:
            training_kwargs['logging_dir'] = logging_dir_path
        if weight_decay is not None:
            training_kwargs['weight_decay'] = weight_decay

    # Create TrainingArguments with all parameters at once
    training_args = TrainingArguments(**training_kwargs)
    if sft_max_length:
        training_args.max_length = int(sft_max_length)
    # FIX: Ensure int types for LoRA config
    peft_config = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        layers_to_transform=list(tune_layers) if tune_layers is not None else None,
        lora_dropout=float(lora_dropout),
        bias=bias,
        task_type=task_type)

    print(f"[LoRA Config] target_modules={target_modules}")
    print(f"[LoRA Config] layers_to_transform="
          f"{list(tune_layers) if tune_layers is not None else 'ALL'}")
    print(f"[LoRA Config] r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")

    try:
        import copy
        if not target_modules or len(target_modules) == 0:
            raise ValueError("target_modules cannot be empty")
        layer_list = list(tune_layers) if tune_layers is not None else None
        if layer_list and (min(layer_list) < 0):
            raise ValueError(f"Invalid layer indices: layers must be >= 0, got {layer_list}")
        _probe = copy.deepcopy(peft_config)
        print(f"[VALID] peft_config: r={r}, target_modules={target_modules}, layers={layer_list}")
    except Exception as e:
        print(f"[WARN] peft_config validation warning: {e}")

    # Clear the previous run's statistics before anything writes new ones.
    # This used to be a keyword argument to tune(), but this module dispatches
    # to several tune() implementations (Tune, Tune_Onnx, Tune_prun) that do
    # not share a signature, so passing it raised
    #   TypeError: tune() got an unexpected keyword argument
    # for any of them that had not been updated. Doing it at the call site
    # keeps every backend working and needs no signature change.
    import os
    resuming = bool(peft or (skip_epoches or 0) > 0
                    or os.environ.get("NNGPT_TRAIN_RESUME_TRAINER_CHECKPOINT"))
    # Seed pinning is configured here for the same reason: nn_gen() reads it,
    # but tune() cannot carry a new keyword without breaking the Tune_Onnx and
    # Tune_prun backends, which do not share its signature. An environment
    # variable reaches nn_gen() through every backend and through the worker
    # processes it spawns.
    try:
        from ab.gpt.util.eval.Seed import set_pinning, reset_pinned_seeds
        set_pinning(pin_seeds)
        if pin_seeds:
            # A new run draws its own seeds; resuming keeps the pinned set so
            # the epochs before and after the break stay comparable.
            if not resuming:
                reset_pinned_seeds()
            print("[SEEDS] Generation seeds are pinned for this run"
                  + ("" if not resuming else " (resuming: keeping the existing set)"))
        else:
            print("[SEEDS] Generation seeds are redrawn every epoch "
                  "(--no-pin_seeds): epochs are not directly comparable")
    except Exception as exc:
        print(f"[SEEDS] Could not configure seed pinning: {exc}")

    if reset_previous_results and not resuming:
        try:
            from ab.gpt.util.data.EpochStats import reset_run_dir
            reset_run_dir(nngpt_dir)
            # run_config.json was written into this directory before the reset
            persist_llm_conf(llm_conf, enable_merge, params=run_params,
                             nn_gen_conf=nn_gen_conf or NN_GEN_CONF,
                             nn_gen_conf_id=nn_gen_conf_id,
                             llm_tune_conf=llm_tune_conf)
        except Exception as exc:
            print(f"[RUN RESET] Skipped: {exc}")
    elif resuming:
        print("[RUN RESET] Resuming an existing run: previous statistics kept "
              "(--peft, --skip_epoches or a trainer checkpoint is set)")
    else:
        print("[RUN RESET] Disabled: previous statistics kept")

    try:
        tune(
            test_nn, nn_train_epochs, skip_epoches, peft,
            llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf,
            training_args, peft_config,
            num_cycles=cycles,
            max_prompts=max_prompts,
            save_llm_output=save_llm_output,
            max_new_tokens=max_new_tokens,
            nn_name_prefix=nn_name_prefix,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            test_metric=test_metric,
            onnx_run=onnx_run,
            trans_mode=trans_mode,
            prompt_batch=prompt_batch,
            use_agents=use_agents,
            use_predictor=use_predictor,
            use_unsloth=use_unsloth,
            enable_merge=enable_merge,
            classification_mode=classification_mode,
            use_backbone=use_backbone,
            suppress_thinking=suppress_thinking,
            budget_tokens=budget_tokens,
            skip_lm_finetune=skip_lm_finetune,
            context_length=context_length,
            max_input_length=max_input_length,
            only_best_accuracy=only_best_accuracy,
            load_in_4bit=load_in_4bit,
            data_dir=data_dir,
            epoch_root=epoch_root,
            llmatic=({"crossover_every": llmatic_crossover_every} if llmatic else None),
            dataset=dataset,
            nn_prefixes=nn_prefixes,
        )

        # Normal completion - auto merge best
        if enable_merge:
            print("\n[MERGE] Training complete - running auto merge...\n")
            try:
                from ab.gpt.util.llm.MergeLLM import rebuild_from_lineage
                rebuild_from_lineage()
                print("[MERGE] Completed successfully.\n")
            except ImportError as e:
                print(f"[WARN] MergeLLM module not found: {e}")
            except Exception as e:
                print(f"[MERGE] Auto merge failed: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training stopped by user")

    finally:
        # Interrupted case - still try to merge best from available epochs
        if enable_merge:
            print("\n[MERGE] Running emergency merge (interrupted)...\n")
            try:
                from ab.gpt.util.llm.MergeLLM import rebuild_from_lineage
                rebuild_from_lineage()
                print("[MERGE] Emergency merge completed.\n")
            except ImportError as e:
                print(f"[WARN] MergeLLM module not found: {e}")
            except Exception as e:
                print(f"[MERGE] Emergency merge failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("FINE-TUNING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"✓ LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target={target_modules}")

    # Show warmup based on what was actually used
    if evaluation_strategy is not None:
        warmup_display = f"warmup_steps={warmup_steps}" if warmup_steps is not None else f"warmup_steps={warmup_steps}"
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
            f"✓ Training (STANDALONE): epochs={num_train_epochs}, scheduler={lr_scheduler}, warmup_steps={warmup_steps}, lr={learning_rate}, batch={per_device_train_batch_size}×{gradient_accumulation_steps}={per_device_train_batch_size * gradient_accumulation_steps}")
        if test_metric:
            print(f"✓ Test metric: {test_metric} (epoch-based evaluation)")

    print(f"✓ Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_new_tokens}")
    print(f"✓ Data: Completion-only training, NO packing (unique code generation)")
    print("=" * 70)


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TARGET_MODULES)
    LAYER_RANGE_STR = ('all layers' if TUNE_LAYERS is None
                       else f'{START_LAYER}..{END_LAYER}')
    parser = argparse.ArgumentParser(description='Evaluate Neural Networks generated by NNAlter.py.')

    # Iterative pipeline mode flag
    parser.add_argument('--run_iterative_pipeline', action='store_true', default=RUN_ITERATIVE_PIPELINE,
                        help='Run the full iterative fine-tuning pipeline instead of standalone fine-tuning')

    # Iterative pipeline-specific arguments (only used when --run_iterative_pipeline is set)
    parser.add_argument("--cycles", type=int, default=CYCLES,
                        help=f"[Pipeline] Number of fine-tuning cycles to run (default: {CYCLES})")
    parser.add_argument("--models_per_cycle", type=int, default=MODELS_PER_CYCLE,
                        help=f"[Pipeline] Number of models to generate per cycle (default: {MODELS_PER_CYCLE})")
    parser.add_argument("--samples_per_prompt", type=int, default=SAMPLES_PER_PROMPT,
                        help=f"[Pipeline] Number of models to generate per prompt (default: {SAMPLES_PER_PROMPT}). ")
    parser.add_argument("--accuracy_threshold", type=float, default=ACCURACY_THRESHOLD,
                        help=f"[Pipeline] Minimum first-epoch accuracy to select models (0.0-1.0, default: {ACCURACY_THRESHOLD})")
    parser.add_argument("--min_selected_k", type=int, default=MIN_SELECTED_K,
                        help=f"[Pipeline] Minimum models to select via fallback (default: {MIN_SELECTED_K})")
    parser.add_argument("--fallback_threshold", type=float, default=FALLBACK_THRESHOLD,
                        help=f"[Pipeline] Lower bound accuracy for fallback selection (default: {FALLBACK_THRESHOLD})")
    parser.add_argument("--adaptive_threshold", action="store_true", default=ADAPTIVE_THRESHOLD,
                        help="[Pipeline] Enable adaptive threshold (60-70th percentile)")
    parser.add_argument("--novelty_check", action="store_true", default=NOVELTY_CHECK,
                        help=f"[Pipeline] Enable novelty checking (default: {NOVELTY_CHECK})")
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false",
                        help="[Pipeline] Disable novelty checking")
    parser.add_argument("--resume_from_cycle", type=int, default=RESUME_FROM_CYCLE,
                        help="[Pipeline] Resume pipeline from a specific cycle (1-cycles)")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES,
                        help=f"[Pipeline] Maximum retry attempts for transient failures (default: {MAX_RETRIES})")
    parser.add_argument("--use_optimized_training", action="store_true", default=USE_OPTIMIZED_TRAINING,
                        help=f"[Pipeline] Use optimized training hyperparameters for stability and quality (default: {USE_OPTIMIZED_TRAINING})")
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
                        help=f'Index of the first fine-tuned layer (default: {LAYER_RANGE_STR}).')
    parser.add_argument('-e', '--end_layer', type=int, default=END_LAYER,
                        help=f'Exclusive index of the last fine-tuned layer (default: {LAYER_RANGE_STR}).')
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
    parser.add_argument('--nn_gen_conf', type=str, default=None,
                        help=f'Config for NN generation (default: {NN_GEN_CONF}).')
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID,
                        help=f'Prompt key for NN generation (default: {NN_GEN_CONF_ID}).')
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF,
                        help=f'Config of LLM (default: {LLM_CONF}).')

    # Training configuration
    parser.add_argument('--num_cycles', type=int, default=None,
                        help='Number of outer generate/eval/SFT cycles (default: 100).')
    parser.add_argument('--suppress_thinking', action='store_true', default=SUPPRESS_THINKING,
                        help='Close OlympicCoder-7B think block before generation (reduces thinking:code ratio).')
    parser.add_argument('--max_output_tokens', type=int, default=None, dest='budget_tokens',
                        help='Override max_new_tokens; enables token-budget sweep (paper Table 2).')
    parser.add_argument('--skip_lm_finetune', action='store_true', default=SKIP_LM_FINETUNE,
                        help='Skip LoRA fine-tuning step each cycle (generation-only run).')
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN,
                        help=f'Count of NNs to generate (default: {TEST_NN}).')
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f'Training epochs for generated NN (default: {NN_TRAIN_EPOCHS}).')
    parser.add_argument('-m', '--max_prompts', type=int, default=MAX_PROMPTS,
                        help=f'Max prompts for fine-tuning (default: {MAX_PROMPTS}).')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help=f'Max tokens in LLM output (default: {MAX_NEW_TOKENS}).')
    parser.add_argument('--use_deepspeed', action='store_true',
                        help=f'Use DeepSpeed (default: {USE_DEEPSPEED}).')
    parser.add_argument('--per_device_train_batch_size', type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f'Per device train batch size (default: {PER_DEVICE_TRAIN_BATCH_SIZE}).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help=f'Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}).')
    parser.add_argument('--warmup_steps', type=_warmup_steps_arg, default=WARMUP_STEPS,
                        help=f'Warmup ratio (default: {WARMUP_STEPS}).')
    parser.add_argument('--logging_steps', type=int, default=LOGGING_STEPS,
                        help=f'Logging steps (default: {LOGGING_STEPS}).')
    parser.add_argument('--report_to', type=str, default=None,
                        help="Comma-separated Trainer logging integrations, e.g. 'tensorboard' or "
                             "'tensorboard,wandb'. Default: none (console print only, via --logging_steps).")
    parser.add_argument('--logging_dir', type=str, default=None,
                        help='Directory for Trainer logs (TensorBoard event files, ...). '
                             'Only used when --report_to is set; defaults to <output_dir>/runs/<timestamp>.')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help=f'Optimizer (default: {OPTIMIZER}).')
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES,
                        help='Epochs to skip NN generation.')
    parser.add_argument('--peft', type=str, default=PEFT, help='Path to saved LoRA layers.')
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Folder with train.jsonl/dev.jsonl/test.jsonl (produced by chat prep).")
    parser.add_argument("--epoch_root", type=str, default=None,
                        help="Root dir for per-epoch generation/eval artifacts (A{n}/synth_nn/...). "
                             "Defaults to the standard epoch_dir() when unset.")
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
    parser.add_argument('--load_best_model_at_end', action=argparse.BooleanOptionalAction, default=LOAD_BEST_MODEL_AT_END,
                        help="[Pipeline] Load best model at end (default: False).")
    parser.add_argument('--metric_for_best_model', type=str, default=None,
                        help="[Pipeline] Metric for best model (default: None).")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="[Pipeline] Weight decay (default: None).")
    parser.add_argument('--trans_mode', action='store_true',
                        help=f"Transform mode only (default: {TRANS_MODE}).")
    parser.add_argument('--onnx_run', action='store_true',
                        help=f"ONNX format (default: {ONNX_RUN}).")
    parser.add_argument('--use_unsloth', action='store_true',
                        help=f"Use Unsloth optimizations (default: {USE_UNSLOTH}).")
    parser.add_argument("--enable_merge", action="store_true", default=ENABLE_MERGE, help="Enable automatic merge decision after fine-tuning.")
    parser.add_argument('--prompt_batch', type=int, default=PROMPT_BATCH,
                        help=f"Prompt batch size (default: {PROMPT_BATCH}).")
    parser.add_argument('--use_agents', action='store_true', default=USE_AGENTS,
                        help=f'Enable LangGraph multi-agent workflow (default: {USE_AGENTS}).')
    parser.add_argument('--no-use_agents', dest='use_agents', action='store_false',
                        help='Disable LangGraph multi-agent workflow during fine-tuning.')
    parser.add_argument('--use_predictor', action='store_true', default=USE_PREDICTOR,
                        help='Enable predictor agent.')
    parser.add_argument('--use_backbone', action='store_true', default=USE_BACKBONE,
                        help=f'Use backbone mode for code generation (default: {USE_BACKBONE}).')
    parser.add_argument('--classification_mode', action='store_true', default=CLASSIFICATION_MODE,
                        help=f'Enable classification-only mode (default: {CLASSIFICATION_MODE}).')
    parser.add_argument('--reset_previous_results', action=argparse.BooleanOptionalAction,
                        default=RESET_PREVIOUS_RESULTS,
                        help=f'Delete the whole out/nngpt run directory before starting '
                             f'(default: {RESET_PREVIOUS_RESULTS}), so no statistics, '
                             f'pinned seeds, epoch directories or adapters of a previous '
                             f'run survive into this one. Always skipped when resuming '
                             f'via --peft or --skip_epoches.')
    parser.add_argument('--use_vllm', action=argparse.BooleanOptionalAction,
                        default=USE_VLLM,
                        help=f'Generate with an in-process vLLM engine in sleep mode '
                             f'(default: {USE_VLLM}). Use --no-use_vllm for the '
                             f'HuggingFace generation path.')
    parser.add_argument('--pin_seeds', action=argparse.BooleanOptionalAction,
                        default=PIN_SEEDS,
                        help=f'Draw the generation seeds once and reuse them for '
                             f'the whole run (default: {PIN_SEEDS}). The LEMUR pool '
                             f'grows as evaluated candidates are written back, so '
                             f'redrawing every epoch makes the epochs incomparable. '
                             f'Use --no-pin_seeds for the previous behaviour.')
    parser.add_argument('--sft_max_length', type=int, default=SFT_MAX_LENGTH,
                        help=f'Packing/truncation budget for supervised fine-tuning '
                             f'(default: {SFT_MAX_LENGTH}). Keep <= --context_length.')
    parser.add_argument('--context_length', type=int, default=None,
                        help='Model context length override (falls back to default_context_length in config).')
    parser.add_argument('--max_input_length', type=int, default=None,
                        help='Unsloth max input length override.')
    parser.add_argument('--mobile_deployment', action='store_true', default=MOBILE_DEPLOYMENT,
                        help=f'Enable standalone mobile deployment pipeline extensions (default: {MOBILE_DEPLOYMENT}).')
    parser.add_argument('--mobile_reval_only', action='store_true', default=MOBILE_REVAL_ONLY,
                        help='[Mobile Pipeline] Re-run GPU eval + mobile for one cycle only (requires --mobile_deployment).')
    parser.add_argument('--mobile_min_quantized_accuracy', type=float, default=MOBILE_MIN_QUANTIZED_ACCURACY,
                        help=f'[Mobile Pipeline] Minimum quantized accuracy required for selection (default: {MOBILE_MIN_QUANTIZED_ACCURACY}).')
    parser.add_argument('--mobile_max_duration_ms', type=float, default=MOBILE_MAX_DURATION_MS,
                        help=f'[Mobile Pipeline] Maximum on-device duration (ms) allowed for selection (default: {MOBILE_MAX_DURATION_MS}).')
    parser.add_argument('--mobile_score_tolerance', type=float, default=MOBILE_SCORE_TOLERANCE,
                        help=f'[Mobile Pipeline] Non-regression tolerance multiplier for cycle gate (default: {MOBILE_SCORE_TOLERANCE}).')
    parser.add_argument('--mobile_min_valid_models', type=int, default=MOBILE_MIN_VALID_MODELS,
                        help=f'[Mobile Pipeline] Minimum valid mobile-scored models required to accept cycle (default: {MOBILE_MIN_VALID_MODELS}).')
    parser.add_argument('--mobile_delegate_priority', type=str, default=MOBILE_DELEGATE_PRIORITY,
                        help=f'[Mobile Pipeline] Tie-break delegate priority for equal scores (default: {MOBILE_DELEGATE_PRIORITY}).')

    # Corpus filters for the iterative pipeline's LEMUR curation
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help=f"[Pipeline] LEMUR dataset to curate the training corpus from (default: {DEFAULT_DATASET}).")
    parser.add_argument('--nn_prefixes', type=lambda s: tuple(p for p in s.split(',') if p),
                        default=DEFAULT_NN_PREFIXES,
                        help=f"[Pipeline] Comma-separated NN name prefixes to curate (default: {','.join(DEFAULT_NN_PREFIXES)}).")
    parser.add_argument('--llmatic', action='store_true', default=LLMATIC,
                        help="[Pipeline] Enable LLMatic MAP-Elites archive-driven seed "
                             "selection (mutation + periodic crossover) in nn_gen, instead "
                             "of random corpus sampling.")
    parser.add_argument('--llmatic_crossover_every', type=int, default=LLMATIC_CROSSOVER_EVERY,
                        help="[Pipeline] With --llmatic, run a crossover generation every N "
                             f"cycles (default: {LLMATIC_CROSSOVER_EVERY}; 0 disables crossover).")

    args = parser.parse_args()

    # Convert start_layer/end_layer → tune_layers (main() expects a range, not two ints)
    kwargs = vars(args)
    _sl, _el = kwargs.pop('start_layer'), kwargs.pop('end_layer')
    kwargs['tune_layers'] = range(_sl or 0, _el) if _el is not None else None

    # --num_cycles in argparse maps to cycles= in main()
    num_cycles = kwargs.pop('num_cycles')
    if num_cycles is not None:
        kwargs['cycles'] = num_cycles

    # --nn_gen_conf_id accepts a comma-separated list of prompt keys (e.g. the
    # LLR Selection + Mechanism buckets). tune()'s conf_keys wraps a bare string
    # into a 1-tuple, so a single key still works unchanged.
    if isinstance(kwargs.get('nn_gen_conf_id'), str) and ',' in kwargs['nn_gen_conf_id']:
        kwargs['nn_gen_conf_id'] = tuple(
            k.strip() for k in kwargs['nn_gen_conf_id'].split(',') if k.strip())

    main(**kwargs)
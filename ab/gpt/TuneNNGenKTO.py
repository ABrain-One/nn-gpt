"""
ab/gpt/TuneNNGenKTO.py — KTO fine-tuning entry point.

Analog of ab/gpt/TuneNNGen.py for KTO.  Spawned as a subprocess by
KTOIterativeFinetuner.run_finetuning() once per cycle, after the
KTODataManager has written the cycle's kto_train.jsonl.

Two ways to invoke:

  1. Per-cycle fine-tune (called by KTOIterativeFinetuner):
       python -m ab.gpt.TuneNNGenKTO \
           --llm_conf nngpt_unique_arch_rag.json \
           --kto_data_file out/kto_pipeline/cycle_3/chat_data_cycle_3/kto_train.jsonl \
           --peft out/kto_pipeline/cycle_2/checkpoint \
           --kto_beta 0.1
           ...

  2. Full iterative pipeline:
       python -m ab.gpt.TuneNNGenKTO --run_iterative_pipeline \
           --llm_conf nngpt_unique_arch_rag.json --cycles 21
           ...
       (Delegates to KTOIterativeFinetuner; same CLI as the iterative
        module itself.)

Output adapter is saved to out/qlora-kto/final/, which the iterative
pipeline then isolates to out/kto_pipeline/cycle_N/checkpoint/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import TrainingArguments

from ab.gpt.util.Const import conf_llm_dir, nngpt_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.nn.util.Const import out_dir


# ── Defaults (mirrors TuneNNGen.py style) ───────────────────────────────────
LLM_CONF = "nngpt_unique_arch_rag.json"
NUM_TRAIN_EPOCHS = 5
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
PER_DEVICE_TRAIN_BATCH_SIZE = 2   # KTO requires actual batch size > 1 for valid KL estimation
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
OPTIMIZER = "paged_adamw_8bit"

# LoRA hyperparameters
R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")
TASK_TYPE = "CAUSAL_LM"
BIAS = "none"
START_LAYER = 0
END_LAYER = 24

# KTO hyperparameters
KTO_BETA = 0.1
KTO_DESIRABLE_WEIGHT = 1.0
KTO_UNDESIRABLE_WEIGHT = 1.0
MAX_PROMPT_LENGTH = 4096
MAX_COMPLETION_LENGTH = 2048

# Where the final KTO adapter lands (KTOIterativeFinetuner copies it
# out to out/kto_pipeline/cycle_N/checkpoint/)
KTO_OUTPUT_DIR = out_dir / "qlora-kto" / "final"


# ── Data loading ────────────────────────────────────────────────────────────

def _load_kto_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a kto_train.jsonl produced by KTODataManager."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[KTO][WARN] line {line_num} of {path}: {e}")
    print(f"[KTO] Loaded {len(records)} KTO records from {path}")
    return records


def _build_kto_dataset(records: List[Dict[str, Any]], tokenizer) -> Dataset:
    """
    Convert raw KTO records → HF Dataset with the three columns KTOTrainer
    expects: prompt (str), completion (str), label (bool).

    Records carry "prompt_messages" so the chat template is applied here
    (at training time, using the live tokenizer) — guaranteeing the
    prompt the trainer sees matches what the generator was conditioned on.
    """
    prompts: List[str] = []
    completions: List[str] = []
    labels: List[bool] = []
    sim_penalties: List[float] = []

    skipped = 0
    for rec in records:
        prompt_messages = rec.get("prompt_messages") or []
        completion = rec.get("completion", "")
        label = rec.get("label", None)
        if not prompt_messages or not completion or label is None:
            skipped += 1
            continue

        try:
            prompt_str = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[KTO][WARN] chat-template render failed, falling back: {e}")
            # Fallback: simple role-prefixed concatenation
            prompt_str = "\n\n".join(
                f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                for m in prompt_messages
            ) + "\n\nASSISTANT:"

        prompts.append(prompt_str)
        completions.append(completion)
        labels.append(bool(label))
        sim_penalties.append(float(rec.get("sim_penalty", 0.0) or 0.0))

    if skipped > 0:
        print(f"[KTO][WARN] Skipped {skipped} malformed records (missing prompt/completion/label)")

    n_des = sum(labels)
    n_und = len(labels) - n_des
    print(f"[KTO] Built dataset: {len(labels)} examples ({n_des} desirable, {n_und} undesirable)")

    if len(labels) == 0:
        raise RuntimeError("KTO dataset is empty after filtering — refusing to train")

    return Dataset.from_dict({
        "prompt": prompts,
        "completion": completions,
        "label": labels,
        "sim_penalty": sim_penalties,
    })


# ── Model loading (mirrors Tune.py's LLM init flow) ─────────────────────────

def _read_llm_conf(llm_conf: str) -> Dict[str, Any]:
    """Read llm_conf JSON; resolve to conf/llm/ if relative."""
    path = Path(llm_conf)
    if not path.exists():
        candidate = conf_llm_dir / path.name
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"LLM config not found: {llm_conf}")
    with open(path) as f:
        return json.load(f)


def _load_base_model_and_tokenizer(
    llm_conf_data: Dict[str, Any],
    peft_path: Optional[str],
    training_args: TrainingArguments,
):
    """
    Load the base LLM (4-bit) and tokenizer, optionally merging a prior
    LoRA adapter to provide a warm start.  Mirrors the pattern in
    ab/gpt/util/Tune.py:tune().
    """
    from ab.gpt.util.LLM import LLM

    base_model_name = llm_conf_data["base_model_name"]
    context_length = llm_conf_data.get("context_length")

    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        training_args=training_args,
        context_length=context_length,
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    if peft_path:
        print(f"[KTO] Loading previous LoRA adapter from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path, is_trainable=True)
        model = model.merge_and_unload()
        print("[KTO] Previous adapter merged into base model — fresh LoRA will be attached on top")

    # ── Tokenizer hardening for DeepSeek v1.5 / mixed-template models ────────
    # Some instruct-tunes (notably deepseek-coder-7b-instruct-v1.5) ship a
    # tokenizer whose pad/eos differs from the model's `config.{pad,eos}_token_id`,
    # producing the warning:
    #   "tokenizer has new PAD/BOS/EOS tokens that differ from the model config"
    # If left untouched, generation never emits an EOS the code-extractor
    # recognises and every generation rolls until max_new_tokens, yielding
    # unparseable trailing junk → ok=False on every sample.
    #
    # Force the model's generation config to use the tokenizer's tokens so
    # both training and downstream inference agree.
    try:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model_cfg = getattr(model, "config", None)
        gen_cfg = getattr(model, "generation_config", None)
        if model_cfg is not None and tokenizer.pad_token_id is not None:
            model_cfg.pad_token_id = tokenizer.pad_token_id
        if model_cfg is not None and tokenizer.eos_token_id is not None:
            model_cfg.eos_token_id = tokenizer.eos_token_id
        if gen_cfg is not None and tokenizer.pad_token_id is not None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
        if gen_cfg is not None and tokenizer.eos_token_id is not None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        print(f"[KTO] Aligned model pad/eos to tokenizer: "
              f"pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
    except Exception as e:
        print(f"[KTO][WARN] Failed to align tokenizer/model special tokens: {e}")

    return model, tokenizer


# ── Standalone KTO entry point ──────────────────────────────────────────────

def run_kto(
    llm_conf: str = LLM_CONF,
    kto_data_file: Optional[str] = None,
    peft_path: Optional[str] = None,
    # KTO hyperparameters
    kto_beta: float = KTO_BETA,
    kto_desirable_weight: float = KTO_DESIRABLE_WEIGHT,
    kto_undesirable_weight: float = KTO_UNDESIRABLE_WEIGHT,
    sim_alpha: float = 0.0,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
    max_completion_length: int = MAX_COMPLETION_LENGTH,
    # Standard training hyperparameters
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_steps: Optional[int] = WARMUP_STEPS,
    warmup_ratio: float = WARMUP_RATIO,
    max_grad_norm: float = MAX_GRAD_NORM,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: Optional[int] = 1,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    lr_scheduler_type: str = LR_SCHEDULER,
    logging_steps: int = LOGGING_STEPS,
    optimizer: str = OPTIMIZER,
    # LoRA configuration
    r: int = R,
    lora_alpha: float = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type: str = TASK_TYPE,
    bias: str = BIAS,
    tune_layers=None,
    # Eval / save
    evaluation_strategy: Optional[str] = "steps",
    eval_steps: Optional[int] = 100,
    save_strategy: Optional[str] = "steps",
    save_steps: Optional[int] = 100,
    save_total_limit: Optional[int] = 3,
    load_best_model_at_end: bool = False,
    metric_for_best_model: Optional[str] = "eval_loss",
    output_dir: Optional[Path] = None,
):
    """
    One-shot KTO fine-tune.  Loads the model + KTO dataset, attaches a
    fresh LoRA adapter, runs KTOTrainer, and saves to output_dir.
    """
    if kto_data_file is None:
        raise ValueError("--kto_data_file is required for KTO training")
    kto_path = Path(kto_data_file)
    if not kto_path.exists():
        raise FileNotFoundError(f"KTO data file not found: {kto_path}")

    output_dir = Path(output_dir) if output_dir else KTO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. TrainingArguments (KTOConfig is a subclass — KTO class casts it) ─
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_kwargs = dict(
        output_dir=str(nngpt_dir / "outputs_kto"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        optim=optimizer,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    if warmup_steps is not None:
        training_kwargs["warmup_steps"] = warmup_steps
    else:
        training_kwargs["warmup_ratio"] = warmup_ratio
    if per_device_eval_batch_size is not None:
        training_kwargs["per_device_eval_batch_size"] = per_device_eval_batch_size
    if evaluation_strategy is not None:
        training_kwargs["eval_strategy"] = evaluation_strategy
        if eval_steps is not None:
            training_kwargs["eval_steps"] = eval_steps
    if save_strategy is not None:
        training_kwargs["save_strategy"] = save_strategy
        if save_steps is not None:
            training_kwargs["save_steps"] = save_steps
        if save_total_limit is not None:
            training_kwargs["save_total_limit"] = save_total_limit
    if load_best_model_at_end:
        training_kwargs["load_best_model_at_end"] = True
        if metric_for_best_model is not None:
            training_kwargs["metric_for_best_model"] = metric_for_best_model

    training_args = TrainingArguments(**training_kwargs)

    # ── 2. LoRA config ──────────────────────────────────────────────────────
    if tune_layers is None:
        tune_layers = range(START_LAYER, END_LAYER)
    from ab.gpt.util.KTO import kto_lora_config
    peft_config = kto_lora_config(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        layers_to_transform=tune_layers,
    )

    # ── 3. Load model + tokenizer ───────────────────────────────────────────
    llm_conf_data = _read_llm_conf(llm_conf)
    model, tokenizer = _load_base_model_and_tokenizer(
        llm_conf_data, peft_path, training_args
    )

    # ── 4. Build KTO dataset ────────────────────────────────────────────────
    records = _load_kto_jsonl(kto_path)
    dataset = _build_kto_dataset(records, tokenizer)

    # ── 5. KTO training ─────────────────────────────────────────────────────
    # Late import so the module can also be used in pipeline-dispatch mode
    # without paying the KTO trainer import cost up front.
    from ab.gpt.util.KTO import KTO

    kto = KTO(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        peft_config=peft_config,
    )

    kto.train(
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        beta=kto_beta,
        desirable_weight=kto_desirable_weight,
        undesirable_weight=kto_undesirable_weight,
        sim_alpha=sim_alpha,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )

    print(f"[KTO] Done.  Adapter saved to: {output_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    TARGET_MODULES_STR = ",".join(TARGET_MODULES)
    parser = argparse.ArgumentParser(
        description="KTO fine-tune (one-shot) or iterative KTO pipeline dispatch."
    )

    # Pipeline-dispatch route (delegates to KTOIterativeFinetuner)
    parser.add_argument("--run_iterative_pipeline", action="store_true", default=False,
                        help="Run the full iterative KTO pipeline (delegates to KTOIterativeFinetuner)")
    parser.add_argument("--cycles", type=int, default=21)
    parser.add_argument("--models_per_cycle", type=int, default=150)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--accuracy_threshold", type=float, default=0.40)
    parser.add_argument("--min_selected_k", type=int, default=15)
    parser.add_argument("--fallback_threshold", type=float, default=0.35)
    parser.add_argument("--adaptive_threshold", action="store_true", default=False)
    parser.add_argument("--novelty_check", action="store_true", default=True)
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false")
    parser.add_argument("--resume_from_cycle", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--use_optimized_training", action="store_true", default=True)
    parser.add_argument("--no_optimized_training", dest="use_optimized_training", action="store_false")
    parser.add_argument("--undesirable_floor_accuracy", type=float, default=0.10)
    parser.add_argument("--undesirable_ratio", type=float, default=1.0)
    parser.add_argument("--max_undesirable_per_cycle", type=int, default=500)
    parser.add_argument("--kto_output_subdir", type=str, default="kto_pipeline")

    # KTO standalone / shared
    parser.add_argument("--llm_conf", type=str, default=LLM_CONF)
    parser.add_argument("--kto_data_file", type=str, default=None,
                        help="Path to kto_train.jsonl (standalone mode)")
    parser.add_argument("--kto_checkpoint_dir", type=str, default=None,
                        help="Where to save the final KTO adapter "
                             "(default: out/qlora-kto/<kto_output_subdir>/final). "
                             "Set per-run to avoid collisions when two runs train concurrently.")
    parser.add_argument("--peft", dest="peft_path", type=str, default=None,
                        help="Previous LoRA adapter to warm-start from (merged into base, then fresh LoRA on top)")

    # KTO hyperparameters
    parser.add_argument("--kto_beta", type=float, default=KTO_BETA)
    parser.add_argument("--kto_desirable_weight", type=float, default=KTO_DESIRABLE_WEIGHT)
    parser.add_argument("--kto_undesirable_weight", type=float, default=KTO_UNDESIRABLE_WEIGHT)
    parser.add_argument("--sim_alpha", type=float, default=0.0,
                        help="If >0, subtract alpha*sim_penalty from the KTO chosen reward")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--max_completion_length", type=int, default=MAX_COMPLETION_LENGTH)

    # Standard training knobs (mirror TuneNNGen.py)
    parser.add_argument("-ne", "--num_train_epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("-l", "--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("-g", "--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("-ls", "--lr_scheduler", dest="lr_scheduler_type", type=str, default=LR_SCHEDULER)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER)

    # LoRA
    parser.add_argument("-r", "--r", type=int, default=R)
    parser.add_argument("-a", "--lora_alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("-d", "--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("-t", "--target_modules", type=lambda s: s.split(","), default=TARGET_MODULES,
                        help=f"Comma-separated target modules (default: {TARGET_MODULES_STR})")
    parser.add_argument("-y", "--task_type", type=str, default=TASK_TYPE)
    parser.add_argument("-b", "--bias", type=str, default=BIAS)
    parser.add_argument("-s", "--start_layer", type=int, default=START_LAYER)
    parser.add_argument("-e", "--end_layer", type=int, default=END_LAYER)

    # Eval / save (passed through from iterative pipeline)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")

    # Inherited from TuneNNGen optimized-training preset — accepted but unused
    # by KTO standalone training (KTO generator parameters live in the iterative
    # pipeline, not here).  Defining them keeps the iterative-pipeline subprocess
    # call signature compatible.
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_train_epochs_pipeline", type=int, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── Route: iterative pipeline ───────────────────────────────────────────
    if args.run_iterative_pipeline:
        from ab.gpt.kto_iterative_finetune import KTOIterativeFinetuner

        if args.resume_from_cycle is not None:
            if args.resume_from_cycle < 1 or args.resume_from_cycle > args.cycles:
                print(f"[ERROR] Invalid resume_from_cycle {args.resume_from_cycle}; "
                      f"must be 1..{args.cycles}")
                sys.exit(1)

        pipeline = KTOIterativeFinetuner(
            llm_conf=args.llm_conf,
            cycles=args.cycles,
            models_per_cycle=args.models_per_cycle,
            samples_per_prompt=args.samples_per_prompt,
            accuracy_threshold=args.accuracy_threshold,
            min_selected_k=args.min_selected_k,
            fallback_threshold=args.fallback_threshold,
            adaptive_threshold=args.adaptive_threshold,
            novelty_check=args.novelty_check,
            resume_from_cycle=args.resume_from_cycle,
            max_retries=args.max_retries,
            use_optimized_training=args.use_optimized_training,
            num_train_epochs=args.num_train_epochs,
            undesirable_floor_accuracy=args.undesirable_floor_accuracy,
            undesirable_ratio=args.undesirable_ratio,
            max_undesirable_per_cycle=args.max_undesirable_per_cycle,
            kto_beta=args.kto_beta,
            kto_desirable_weight=args.kto_desirable_weight,
            kto_undesirable_weight=args.kto_undesirable_weight,
            kto_output_subdir=args.kto_output_subdir,
        )
        pipeline.run()
        return

    # ── Route: standalone KTO fine-tune ─────────────────────────────────────
    if args.kto_data_file is None:
        print("[ERROR] Either --run_iterative_pipeline or --kto_data_file must be provided")
        sys.exit(1)

    print(f"[KTO] Standalone KTO fine-tune")
    print(f"  llm_conf:           {args.llm_conf}")
    print(f"  kto_data_file:      {args.kto_data_file}")
    print(f"  peft (warm-start):  {args.peft_path}")
    print(f"  beta:               {args.kto_beta}")
    print(f"  desirable_weight:   {args.kto_desirable_weight}")
    print(f"  undesirable_weight: {args.kto_undesirable_weight}")
    print(f"  num_train_epochs:   {args.num_train_epochs}")
    effective_output_dir = Path(args.kto_checkpoint_dir) if args.kto_checkpoint_dir else KTO_OUTPUT_DIR
    print(f"  output_dir:         {effective_output_dir}")

    tune_layers = range(args.start_layer, args.end_layer)

    run_kto(
        llm_conf=args.llm_conf,
        kto_data_file=args.kto_data_file,
        peft_path=args.peft_path,
        kto_beta=args.kto_beta,
        kto_desirable_weight=args.kto_desirable_weight,
        kto_undesirable_weight=args.kto_undesirable_weight,
        sim_alpha=args.sim_alpha,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        optimizer=args.optimizer,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=args.task_type,
        bias=args.bias,
        tune_layers=tune_layers,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        output_dir=effective_output_dir,
    )


if __name__ == "__main__":
    main()

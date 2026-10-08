"""
ab/gpt/TuneNNGenDPO.py — DPO fine-tuning entry point (DivPO training subprocess).

Analog of ab/gpt/TuneNNGenKTO.py for DPO's paired preference format. Spawned as a
subprocess by the DivPO pipeline once per cycle, after it has written the cycle's
divpo_train.jsonl (records: {prompt_messages, chosen, rejected}).

Model loading is REUSED from TuneNNGenKTO (identical 4-bit load + tokenizer
hardening) so DPO runs are directly comparable to the KTO runs. DivPO's diversity
lives entirely in the pair *selection* (done in the pipeline); training here is
plain DPO.

    python -m ab.gpt.act.tune.DPO \\
        --llm_conf nngpt_unique_arch_rag.json \\
        --dpo_data_file out/.../cycle_3/divpo_train.jsonl \\
        --dpo_checkpoint_dir out/.../cycle_3/checkpoint \\
        --peft out/.../cycle_2/checkpoint --dpo_beta 0.1 ...
"""

from __future__ import annotations

import os
# DPO runs forward passes for BOTH chosen and rejected (plus the reference model),
# so peak GPU memory is ~2-4x KTO's. Reduce the allocator's fragmentation (the OOM
# error explicitly recommends this) BEFORE torch initialises the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments

from ab.gpt.util.Const import nngpt_dir
from ab.nn.util.Const import out_dir
# Reuse KTO's model-loading + conf helpers verbatim so the base model, tokenizer
# hardening, and warm-start behaviour are IDENTICAL to the KTO runs.
from ab.gpt.act.tune.KTO import _read_llm_conf, _load_base_model_and_tokenizer


# ── Defaults (mirror TuneNNGenKTO) ──────────────────────────────────────────
LLM_CONF = "nngpt_unique_arch_rag.json"
NUM_TRAIN_EPOCHS = 5
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
# DPO's paired forward (chosen+rejected+reference) makes batch 2 OOM on 48GB with
# 4096-token architecture code; batch 1 halves peak memory. grad-accum doubled so
# the effective batch (8) still matches KTO. DPO needs no batch>1 (per-pair loss).
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
OPTIMIZER = "paged_adamw_8bit"

R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")
TASK_TYPE = "CAUSAL_LM"
BIAS = "none"
START_LAYER = 0
END_LAYER = 24

DPO_BETA = 0.1
MAX_PROMPT_LENGTH = 4096
MAX_COMPLETION_LENGTH = 2048

DPO_OUTPUT_DIR = out_dir / "qlora-dpo" / "final"


# ── Data loading ────────────────────────────────────────────────────────────

def _load_dpo_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[DPO][WARN] line {line_num} of {path}: {e}")
    print(f"[DPO] Loaded {len(records)} DPO pairs from {path}")
    return records


def _build_dpo_dataset(records: List[Dict[str, Any]], tokenizer) -> Dataset:
    """Convert raw DivPO pair records → HF Dataset with {prompt, chosen, rejected}.

    Records carry "prompt_messages" so the chat template is applied here (live
    tokenizer) — matching how the generator was conditioned, exactly like KTO.
    chosen/rejected are already fenced code strings written by the pipeline.
    """
    prompts: List[str] = []
    chosens: List[str] = []
    rejecteds: List[str] = []

    skipped = 0
    for rec in records:
        prompt_messages = rec.get("prompt_messages") or []
        chosen = rec.get("chosen", "")
        rejected = rec.get("rejected", "")
        if not prompt_messages or not chosen or not rejected or chosen == rejected:
            skipped += 1
            continue
        try:
            prompt_str = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[DPO][WARN] chat-template render failed, falling back: {e}")
            prompt_str = "\n\n".join(
                f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                for m in prompt_messages
            ) + "\n\nASSISTANT:"
        prompts.append(prompt_str)
        chosens.append(chosen)
        rejecteds.append(rejected)

    if skipped > 0:
        print(f"[DPO][WARN] Skipped {skipped} malformed/degenerate pairs "
              "(missing prompt/chosen/rejected or chosen==rejected)")
    print(f"[DPO] Built dataset: {len(prompts)} preference pairs")
    if len(prompts) == 0:
        raise RuntimeError("DPO dataset is empty after filtering — refusing to train")

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


# ── Standalone DPO entry point ──────────────────────────────────────────────

def run_dpo(
    llm_conf: str = LLM_CONF,
    dpo_data_file: Optional[str] = None,
    peft_path: Optional[str] = None,
    dpo_beta: float = DPO_BETA,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
    max_completion_length: int = MAX_COMPLETION_LENGTH,
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
    r: int = R,
    lora_alpha: float = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type: str = TASK_TYPE,
    bias: str = BIAS,
    tune_layers=None,
    evaluation_strategy: Optional[str] = "steps",
    eval_steps: Optional[int] = 100,
    save_strategy: Optional[str] = "steps",
    save_steps: Optional[int] = 100,
    save_total_limit: Optional[int] = 3,
    output_dir: Optional[Path] = None,
):
    """One-shot DPO fine-tune: load model + DivPO pairs, attach fresh LoRA, train."""
    if dpo_data_file is None:
        raise ValueError("--dpo_data_file is required for DPO training")
    dpo_path = Path(dpo_data_file)
    if not dpo_path.exists():
        raise FileNotFoundError(f"DPO data file not found: {dpo_path}")

    output_dir = Path(output_dir) if output_dir else DPO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. TrainingArguments (DPOConfig is a subclass — DPO class casts it) ─
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_kwargs = dict(
        output_dir=str(nngpt_dir / "outputs_dpo"),
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

    training_args = TrainingArguments(**training_kwargs)

    # ── 2. LoRA config (shared drift-control builder with KTO) ──────────────
    if tune_layers is None:
        tune_layers = range(START_LAYER, END_LAYER)
    from ab.gpt.util.llm.KTO import kto_lora_config
    peft_config = kto_lora_config(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        layers_to_transform=tune_layers,
    )

    # ── 3. Load model + tokenizer (reused from KTO — identical behaviour) ───
    llm_conf_data = _read_llm_conf(llm_conf)
    model, tokenizer = _load_base_model_and_tokenizer(llm_conf_data, peft_path, training_args)

    # ── 4. Build DPO dataset ────────────────────────────────────────────────
    records = _load_dpo_jsonl(dpo_path)
    dataset = _build_dpo_dataset(records, tokenizer)

    # ── 5. DPO training ─────────────────────────────────────────────────────
    from ab.gpt.util.DPO import DPO

    dpo = DPO(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        peft_config=peft_config,
    )
    dpo.train(
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        beta=dpo_beta,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    print(f"[DPO] Done.  Adapter saved to: {output_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tune (one-shot) for the DivPO pipeline.")
    parser.add_argument("--llm_conf", type=str, default=LLM_CONF)
    parser.add_argument("--dpo_data_file", type=str, default=None,
                        help="Path to divpo_train.jsonl ({prompt_messages, chosen, rejected})")
    parser.add_argument("--dpo_checkpoint_dir", type=str, default=None,
                        help="Where to save the final DPO adapter")
    parser.add_argument("--peft", dest="peft_path", type=str, default=None,
                        help="Previous LoRA adapter to warm-start from (merged, fresh LoRA on top)")

    parser.add_argument("--dpo_beta", type=float, default=DPO_BETA,
                        help="DPO KL strength; higher keeps the policy nearer the diverse base")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--max_completion_length", type=int, default=MAX_COMPLETION_LENGTH)

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

    parser.add_argument("-r", "--r", type=int, default=R)
    parser.add_argument("-a", "--lora_alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("-d", "--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("-t", "--target_modules", type=lambda s: s.split(","), default=TARGET_MODULES)
    parser.add_argument("-y", "--task_type", type=str, default=TASK_TYPE)
    parser.add_argument("-b", "--bias", type=str, default=BIAS)
    parser.add_argument("-s", "--start_layer", type=int, default=START_LAYER)
    parser.add_argument("-e", "--end_layer", type=int, default=END_LAYER)

    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)

    args = parser.parse_args()

    if args.dpo_data_file is None:
        print("[ERROR] --dpo_data_file is required")
        sys.exit(1)

    print("[DPO] Standalone DPO fine-tune")
    print(f"  llm_conf:          {args.llm_conf}")
    print(f"  dpo_data_file:     {args.dpo_data_file}")
    print(f"  peft (warm-start): {args.peft_path}")
    print(f"  beta:              {args.dpo_beta}")
    print(f"  num_train_epochs:  {args.num_train_epochs}")
    effective_output_dir = Path(args.dpo_checkpoint_dir) if args.dpo_checkpoint_dir else DPO_OUTPUT_DIR
    print(f"  output_dir:        {effective_output_dir}")

    run_dpo(
        llm_conf=args.llm_conf,
        dpo_data_file=args.dpo_data_file,
        peft_path=args.peft_path,
        dpo_beta=args.dpo_beta,
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
        tune_layers=range(args.start_layer, args.end_layer),
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=effective_output_dir,
    )


if __name__ == "__main__":
    main()

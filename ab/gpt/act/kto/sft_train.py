#!/usr/bin/env python3
"""Supervised fine-tuning step for the v2 pipeline's SFT control.

Same model load, LoRA layers/rank, optimizer, schedule, continual adapter and
90/10 split as the KTO/DPO trainers; only the loss differs (next-token
cross-entropy on the completion, prompt tokens masked).

  python -m ab.gpt.act.kto.sft_train --sft_data_file f.jsonl --sft_checkpoint_dir d \
      --peft prev/checkpoint --continual_adapter
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from ab.gpt.act.tune.KTO import (
    END_LAYER, LORA_DROPOUT, LOGGING_STEPS, LR_SCHEDULER, OPTIMIZER, START_LAYER,
    TARGET_MODULES, WARMUP_STEPS, WEIGHT_DECAY, _load_base_model_and_tokenizer, _read_llm_conf,
)
from ab.gpt.util.Const import nngpt_dir


def _tokenize(records, tok, max_prompt: int, max_completion: int):
    eos = tok.eos_token_id
    rows = {"input_ids": [], "labels": []}
    for rec in records:
        msgs, completion = rec.get("prompt_messages"), rec.get("completion")
        if not msgs or not completion:
            continue
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False).input_ids[-max_prompt:]
        c = tok(completion, add_special_tokens=False).input_ids[:max_completion - 1] + [eos]
        rows["input_ids"].append(p + c)
        rows["labels"].append([-100] * len(p) + c)
    if not rows["input_ids"]:
        raise RuntimeError("SFT dataset is empty after filtering — refusing to train")
    return Dataset.from_dict(rows)


class _Collate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            lab.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT step (v2 control)")
    ap.add_argument("--llm_conf", default="nngpt_unique_arch_rag.json")
    ap.add_argument("--sft_data_file", required=True)
    ap.add_argument("--sft_checkpoint_dir", required=True)
    ap.add_argument("--peft", dest="peft_path", default=None)
    ap.add_argument("--continual_adapter", action="store_true")
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=0.3)
    ap.add_argument("--max_prompt_length", type=int, default=1536)
    ap.add_argument("--max_completion_length", type=int, default=2560)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = ap.parse_args()

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=str(nngpt_dir / "outputs_sft"), num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=WEIGHT_DECAY, max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=LR_SCHEDULER, warmup_steps=WARMUP_STEPS, logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER, bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        report_to=[], remove_unused_columns=False, save_strategy="no", eval_strategy="no",
    )
    model, tok = _load_base_model_and_tokenizer(
        _read_llm_conf(args.llm_conf), None if args.continual_adapter else args.peft_path, targs)

    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from ab.gpt.util.llm.KTO import kto_lora_config
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    if args.continual_adapter and args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=True)
        print(f"[SFT] Continuing adapter {args.peft_path}")
    else:
        model = get_peft_model(model, kto_lora_config(
            target_modules=TARGET_MODULES, r=args.r, lora_alpha=args.lora_alpha,
            lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
            layers_to_transform=range(START_LAYER, END_LAYER)))
    model.config.use_cache = False

    records = [json.loads(l) for l in Path(args.sft_data_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = _tokenize(records, tok, args.max_prompt_length, args.max_completion_length)
    train_ds = ds.train_test_split(test_size=0.1, seed=43)["train"]   # same split as KTO/DPO
    print(f"[SFT] {len(records)} records -> {len(train_ds)} training examples")

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=_Collate(pad)).train()

    out = Path(args.sft_checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[SFT] Done. Adapter saved to: {out}")


if __name__ == "__main__":
    main()

"""Stage 3: QLoRA fine-tuning with validation-based early stopping."""
from __future__ import annotations

from pathlib import Path

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import EarlyStoppingCallback, TrainingArguments
from datasets import Dataset

from .config import *  # noqa: F401,F403
from .common import _load_messages
from .model import _load_model




def _format_dataset(examples: dict, tokenizer) -> dict:
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text + tokenizer.eos_token)
    return {"text": texts}


def train_model(
    train_path: Path,
    val_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_epochs: int = NUM_EPOCHS,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
) -> None:
    train_examples = _load_messages(train_path)
    val_examples = _load_messages(val_path)

    train_ids = {ex["architecture_id"] for ex in train_examples}
    val_ids = {ex["architecture_id"] for ex in val_examples}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"Architecture leakage: {len(overlap)} architecture_id(s) appear in both train and validation."
        )

    model, tokenizer = _load_model(model_name, max_seq_len)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    FastLanguageModel.for_training(model)

    train_dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in train_examples])
    val_dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in val_examples])
    train_dataset = train_dataset.map(
        _format_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        _format_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    steps_per_epoch = max(
        1,
        (len(train_dataset) + batch_size * gradient_accumulation_steps - 1)
        // (batch_size * gradient_accumulation_steps),
    )
    total_steps = MAX_STEPS if MAX_STEPS > 0 else steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if MAX_STEPS > 0 and warmup_steps >= MAX_STEPS:
        warmup_steps = max(1, MAX_STEPS // 10)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ta_kwargs = {
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": 20,
        "logging_first_step": True,
        "optim": "adamw_8bit",
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": "linear",
        "seed": SEED,
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "save_total_limit": SAVE_TOTAL_LIMIT,
    }
    if MAX_STEPS > 0:
        ta_kwargs["max_steps"] = MAX_STEPS

    try:
        training_args = TrainingArguments(**ta_kwargs)
    except TypeError:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**ta_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

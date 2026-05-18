#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-Coder-7B-Instruct (via Unsloth's bnb-4bit build) using Unsloth + QLoRA for outcome prediction.

Usage:
    python train_deepseek_coder.py --out model2 --max_seq_len 6144

Dataset: CoderLLmModel/data/train_llm_dataset.jsonl
Max sequence length: 6144 (default)
"""

import json
import random
from pathlib import Path
import sys

try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import torch
    from tqdm import tqdm
except ImportError as e:
    raise RuntimeError(
        "Missing required package. Install: pip install unsloth[colab-new] trl datasets"
    ) from e


# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = SCRIPT_DIR / "data" / "train_llm_dataset.jsonl"
DEFAULT_VAL_PATH = SCRIPT_DIR / "data" / "val_llm_dataset.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "model2"
RAW_INPUT_PATH = SCRIPT_DIR / "data" / "llm_finetuning_data.jsonl"
PREP_OUTPUT_DIR = SCRIPT_DIR / "data"
# Unsloth maps Qwen/Qwen2.5-Coder-7B-Instruct to a lowercase unsloth/* id that does not exist on the Hub.
# Use the official Unsloth 4-bit checkpoint (correct casing: Qwen2.5, 7B).
# Note: transformers 5.x often fails to load this Hub repo (generic "pytorch_model.bin" error); see load helper below.
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
OFFICIAL_INSTRUCT_4BIT_FALLBACK = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_MAX_SEQ_LEN = 6144
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.013
WARMUP_RATIO = 0.02
MAX_GRAD_NORM = 0.5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 5
MAX_STEPS = 0
SEED = 42
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SYSTEM_PROMPT = """You are a strict JSON generator.
You must output exactly ONE JSON object and nothing else.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float in [0,100] rounded to 2 decimals.
best_epoch must be a positive integer representing the absolute epoch where peak validation accuracy occurs.

Do not explain.
Do not add text.
Stop immediately after the closing brace }.
"""


def load_model_and_tokenizer(model_name: str, max_seq_len: int):
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )


def load_jsonl_dataset(file_path):
    """
    Load dataset from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        list: List of dictionaries with "messages" key
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def format_dataset(examples, tokenizer):
    """
    Format dataset using tokenizer's chat template.

    This formats the full conversation including system, user, and assistant messages.
    Uses tokenizer.apply_chat_template to ensure proper formatting.

    Args:
        examples: Dictionary with "messages" key (batched)
        tokenizer: Tokenizer to apply chat template

    Returns:
        dict: Formatted dataset with "text" key
    """
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text + tokenizer.eos_token)

    return {"text": texts}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def compute_targets(record: dict) -> dict:
    best_accuracy = max(0.0, min(_safe_float(record.get("best_accuracy"), 0.0), 100.0))
    best_epoch = max(1, _safe_int(record.get("epochs_to_best"), 0))
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    best_epoch = max(1, min(best_epoch, max_epochs))
    return {"best_accuracy": best_accuracy, "best_epoch": best_epoch}


def build_user_message(record: dict, nn_code: str) -> str:
    task = record.get("task", "")
    dataset = record.get("dataset", "")
    metric = record.get("metric", "")
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    epoch_1_accuracy = _safe_float(record.get("accuracy_epoch_1"), 0.0)
    epoch_2_accuracy = _safe_float(record.get("accuracy_epoch_2"), 0.0)
    epoch_3_accuracy = _safe_float(record.get("accuracy_epoch_3"), 0.0)

    lines = [
        "INPUT",
        f"task: {task}",
        f"dataset: {dataset}",
        f"metric: {metric}",
        "",
        "TRAINING_BUDGET",
        f"max_epochs: {max_epochs}",
        "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {epoch_1_accuracy}",
        f"epoch_2_accuracy: {epoch_2_accuracy}",
        f"epoch_3_accuracy: {epoch_3_accuracy}",
        "NEURAL_NETWORK_CODE",
        "```python",
        (nn_code or "").strip(),
        "```",
        "",
        "Analyze the training dynamics, neural network code and dataset to estimate the final training outcome.",
        "",
        "When analyzing the neural network code, pay attention to:",
        "- depth",
        "- normalization layers",
        "- residual connections",
        "- dropout",
        "- activation functions",
        "- parameter scale",
        "- output layer suitability",
        "",
        "Using these signals, estimate the final best validation accuracy and the epoch where it occurs.",
        "",
        "The value best_epoch represents the training epoch where the highest validation accuracy is reached.",
        "",
        "Constraints:",
        "- best_epoch must be an integer between 1 and max_epochs.",
        "",
        "OUTPUT (JSON ONLY):",
    ]
    return "\n".join(str(item) for item in lines)


def record_to_chatml(record: dict) -> dict | None:
    required = {"task", "dataset", "metric", "nn", "nn_code", "best_accuracy", "epochs_to_best"}
    if not all(k in record for k in required):
        return None

    nn_code = record.get("nn_code", "") or ""
    if not isinstance(nn_code, str):
        return None

    targets = compute_targets(record)
    user_content = build_user_message(record, nn_code)
    assistant_content = json.dumps(
        {"best_accuracy": round(targets["best_accuracy"], 2), "best_epoch": int(targets["best_epoch"])},
        separators=(",", ":"),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def stream_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def split_by_task_dataset_architecture(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    key_to_records: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        task = rec.get("task")
        dataset = rec.get("dataset")
        nn = rec.get("nn")
        if task is None or task == "" or dataset is None or dataset == "" or nn is None or nn == "":
            continue
        key = (str(task), str(dataset), str(nn))
        key_to_records.setdefault(key, []).append(rec)

    unique_keys = list(key_to_records.keys())
    random.seed(SPLIT_SEED)
    random.shuffle(unique_keys)

    n_total = len(unique_keys)
    if n_total == 0:
        return [], [], []

    if n_total == 1:
        train_keys = {unique_keys[0]}
        val_keys = set()
        test_keys = set()
    elif n_total == 2:
        train_keys = {unique_keys[0]}
        val_keys = {unique_keys[1]}
        test_keys = set()
    else:
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val
        if n_val < 1:
            n_val += 1
            n_train -= 1
        if n_test < 1:
            n_test += 1
            n_train -= 1
        if n_train < 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
                n_train += 1
            elif n_test > 1:
                n_test -= 1
                n_train += 1
            else:
                n_train = 1
                n_val = max(0, n_val - 1)
                n_test = n_total - n_train - n_val
        train_keys = set(unique_keys[:n_train])
        val_keys = set(unique_keys[n_train : n_train + n_val])
        test_keys = set(unique_keys[n_train + n_val :])

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    for key, recs in key_to_records.items():
        if key in train_keys:
            train_records.extend(recs)
        elif key in val_keys:
            val_records.extend(recs)
        elif key in test_keys:
            test_records.extend(recs)
    return train_records, val_records, test_records


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def prepare_llm_datasets(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        sys.exit(1)

    records: list[dict] = []
    for raw in tqdm(stream_jsonl(input_path), desc="Processing"):
        chatml = record_to_chatml(raw)
        if chatml is None:
            continue
        task = raw.get("task")
        dataset = raw.get("dataset")
        nn = raw.get("nn")
        if task is None or task == "" or dataset is None or dataset == "" or nn is None or nn == "":
            continue
        records.append({"task": str(task), "dataset": str(dataset), "nn": str(nn), "chatml": chatml})

    if not records:
        sys.exit(1)

    train_data, val_data, test_data = split_by_task_dataset_architecture(records)
    train_path = output_dir / "train_llm_dataset.jsonl"
    val_path = output_dir / "val_llm_dataset.jsonl"
    test_path = output_dir / "test_llm_dataset.jsonl"
    _write_jsonl(train_path, [r["chatml"] for r in train_data])
    _write_jsonl(val_path, [r["chatml"] for r in val_data])
    _write_jsonl(test_path, [r["chatml"] for r in test_data])
    return train_path, val_path, test_path


def train_model(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> None:
    loaded_model_id = model_name
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, max_seq_len)
    except Exception:
        if model_name == MODEL_NAME:
            loaded_model_id = OFFICIAL_INSTRUCT_4BIT_FALLBACK
            try:
                model, tokenizer = load_model_and_tokenizer(
                    OFFICIAL_INSTRUCT_4BIT_FALLBACK, max_seq_len
                )
            except Exception:
                sys.exit(1)
        else:
            sys.exit(1)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    FastLanguageModel.for_training(model)

    train_data = load_jsonl_dataset(train_path)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        format_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=train_dataset.column_names
    )

    val_data = load_jsonl_dataset(val_path) if val_path.exists() else []
    has_val = len(val_data) > 0
    if has_val:
        val_dataset = Dataset.from_list(val_data)
        val_dataset = val_dataset.map(
            format_dataset,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            remove_columns=val_dataset.column_names
        )
    else:
        val_dataset = None

    steps_per_epoch = max(
        1,
        (len(train_dataset) + (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS - 1))
        // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    if MAX_STEPS > 0:
        total_steps = MAX_STEPS
    else:
        total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    if MAX_STEPS > 0 and warmup_steps >= MAX_STEPS:
        warmup_steps = max(1, MAX_STEPS // 10)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    ta_kwargs = dict(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=warmup_steps,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=20,
        logging_first_step=True,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=str(output_dir),
        report_to="none",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=has_val,
        max_grad_norm=MAX_GRAD_NORM,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )
    if MAX_STEPS > 0:
        ta_kwargs["max_steps"] = MAX_STEPS
    if has_val:
        ta_kwargs.update(
            dict(
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
        )
    else:
        ta_kwargs.update(dict(eval_strategy="no"))
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
    )
    trainer.train()

    if has_val:
        trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main():
    train_path, val_path, _ = prepare_llm_datasets(RAW_INPUT_PATH, PREP_OUTPUT_DIR)
    train_model(
        train_path=train_path,
        val_path=val_path,
        output_dir=DEFAULT_OUTPUT_DIR,
        model_name=MODEL_NAME,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
    )


if __name__ == "__main__":
    main()

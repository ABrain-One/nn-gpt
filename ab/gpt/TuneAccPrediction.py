#!/usr/bin/env python3
# --- OOM-Hardened Qwen3-8B QLoRA training/eval (minimal good fixes) ---

# =======================
# Std libs
# =======================
import os, re, math, json
import numpy as np
import pandas as pd
import torch

# =======================
# 3rd-party
# =======================
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel

# =======================
# Project utils
# =======================
from ab.gpt.util.Const import out_dir, conf_train_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit, tokenize
from ab.gpt.util.LoRA import create_peft_config, print_trainable_parameters

# =======================
# CUDA allocator settings
# =======================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =======================
# Tunables (REDUCED for OOM fix)
# =======================
MODEL_NAME   = "Qwen/Qwen3-8B"
MAX_LENGTH   = int(os.getenv("MAX_LENGTH", 1024))  # REDUCED from 1536
EPOCHS       = int(os.getenv("EPOCHS", 3))
TRAIN_FRAC   = float(os.getenv("TRAIN_FRAC", 0.8))
SEED         = int(os.getenv("SEED", 42))

BASE_DIR  = out_dir / "qwen_min"
MODEL_DIR = BASE_DIR / "model"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# Template safety helpers
# =======================
_ALLOWED_FIELDS = {
    "model_code", "hyperparameters",
    "epoch_1_accuracy", "epoch_2_accuracy", "epoch_3_accuracy",
    "best_epoch_1_accuracy", "best_epoch_2_accuracy",
    "dataset", "task_type", "transform_code", "metric_code",
    "max_epoch"
}
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def _escape_unfilled_placeholders(template: str) -> str:
    """
    Replace any {field} not in _ALLOWED_FIELDS with {{field}} so .format() won't touch it.
    Prevents KeyError for placeholders like {best_accuracy}/{best_epoch} living in the template.
    """
    def repl(m):
        key = m.group(1)
        return m.group(0) if key in _ALLOWED_FIELDS else "{{" + key + "}}"
    return _PLACEHOLDER_RE.sub(repl, template)

# =======================
# Prompt template
# =======================
def load_prompt_template():
    with open(conf_train_dir / "NN_pre.json") as f:
        tpl = json.load(f)["predict_metrics"]["prompt"]
    s = "\n".join([l for l in tpl if "{acc_at_max_epoch}" not in l])
    # Escape unknown placeholders (e.g., {best_accuracy}, {best_epoch}) → no KeyError in .format()
    return _escape_unfilled_placeholders(s)

# =======================
# Data prep (from full history; one row per (task,dataset,nn))
# =======================
def prepare_data():
    from ab.nn.api import data as nn_data
    output_csv = BASE_DIR / 'nn_results_enhanced_final.csv'
    if output_csv.exists():
        print(f"File '{output_csv}' already exists. Skipping processing.")
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(nn_data(only_best_accuracy=False))
 
        if 'task' in df.columns:
            df = df[df['task'].str.contains('img-classification|image|vision', case=False, na=False)].copy()

        group_keys = ['task', 'dataset', 'nn']
        idxmax = df.groupby(group_keys)['accuracy'].idxmax()
        bests = df.loc[idxmax, group_keys + ['accuracy', 'epoch']]
        bests = bests.rename(columns={'accuracy': 'best_accuracy', 'epoch': 'best_epoch'})
        max_epochs = df.groupby(group_keys)['epoch'].max().rename('max_epoch')

        early1 = df[df['epoch'] == 1].groupby(group_keys)['accuracy'].max().rename('best_epoch_1_accuracy')
        early2 = df[df['epoch'] == 2].groupby(group_keys)['accuracy'].max().rename('best_epoch_2_accuracy')
        df = df.merge(bests, on=group_keys, how='left')
        df = df.merge(max_epochs, on=group_keys, how='left')
        df = df.merge(early1, on=group_keys, how='left')
        df = df.merge(early2, on=group_keys, how='left')
        cols = [c for c in [
            'task', 'dataset', 'metric', 'metric_code', 'nn', 'nn_code', 'duration', 'transform_code', 'prm',
            'epoch', 'accuracy', 'best_accuracy', 'best_epoch', 'max_epoch',
            'best_epoch_1_accuracy', 'best_epoch_2_accuracy'
        ] if c in df.columns]
        df[cols].to_csv(output_csv, index=False)
        print(f"Processed and saved to '{output_csv}'")
        df = df[cols]

    nns = df['nn'].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(nns)
    n_train = int(len(nns) * TRAIN_FRAC)
    train_nns = set(nns[:n_train])
    test_nns  = set(nns[n_train:])
    tr_df = df[df['nn'].isin(train_nns)].reset_index(drop=True)
    ev_df = df[df['nn'].isin(test_nns)].reset_index(drop=True)
    print(f"split -> train {len(tr_df)} | test {len(ev_df)}")
    print(f"Unique nn in train: {len(train_nns)}, test: {len(test_nns)}")
    assert train_nns.isdisjoint(test_nns), "Train and test nn sets overlap!"
    return tr_df, ev_df

# =======================
# Prompt builder (keeps your '### Response:' convention)
# =======================
def create_prompt(row, tpl, include_answer=True):
    p = tpl.format(
        model_code=row.get('nn_code',''),
        hyperparameters=row.get('prm',{}),
        epoch_1_accuracy=row.get('epoch_1_accuracy',''),
        epoch_2_accuracy=row.get('epoch_2_accuracy',''),
        epoch_3_accuracy=row.get('epoch_3_accuracy',''),
        best_epoch_1_accuracy=row.get('best_epoch_1_accuracy',''),
        best_epoch_2_accuracy=row.get('best_epoch_2_accuracy',''),
        dataset=row.get('dataset',''),
        task_type=row.get('task',''),
        transform_code=row.get('transform_code',''),
        metric_code=row.get('metric_code',''),
        max_epoch=int(row.get('max_epoch',0)) if not pd.isna(row.get('max_epoch', np.nan)) else 0,
    )
    if include_answer:
        # Keep your simple gold target format (compatible with your regex)
        ans = f"best_accuracy: {row['best_accuracy']:.4f}\nbest_epoch: {int(row['best_epoch'])}"
        return f"{p}\n\n### Response:\n{ans}"
    return f"{p}\n\n### Response:\n"

# =======================
# Model loader
# =======================
def _best_dtype_args():
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16_ok, "fp16": not bf16_ok}

def load_model_and_tokenizer(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    max_mem = {0: "20GiB"} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quantization_config_4bit,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if _best_dtype_args()["bf16"] else torch.float16,
        trust_remote_code=True,
        max_memory=max_mem
    )
    try:
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass
    return model, tok

# =======================
# Tokenization
# =======================
def tokenize_dataset(df, tok, tpl, max_len):
    texts = [create_prompt(r, tpl) for _, r in df.iterrows()]
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(ex):
        out = tokenize(ex["text"], tok)
        if "input_ids" in out and len(out["input_ids"]) > max_len:
            out["input_ids"]      = out["input_ids"][:max_len]
            out["attention_mask"] = out["attention_mask"][:max_len]
        return out
    return ds.map(tok_fn, batched=True, remove_columns=["text"])

# =======================
# Callback
# =======================
class LossPrinter(TrainerCallback):
    """Print loss and sample prompts/responses during training"""
    def __init__(self, tokenizer, train_df, tpl):
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.tpl = tpl
        self.sample_printed = False
    
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            print(f"\n{'='*80}")
            print(f"Step {int(state.global_step)} | Loss {logs['loss']:.4f}")
            print(f"{'='*80}\n", flush=True)
            # Print sample prompt/response every 50 steps
            if not self.sample_printed or int(state.global_step) % 50 == 0:
                self.sample_printed = True
                sample_idx = int(state.global_step) % max(1, len(self.train_df))
                row = self.train_df.iloc[sample_idx]
                prompt = create_prompt(row, self.tpl, include_answer=True)
                print(f"\n{'#'*80}")
                print("### TRAINING SAMPLE PROMPT & TARGET ###")
                print(f"{'#'*80}")
                print(prompt)
                print(f"{'#'*80}\n", flush=True)

# =======================
# Train
# =======================
def train():
    print("\n==== TRAIN ====\n", flush=True)
    tpl = load_prompt_template()
    tr_df, ev_df = prepare_data()
    print(f"[DEBUG] Train size: {len(tr_df)}, Eval size: {len(ev_df)}", flush=True)

    model, tok = load_model_and_tokenizer(MODEL_NAME)

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    print("✅ Gradient checkpointing enabled (reentrant=False)")

    lora_config = create_peft_config(["q_proj","k_proj"])
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Diagnostics
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count == 0:
        print(f"[WARNING] All parameters are frozen! Trainable params: 0 / {total_count}. LoRA may not be applied correctly.")
    else:
        print(f"[INFO] Trainable params: {trainable_count} / {total_count} ({100.0*trainable_count/total_count:.2f}%)")

    tr_ds = tokenize_dataset(tr_df, tok, tpl, MAX_LENGTH)
    ev_ds = tokenize_dataset(ev_df, tok, tpl, MAX_LENGTH)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    dtype_flags = _best_dtype_args()

    args = TrainingArguments(
        output_dir=str(BASE_DIR / "ckpts"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,        # REDUCED from 12 to 4
        learning_rate=1e-5,
        warmup_steps=1,
        logging_steps=10,
        save_steps=3,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        **dtype_flags,
        report_to=[],
        eval_accumulation_steps=4,            # REDUCED from 16 to 4
        group_by_length=False,
        torch_compile=False,
        max_grad_norm=1.0,                    # gradient clipping
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=False, pad_to_multiple_of=64
    )

    # Initialize callback with training data for printing samples
    loss_printer = LossPrinter(tok, tr_df, tpl)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=ev_ds,
        data_collator=data_collator,
        callbacks=[loss_printer],
    )

    print("[DEBUG] Starting training", flush=True)
    trainer.train()
    print("[DEBUG] Training complete", flush=True)

    torch.cuda.empty_cache()
    eval_res = trainer.evaluate()
    if "eval_loss" in eval_res:
        ppl = math.exp(eval_res["eval_loss"]) if eval_res["eval_loss"] < 20 else float("inf")
        print(f"eval_loss {eval_res['eval_loss']:.4f} | ppl {ppl:.2f}", flush=True)

    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    print(f"✅ saved -> {MODEL_DIR}", flush=True)

# =======================
# Eval (minimal safer parsing)
# =======================
def _clamp_epoch(ep, max_epoch):
    try:
        ep = int(ep)
        if pd.isna(max_epoch):
            return max(1, ep)
        return max(1, min(int(max_epoch), ep))
    except Exception:
        return np.nan

def evaluate(n=50, max_new_tokens=150):
    print("\n==== EVAL ====\n")
    tpl = load_prompt_template()
    _, te_df = prepare_data(TRAIN_FRAC, SEED)
    if n: te_df = te_df.head(n)
    base, tok = load_model_and_tokenizer(MODEL_NAME)
    model = PeftModel.from_pretrained(base, str(MODEL_DIR)) if MODEL_DIR.exists() else base
    model.eval()

    rows = []
    from tqdm import tqdm
    for _, row in tqdm(te_df.iterrows(), total=len(te_df), desc="Generating"):
        prompt = create_prompt(row, tpl, include_answer=False)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                temperature=0.7
            )
        resp = tok.decode(out[0], skip_special_tokens=True).split("### Response:")[-1].strip()
        print("\n" + "-"*80)
        print("### PROMPT ###"); print(prompt)
        print("\n### RAW RESPONSE ###"); print(resp)
        print("-"*80)
        # Minimal, safer extraction:
        # 1) Accuracy in [0,1]
        acc_match = re.findall(r'\bbest_accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)', resp)
        try:
            pred_acc = float(acc_match[-1]) if acc_match else np.nan
            if not (0.0 <= pred_acc <= 1.0):
                pred_acc = np.nan
        except Exception:
            pred_acc = np.nan

        # 2) Prefer "best_epoch", then fallback (avoid matching epoch_1/2/3)
        ep_match = re.findall(r'\bbest_epoch\s*[:=]\s*([0-9]+)', resp)
        if not ep_match:
            ep_match = re.findall(r'(?<!_)\\bepoch\\b\\s*[:=\\(]\\s*([0-9]+)', resp)  # won't match epoch_1/2/3
        try:
            pred_epoch = int(ep_match[-1]) if ep_match else np.nan
            pred_epoch = _clamp_epoch(pred_epoch, row.get("max_epoch", np.nan))
        except Exception:
            pred_epoch = np.nan

        rows.append({
            "dataset": row["dataset"], "nn": row["nn"],
            "true_accuracy": float(row["best_accuracy"]),
            "pred_accuracy": float(pred_acc) if not np.isnan(pred_acc) else np.nan,
            "true_epoch": int(row["best_epoch"]),
            "pred_epoch": int(pred_epoch) if not (isinstance(pred_epoch, float) and np.isnan(pred_epoch)) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "predictions.csv", index=False)

    y_true, y_pred = df["true_accuracy"].values, df["pred_accuracy"].values
    m = ~ (np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() > 0:
        yt, yp = y_true[m], y_pred[m]
        rmse = float(np.sqrt(np.mean((yt-yp)**2)))
        mae  = float(np.mean(np.abs(yt-yp)))
        w5   = float(np.mean(np.abs(yt-yp) <= 0.05)*100)
        w10  = float(np.mean(np.abs(yt-yp) <= 0.10)*100)
        print("\n📈 metrics:")
        print(f"RMSE {rmse:.4f} | MAE {mae:.4f} | ≤5% {w5:.1f}% | ≤10% {w10:.1f}% | valid {m.sum()}/{len(y_true)}")
    else:
        print("⚠️ no valid predictions")
    print(f"✅ saved -> {BASE_DIR / 'predictions.csv'}")

# =======================
# Main
# =======================
if __name__ == "__main__":
    train()
    evaluate(n=None)

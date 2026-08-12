#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B (Unsloth 4-bit) for outcome prediction with early stopping.

Pipeline:
  1. data_preprocessing   — nn_dataset API → llm_finetuning_data JSONL/CSV
  2. prepare_llm_datasets — raw JSONL → ChatML train/val/test splits
  3. train_model          — QLoRA fine-tune with validation + early stopping
  4. test_model           — evaluate on test set, save predictions and metrics

Usage:
    python AccPredictor.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from ab.nn.util.Const import out_dir
from ab.gpt.util.method.zero_cost_proxies import PROXY_NAMES, normalize_proxy_value, compute_proxies, DEFAULT_PROXY_NORM_STATS
# semantic_retrieval is imported lazily where used (only when USE_SEMANTIC_RETRIEVAL).

try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import EarlyStoppingCallback, TrainingArguments
    from datasets import Dataset
    import torch
    from tqdm import tqdm
except ImportError as e:
    raise RuntimeError(
        "Missing required package. Install: pip install unsloth[colab-new] trl datasets transformers"
    ) from e

ACC_DIR = out_dir / "acc_predict"

RAW_INPUT_PATH = ACC_DIR / "llm_finetuning_data.jsonl"
RAW_CSV_PATH = ACC_DIR / "llm_finetuning_data.csv"
DEFAULT_TRAIN_PATH = ACC_DIR / "train_llm_dataset.jsonl"
DEFAULT_VAL_PATH = ACC_DIR / "val_llm_dataset.jsonl"
DEFAULT_TEST_PATH = ACC_DIR / "test_llm_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ACC_DIR / "tuned_model"
DEFAULT_TEST_OUTPUT_PATH = ACC_DIR / "test_predictions.csv"
DEFAULT_TEST_METRICS_PATH = ACC_DIR / "test_metrics.log"
TEST_MAX_NEW_TOKENS = 64
TEST_TEMPERATURE = 0.0

# Zero-cost proxies (see out/IMPROVEMENT_PLAN_Zero_Cost_Proxies.md, Step 3).
# Explicit toggle rather than "use it if the files happen to exist": makes
# the A/B ablation (baseline vs +proxies) an intentional, visible switch,
# and fails loudly if the cache is missing while this is True rather than
# silently falling back to the baseline prompt (which would look like a
# proxy run but secretly not be one).
# Toggled True here for the Fisher-added arm (see
# out/IMPROVEMENT_PLAN_Additional_Proxies.md) -- testing Fisher (the
# strongest single proxy found in the transfer-check, r=0.532) added to the
# current best config (2-epoch, no-code, 5 proxies: R^2=0.820/0.056), one
# variable at a time per the discipline established after the 3-epoch +
# proxies negative result.
USE_ZERO_COST_PROXIES = True
PROXY_CACHE_PATH = ACC_DIR / "proxy_cache.jsonl"
PROXY_NORM_STATS_PATH = ACC_DIR / "proxy_norm_stats.json"

# Network-statistics feature (see out/IMPROVEMENT_PLAN_NN_Stat_Feature.md).
# Adds the single most-predictive nn_stat structural statistic that is NOT
# already captured by the proxies. Feature selection (ab/gpt: select_stat
# analysis) ranked features by |Spearman| with dataset-adjusted best_accuracy
# on the train family only; FLOPs was the strongest novel, continuous,
# cross-family-transferable predictor (rho=+0.130, only 0.71-correlated with
# log_params so it carries genuinely new information). Cache holds the
# train-only z-scored log1p(FLOPs) per architecture.
USE_NN_STATS = False
NN_STATS_CACHE_PATH = ACC_DIR / "nn_stat_flops_cache.json"

# Which proxies actually appear in the prompt -- a SUBSET of everything in
# the cache (zero_cost_proxies.PROXY_NAMES now has 8: the original 5 plus
# SNIP/Fisher/GraSP, see IMPROVEMENT_PLAN_Additional_Proxies.md). Every prior
# proxy experiment implicitly used all of PROXY_NAMES; now that the cache has
# more fields than any experiment has tested, this explicit subset is
# required to keep experiments a genuine one-variable-at-a-time comparison.
# The SWAP test (depth -> fisher) was tried but underperformed: the fisher
# variant scored 0.758 vs the depth set's 0.820 (Experiment 7 / index line 27),
# confirming the regression was not about signal COUNT. The original 5-proxy
# set with `depth` is the best-accuracy proxy configuration (0.820) and is the
# set the published ABrain/Accuracy-Prediction checkpoint was trained with, so
# inference must inject this same set. See IMPROVEMENT_PLAN_Additional_Proxies.md.
ACTIVE_PROXY_NAMES = ("synflow", "nwot", "grad_norm", "log_params", "depth")

# Reset to False here: the 3-early-epoch experiment (see
# out/IMPROVEMENT_PLAN_Three_Early_Epochs.md) isolates ONLY the epoch-count
# variable against the early-signals-only baseline, so every other extra
# signal stays off.
USE_SEMANTIC_RETRIEVAL = False
SEMANTIC_RETRIEVAL_K = 3

# Source code in the prompt (see out/IMPROVEMENT_PLAN_Early_Signals_Only.md).
# The paper's own third comparison arm ("Qwen3 Early Signals Only" vs.
# "Qwen3 + Source Code", Table 2) was never implemented in this codebase --
# this toggle adds it. False = omit the NEURAL_NETWORK_CODE block entirely,
# leaving only early accuracy + task/dataset/metric/budget metadata.
USE_SOURCE_CODE = False

# Third early-training-signal epoch (see out/IMPROVEMENT_PLAN_Three_Early_Epochs.md).
# The paper's own problem formulation (s = s1, s2, s3) uses THREE early
# epochs; this codebase originally only extracted/used two (a deviation
# flagged in the project's first analysis). Requires Stage 1
# (data_preprocessing) to have been re-run with _DP_EARLY_EPOCHS including 3.
# Reset to False for the Fisher-added test (out/IMPROVEMENT_PLAN_Additional_Proxies.md)
# -- comparing against the 2-epoch + 5-proxy baseline specifically (R^2=0.820/0.056),
# not the 3-epoch one, to keep this a genuine single-variable test.
USE_THIRD_EARLY_EPOCH = False

# Chain-of-thought reasoning output (see out/IMPROVEMENT_PLAN_Chain_Of_Thought.md).
# When True, the assistant target becomes a short input-grounded reasoning
# trace followed by the same JSON answer, and the model is asked to reason
# before answering -- giving it more effective compute per prediction rather
# than more input fields (which the "signal capacity ceiling" has repeatedly
# shown backfire). Tested against the 2-epoch + 5-proxy reference
# (R^2=0.820/0.056) as a single-variable change: everything else identical.
USE_CHAIN_OF_THOUGHT = False

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MODEL_FALLBACKS = ("unsloth/Qwen3-8B-bnb-4bit", "Qwen/Qwen3-8B")
DEFAULT_MAX_SEQ_LEN = 6144

PREDICTOR_HF_REPO = "ABrain/Accuracy-Prediction"
PREDICTOR_MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN
PREDICTOR_MAX_NEW_TOKENS = 64
PREDICTOR_DEFAULT_MAX_EPOCHS = 50

_predictor_model = None
_predictor_tokenizer = None

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.013
WARMUP_RATIO = 0.02
MAX_GRAD_NORM = 0.5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 20
MAX_STEPS = 0
SEED = 42
EARLY_STOPPING_PATIENCE = 3
SAVE_TOTAL_LIMIT = 2

TRAIN_ARCH_FAMILIES = frozenset({"cnn", "segmentation"})
VAL_ARCH_FAMILIES = frozenset({"detector", "transformer", "rnn"})
TEST_ARCH_FAMILIES = frozenset({"other"})

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

COT_SYSTEM_PROMPT = """You are a neural-network performance predictor.
First reason briefly (a few short lines) about the training signals you are
given, then output your final answer as exactly ONE JSON object.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float representing the predicted final best validation accuracy.
best_epoch must be a positive integer representing the epoch where peak validation accuracy occurs.

Put your reasoning first, then a line that begins with OUTPUT: followed by the JSON.
Stop immediately after the closing brace }.
"""


def _active_system_prompt() -> str:
    return COT_SYSTEM_PROMPT if USE_CHAIN_OF_THOUGHT else SYSTEM_PROMPT


def _build_reasoning_trace(record: dict, targets: dict) -> str:
    """Synthesize a short, input-grounded reasoning trace ending in the answer.

    The rationale references only signals available at inference time (early
    accuracy trajectory, training budget, task/dataset, and the presence of
    zero-cost proxy scores) and reasons toward the known training target. It
    contains no '{' so the JSON answer that follows remains the first brace in
    the message (the test-time parser extracts the first {...} object)."""
    e1 = _safe_float(record.get("accuracy_epoch_1"), 0.0)
    e2 = _safe_float(record.get("accuracy_epoch_2"), 0.0)
    delta = e2 - e1
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    best_acc = _safe_float(targets.get("best_accuracy"), 0.0)
    best_epoch = _safe_int(targets.get("best_epoch"), 1)

    if delta >= 0.10:
        climb = "a steep early climb"
    elif delta >= 0.03:
        climb = "moderate early progress"
    elif delta > -0.03:
        climb = "nearly flat early progress"
    else:
        climb = "an early dip in accuracy"

    if best_acc >= 0.75:
        region = "a high"
    elif best_acc >= 0.50:
        region = "a solid mid-to-high"
    elif best_acc >= 0.30:
        region = "a moderate"
    else:
        region = "a low"

    frac = best_epoch / max_epochs if max_epochs else 1.0
    if frac < 0.34:
        timing = "early"
    elif frac < 0.67:
        timing = "middle"
    else:
        timing = "later"

    proxy_note = (
        " The zero-cost proxy scores give a training-free read on this "
        "architecture's quality, complementing the early curve."
        if USE_ZERO_COST_PROXIES else ""
    )
    lines = [
        "Reasoning:",
        f"- Task: {record.get('task', '')} on {record.get('dataset', '')}, "
        f"with a budget of {max_epochs} epochs.",
        f"- Early validation accuracy is {e1 * 100:.1f}% after epoch 1 and "
        f"{e2 * 100:.1f}% after epoch 2 ({climb}).{proxy_note}",
        f"- {climb.capitalize()} on a {max_epochs}-epoch budget points to "
        f"{region} final accuracy, with the peak reached in the {timing} "
        "portion of training.",
        "OUTPUT:",
    ]
    return "\n".join(lines)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _resolve_call_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_call_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _layer_counts(nn_code: str) -> dict[str, int]:
    """Count layer types in nn_code (AST parse with regex fallback)."""
    counts = {
        "conv2d": 0,
        "attention": 0,
        "lstm": 0,
        "gru": 0,
        "rnn": 0,
    }
    nn_code = nn_code or ""

    try:
        tree = ast.parse(nn_code)

        class LayerCounter(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                call_name = _resolve_call_name(node.func)
                if call_name is None:
                    self.generic_visit(node)
                    return

                if call_name in ("nn.Conv2d", "torch.nn.Conv2d", "Conv2d"):
                    counts["conv2d"] += 1
                if call_name in (
                    "nn.MultiheadAttention",
                    "torch.nn.MultiheadAttention",
                    "MultiheadAttention",
                ):
                    counts["attention"] += 1
                if call_name in ("nn.LSTM", "torch.nn.LSTM", "LSTM"):
                    counts["lstm"] += 1
                if call_name in ("nn.GRU", "torch.nn.GRU", "GRU"):
                    counts["gru"] += 1
                if call_name in ("nn.RNN", "torch.nn.RNN", "RNN"):
                    counts["rnn"] += 1

                self.generic_visit(node)

        LayerCounter().visit(tree)
    except (SyntaxError, ValueError):
        counts["conv2d"] = len(re.findall(r"\b(nn|torch\.nn)\.Conv2d\s*\(|\bConv2d\s*\(", nn_code))
        counts["attention"] = len(
            re.findall(
                r"\b(nn|torch\.nn)\.MultiheadAttention\s*\(|\bMultiheadAttention\s*\(",
                nn_code,
            )
        )
        counts["lstm"] = len(re.findall(r"\b(nn|torch\.nn)\.LSTM\s*\(|\bLSTM\s*\(", nn_code))
        counts["gru"] = len(re.findall(r"\b(nn|torch\.nn)\.GRU\s*\(|\bGRU\s*\(", nn_code))
        counts["rnn"] = len(re.findall(r"\b(nn|torch\.nn)\.RNN\s*\(|\bRNN\s*\(", nn_code))

    return counts


def infer_arch_family(nn_code: str, task: str) -> str:
    """
    Infer architecture family from nn_code and task.

    Mirrors the arch_family logic in extractor.extract_model_summary.
    Returns one of: detector, segmentation, transformer, rnn, cnn, other.
    """
    nn_code = nn_code or ""
    task = (task or "").lower()
    counts = _layer_counts(nn_code)

    if "detect" in task or re.search(r"\b(SSD|YOLO|RCNN|RetinaNet)\b", nn_code, re.IGNORECASE):
        return "detector"
    if "segmentation" in task or re.search(r"\b(FCN|DeepLab|UNet|LRASPP)\b", nn_code, re.IGNORECASE):
        return "segmentation"
    if counts["attention"] > 0 or re.search(r"\b(Transformer|MultiheadAttention|ViT|Swin)\b", nn_code):
        return "transformer"
    if counts["lstm"] > 0 or counts["gru"] > 0 or counts["rnn"] > 0:
        return "rnn"
    if counts["conv2d"] > 0:
        return "cnn"
    return "other"


def _arch_family(record: dict, nn_code: str) -> str:
    return infer_arch_family(nn_code, record.get("task", "") or "")


def _compute_targets(record: dict) -> dict:
    best_accuracy = max(0.0, min(_safe_float(record.get("best_accuracy"), 0.0), 100.0))
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    best_epoch = max(1, min(_safe_int(record.get("epochs_to_best"), 1), max_epochs))
    return {"best_accuracy": best_accuracy, "best_epoch": best_epoch}


def _format_proxy_lines(record: dict) -> list[str]:
    """
    ZERO_COST_PROXIES block (see out/IMPROVEMENT_PLAN_Zero_Cost_Proxies.md).
    record is expected to carry normalized values under "proxy_<name>" keys
    (set by prepare_llm_datasets before this is called). Proxies that are
    missing for this specific architecture (~2.6% of cases, mostly detection
    nets whose forward() doesn't return a plain tensor) are omitted line by
    line rather than filled with a placeholder -- a missing line is a
    cleaner signal than an arbitrary "unavailable" token the model might
    otherwise learn to treat as meaningful.
    """
    lines = []
    for name in ACTIVE_PROXY_NAMES:
        value = record.get(f"proxy_{name}")
        if value is not None:
            lines.append(f"{name}: {round(float(value), 4)}")
    if not lines:
        return []
    return ["ZERO_COST_PROXIES (training-free, normalized architecture signals)", *lines, ""]


def _format_nn_stats_lines(record: dict) -> list[str]:
    """NETWORK_STATISTICS block: the single most-predictive nn_stat feature
    (normalized log-FLOPs, the model's compute cost). Omitted entirely when
    the value is missing, same convention as the proxy block."""
    if not USE_NN_STATS:
        return []
    value = record.get("nn_stat_flops")
    if value is None:
        return []
    return [
        "NETWORK_STATISTICS (structural, normalized)",
        f"compute_flops: {round(float(value), 4)}",
        "",
    ]


def _format_similar_architectures_lines(record: dict) -> list[str]:
    """
    SIMILAR_ARCHITECTURES block (see out/PROGRESS_Semantic_Retrieval.md).
    record is expected to carry a "similar_architectures" key -- a list of
    neighbor dicts from SemanticIndex.retrieve() (set by prepare_llm_datasets
    before this is called). Plain-language precedent, not raw numbers: this
    is deliberately a different mechanism from ZERO_COST_PROXIES (which
    required the model to learn to interpret abstract scores, and did not
    show a benefit across two separate training runs) -- real, similar
    architectures with their actual known outcomes, which an LLM can use
    as in-context precedent without learning a new numeric-interpretation
    skill. Same scale convention as EARLY_TRAINING_SIGNAL (0-1, not
    percentage) for internal consistency.
    """
    neighbors = record.get("similar_architectures") or []
    if not neighbors:
        return []
    lines = []
    for i, nb in enumerate(neighbors, 1):
        acc = _safe_float(nb.get("best_accuracy"), None)
        epoch = nb.get("epochs_to_best")
        if acc is None or epoch is None:
            continue
        lines.append(
            f"{i}. {nb.get('family', 'unknown')} architecture on {nb.get('dataset', 'unknown')}: "
            f"reached {round(acc, 4)} accuracy, peaked at epoch {int(epoch)}."
        )
    if not lines:
        return []
    return ["SIMILAR_ARCHITECTURES (retrieved by structural similarity, from training set)", *lines, ""]


def _format_code_lines(nn_code: str) -> list[str]:
    """NEURAL_NETWORK_CODE block, omitted entirely when USE_SOURCE_CODE is
    False (see out/IMPROVEMENT_PLAN_Early_Signals_Only.md) -- this is the
    paper's own "Qwen3 Early Signals Only" ablation arm, never implemented
    in this codebase until now."""
    if not USE_SOURCE_CODE:
        return []
    return ["NEURAL_NETWORK_CODE", "```python", (nn_code or "").strip(), "```", ""]


def _build_user_message(record: dict, nn_code: str) -> str:
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    architecture_lines = ["- Architecture depth and complexity"] if USE_SOURCE_CODE else []
    lines = [
        "INPUT",
        f"task: {record.get('task', '')}",
        f"dataset: {record.get('dataset', '')}",
        f"metric: {record.get('metric', '')}",
        "",
        "TRAINING_BUDGET",
        f"max_epochs: {max_epochs}",
        "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {round(_safe_float(record.get('accuracy_epoch_1'), 0.0), 6)}",
        f"epoch_2_accuracy: {round(_safe_float(record.get('accuracy_epoch_2'), 0.0), 6)}",
        *([f"epoch_3_accuracy: {round(_safe_float(record.get('accuracy_epoch_3'), 0.0), 6)}"] if USE_THIRD_EARLY_EPOCH else []),
        "",
        *_format_proxy_lines(record),
        *_format_nn_stats_lines(record),
        *_format_code_lines(nn_code),
        *_format_similar_architectures_lines(record),
        "Analyze the training dynamics"
        + (", architecture complexity, and" if USE_SOURCE_CODE else " and")
        + " optimization hyperparameters to estimate the final training outcome.",
        "",
        "Important signals to consider:",
        "- Early learning progress (epoch accuracies)",
        "- Saturation of improvement across epochs",
        *architecture_lines,
        "- Optimization scale (learning rate, batch size, effective_lr)",
        "- Zero-cost proxy scores, if present (training-free estimates of architecture quality that generalize across architecture families better than early accuracy alone)",
        "- Outcomes of similar architectures, if present (real precedents from structurally similar networks)",
        "",
        "Using these signals, estimate the final best validation accuracy and the epoch where it occurs.",
        "",
        "The value best_epoch represents the training epoch where the highest validation accuracy is reached.",
        "",
        "Constraints:",
        "- best_epoch must be an integer between 1 and max_epochs.",
        "",
        (
            "First reason briefly through the signals above, then end with a line "
            "starting with OUTPUT: followed by the JSON answer."
            if USE_CHAIN_OF_THOUGHT
            else "OUTPUT (JSON ONLY):"
        ),
    ]
    return "\n".join(lines)


def _parse_prediction_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _get_predictor_model():
    global _predictor_model, _predictor_tokenizer
    if _predictor_model is None or _predictor_tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PREDICTOR_HF_REPO,
            max_seq_length=PREDICTOR_MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)
        _predictor_model = model
        _predictor_tokenizer = tokenizer
    return _predictor_model, _predictor_tokenizer


def _apply_chat_template_for_inference(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def predict_best_accuracy(
    task: str,
    dataset: str,
    metric: str,
    nn_code: str,
    epoch_1_accuracy: float,
    epoch_2_accuracy: float,
    prm: dict | None = None,
) -> tuple[float, int]:
    """Predict best_accuracy and best_epoch using ABrain/Accuracy-Prediction.
    If proxies are enabled and nn_code is given, they are computed from the code
    and added to the prompt; prm is only used for that computation."""
    record = {
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "total_epochs": PREDICTOR_DEFAULT_MAX_EPOCHS,
        "accuracy_epoch_1": epoch_1_accuracy,
        "accuracy_epoch_2": epoch_2_accuracy,
    }
    _attach_computed_proxies(record, nn_code, prm)
    user_content = _build_user_message(record, nn_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model, tokenizer = _get_predictor_model()
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    prompt_text = _apply_chat_template_for_inference(text_tokenizer, messages)
    input_ids = text_tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(input_ids) > PREDICTOR_MAX_SEQ_LEN:
        input_ids = input_ids[:PREDICTOR_MAX_SEQ_LEN]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids_t)
    stop_token_ids = []
    for tok_str in ("}", "<|im_end|>", "<|endoftext|>"):
        ids = text_tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])
    if text_tokenizer.eos_token_id is not None:
        stop_token_ids.append(text_tokenizer.eos_token_id)
    stop_token_ids = list(set(stop_token_ids))

    with torch.no_grad():
        outputs = model.generate(
            **{"input_ids": input_ids_t, "attention_mask": attention_mask},
            max_new_tokens=PREDICTOR_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
            eos_token_id=stop_token_ids or None,
            use_cache=False,
        )

    generated = text_tokenizer.decode(
        outputs[0][input_ids_t.shape[1] :],
        skip_special_tokens=True,
    )
    parsed = _parse_prediction_json(generated)

    fallback_acc = max(float(epoch_1_accuracy), float(epoch_2_accuracy))
    fallback_epoch = 2 if float(epoch_2_accuracy) >= float(epoch_1_accuracy) else 1

    if not parsed:
        return fallback_acc, fallback_epoch

    try:
        best_accuracy = float(parsed.get("best_accuracy", fallback_acc))
    except (TypeError, ValueError):
        best_accuracy = fallback_acc

    try:
        best_epoch = int(parsed.get("best_epoch", fallback_epoch))
        if best_epoch < 1:
            best_epoch = fallback_epoch
    except (TypeError, ValueError):
        best_epoch = fallback_epoch

    return best_accuracy, best_epoch


def _record_to_chatml(record: dict) -> dict | None:
    required = {"task", "dataset", "metric", "nn", "nn_code", "best_accuracy", "epochs_to_best"}
    if not all(k in record for k in required):
        return None

    nn_code = record.get("nn_code", "") or ""
    if not isinstance(nn_code, str):
        return None

    targets = _compute_targets(record)
    answer_json = json.dumps(
        {
            "best_accuracy": round(targets["best_accuracy"], 2),
            "best_epoch": int(targets["best_epoch"]),
        },
        separators=(",", ":"),
    )
    if USE_CHAIN_OF_THOUGHT:
        assistant_content = _build_reasoning_trace(record, targets) + " " + answer_json
    else:
        assistant_content = answer_json
    return {
        "messages": [
            {"role": "system", "content": _active_system_prompt()},
            {"role": "user", "content": _build_user_message(record, nn_code)},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _stream_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _split_by_architecture_family(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    family_to_records: dict[str, list[dict]] = {}
    for rec in records:
        raw = rec["raw"]
        nn_code = raw.get("nn_code", "") or ""
        family = _arch_family(raw, nn_code) if isinstance(nn_code, str) else "other"
        family_to_records.setdefault(family, []).append(rec)

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    for family, recs in family_to_records.items():
        if family in TRAIN_ARCH_FAMILIES:
            train_records.extend(recs)
        elif family in VAL_ARCH_FAMILIES:
            val_records.extend(recs)
        elif family in TEST_ARCH_FAMILIES:
            test_records.extend(recs)
    return train_records, val_records, test_records


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- data_preprocessing  ---

_DP_GROUP_KEYS = ("nn_id", "prm_id", "transform_id", "dataset")
_DP_REQUIRED_COLS = ("id", "nn_id", "prm_id", "transform_id", "dataset", "epoch", "accuracy")
_DP_EARLY_EPOCHS = (1, 2, 3)
_DP_MIN_EPOCHS = 50
_DP_OUTPUT_COLUMNS = [
    "id",
    "task",
    "dataset",
    "metric",
    "nn",
    "nn_code",
    "transform_code",
    "prm",
    "accuracy_epoch_1",
    "accuracy_epoch_2",
    "accuracy_epoch_3",
    "best_accuracy",
    "epochs_to_best",
    "total_epochs",
]


def _dp_load_nn_data() -> pd.DataFrame:
    try:
        from ab.nn.api import data as nn_data
    except ImportError as exc:
        raise ImportError(
            "Could not import ab.nn.api.data. Install the nn_dataset package."
        ) from exc

    df = pd.DataFrame(nn_data(only_best_accuracy=False))
    missing = [col for col in _DP_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _dp_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["accuracy"].max() <= 1.0:
        df["accuracy"] = df["accuracy"] * 100
    df["epoch"] = df["epoch"].astype(int)
    return df


def _dp_filter_long_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    group_cols = list(_DP_GROUP_KEYS)
    total_runs = df.groupby(group_cols, observed=True).ngroups
    run_max_epoch = df.groupby(group_cols, observed=True)["epoch"].transform("max")
    filtered = df[run_max_epoch >= _DP_MIN_EPOCHS].copy()
    valid_runs = filtered.groupby(group_cols, observed=True).ngroups
    return filtered, total_runs, valid_runs


def _dp_parse_prm(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if pd.isna(value):
        return {}
    return value


def _dp_process_run(
    group_df: pd.DataFrame,
    dataset: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (record, drop_reason). drop_reason is None on success."""
    group_df = group_df.sort_values("epoch").drop_duplicates("epoch", keep="first")
    acc_by_epoch = group_df.set_index("epoch")["accuracy"]

    for epoch in _DP_EARLY_EPOCHS:
        if epoch not in acc_by_epoch.index:
            return None, f"missing_epoch_{epoch}"
        if pd.isna(acc_by_epoch.at[epoch]):
            return None, f"missing_epoch_{epoch}"

    best_accuracy = group_df["accuracy"].max()
    if pd.isna(best_accuracy):
        return None, "missing_best_accuracy"

    epochs_to_best = int(
        group_df.loc[group_df["accuracy"] == best_accuracy, "epoch"].min()
    )
    first_row = group_df.iloc[0]

    record = {
        "id": str(first_row["id"]),
        "task": first_row["task"],
        "dataset": dataset,
        "metric": first_row["metric"],
        "nn": first_row["nn"],
        "nn_code": first_row["nn_code"],
        "transform_code": first_row["transform_code"],
        "prm": _dp_parse_prm(first_row["prm"]),
        "accuracy_epoch_1": float(acc_by_epoch.at[1]),
        "accuracy_epoch_2": float(acc_by_epoch.at[2]),
        "accuracy_epoch_3": float(acc_by_epoch.at[3]),
        "best_accuracy": float(best_accuracy),
        "epochs_to_best": epochs_to_best,
        "total_epochs": int(group_df["epoch"].max()),
    }
    return record, None


def _dp_build_records(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    drop_counts: dict[str, int] = {}

    grouped = df.groupby(list(_DP_GROUP_KEYS), observed=True)
    for (_, _, _, dataset), group_df in tqdm(grouped, desc="Processing runs"):
        record, drop_reason = _dp_process_run(group_df, dataset)
        if record is None:
            drop_counts[drop_reason] = drop_counts.get(drop_reason, 0) + 1
            continue
        records.append(record)

    return records, drop_counts


def _dp_serialize_prm(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, dict) else value


def _dp_save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            row = {**record, "prm": _dp_serialize_prm(record["prm"])}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dp_save_csv(df_final: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df_final.empty:
        pd.DataFrame(columns=_DP_OUTPUT_COLUMNS).to_csv(path, index=False)
        return

    df_out = df_final.copy()
    df_out["prm"] = df_out["prm"].map(_dp_serialize_prm)
    df_out.to_csv(path, index=False)


def data_preprocessing(
    output_jsonl_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Prepare data for supervised fine-tuning from nn_dataset.

    Loads per-epoch training stats, keeps runs with >= 50 epochs, extracts
    accuracy at epochs 1 and 2 as input features, and computes best_accuracy
    and epochs_to_best from the full run.

    Args:
        output_jsonl_path: Optional path to save JSONL output.
        output_csv_path: Optional path to save CSV output.

    Returns:
        DataFrame with one row per valid training run.
    """
    print("Loading data from nn_dataset API...")
    raw_df = _dp_load_nn_data()
    rows_loaded = len(raw_df)
    print(f"Loaded {rows_loaded:,} rows")

    df = _dp_normalize_dataframe(raw_df)
    df, total_runs, valid_runs = _dp_filter_long_runs(df)
    kept_pct = (100 * valid_runs / total_runs) if total_runs else 0.0
    print(
        f"Runs with >= {_DP_MIN_EPOCHS} epochs: {valid_runs:,} / {total_runs:,} "
        f"({kept_pct:.1f}% kept)"
    )

    records, drop_counts = _dp_build_records(df)
    dropped = sum(drop_counts.values())
    df_final = pd.DataFrame(records)

    print(f"Final records: {len(df_final):,} (dropped {dropped:,})")
    for reason, count in sorted(drop_counts.items()):
        print(f"  - {reason}: {count:,}")

    if output_jsonl_path:
        jsonl_path = Path(output_jsonl_path)
        _dp_save_jsonl(records, jsonl_path)
        print(f"Wrote JSONL: {jsonl_path}")

    if output_csv_path:
        csv_path = Path(output_csv_path)
        _dp_save_csv(df_final, csv_path)
        print(f"Wrote CSV: {csv_path}")

    if valid_runs:
        success_rate = 100 * len(df_final) / valid_runs
        print(f"Success rate: {success_rate:.1f}% of long runs")

    return df_final


def _load_proxy_cache(path: Path) -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return cache
    for rec in _stream_jsonl(path):
        cache[(rec.get("nn"), rec.get("dataset"))] = rec
    return cache


def _load_proxy_norm_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _attach_proxy_features(record: dict, proxy_cache: dict, norm_stats: dict) -> None:
    """Mutates record in place: adds normalized proxy_<name> keys, read by
    _format_proxy_lines via _build_user_message. Architectures with no
    cached proxies (~2.6%, see IMPROVEMENT_PLAN_Zero_Cost_Proxies.md Step 2)
    are left untouched -- _format_proxy_lines already handles the resulting
    all-missing case by omitting the block entirely."""
    proxies = proxy_cache.get((record.get("nn"), record.get("dataset")))
    if not proxies:
        return
    for name in ACTIVE_PROXY_NAMES:
        raw_value = proxies.get(name)
        if raw_value is not None:
            record[f"proxy_{name}"] = normalize_proxy_value(name, raw_value, norm_stats)


_PREDICTOR_PROXY_NORM_STATS: dict | None = None

# Embedded fallback so inference works without a proxy_norm_stats.json file; a
# local file, if present, overrides it (see _get_predictor_proxy_norm_stats).
_DEFAULT_PROXY_NORM_STATS = DEFAULT_PROXY_NORM_STATS


def _get_predictor_proxy_norm_stats() -> dict:
    """Load and cache the proxy normalization stats: the on-disk
    proxy_norm_stats.json if present, else the embedded defaults."""
    global _PREDICTOR_PROXY_NORM_STATS
    if _PREDICTOR_PROXY_NORM_STATS is None:
        stats = _load_proxy_norm_stats(PROXY_NORM_STATS_PATH)
        _PREDICTOR_PROXY_NORM_STATS = stats if stats else _DEFAULT_PROXY_NORM_STATS
    return _PREDICTOR_PROXY_NORM_STATS


def _attach_computed_proxies(record: dict, nn_code: str, prm: dict | None = None) -> None:
    """Compute the active proxies from nn_code, normalize them, and store them as
    proxy_<name> keys on record. Best-effort: proxies that fail to compute are
    skipped."""
    if not USE_ZERO_COST_PROXIES or not (nn_code or "").strip():
        return
    norm_stats = _get_predictor_proxy_norm_stats()
    proxies = compute_proxies(nn_code, record.get("dataset", "") or "", prm=prm)
    for name in ACTIVE_PROXY_NAMES:
        raw_value = proxies.get(name)
        if raw_value is not None:
            record[f"proxy_{name}"] = normalize_proxy_value(name, raw_value, norm_stats)


def _attach_nn_stats(record: dict, cache: dict) -> None:
    """Mutates record in place: adds "nn_stat_flops" (the precomputed,
    train-only z-scored log-FLOPs) keyed by architecture name. Missing
    architectures are left untouched -- _format_nn_stats_lines omits the
    block when the value is absent."""
    value = cache.get(record.get("nn"))
    if value is not None:
        record["nn_stat_flops"] = value


def _attach_similar_architectures(record: dict, index: SemanticIndex, k: int) -> None:
    """Mutates record in place: adds a "similar_architectures" key, read by
    _format_similar_architectures_lines via _build_user_message. Reuses the
    query architecture's own precomputed embedding from the Step 2 cache
    (no fresh model inference here) and retrieves from the leakage-guarded
    train-only pool, excluding the query itself."""
    key = (record.get("nn"), record.get("dataset"))
    query_embedding = index.get_cached_embedding(*key)
    if query_embedding is None:
        return
    record["similar_architectures"] = index.retrieve(query_embedding, k=k, exclude_key=key)


def prepare_llm_datasets(
    input_path: Path = RAW_INPUT_PATH,
    output_dir: Path = ACC_DIR,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    proxy_cache: dict = {}
    proxy_norm_stats: dict = {}
    if USE_ZERO_COST_PROXIES:
        # Build the cache + stats on first run (cached to disk, resumable).
        if not PROXY_CACHE_PATH.exists():
            print("Proxy cache missing -- building it now...")
            from ab.gpt.util.method.compute_proxy_cache import build_proxy_cache
            build_proxy_cache(input_path=input_path, cache_path=PROXY_CACHE_PATH)
        if not PROXY_NORM_STATS_PATH.exists():
            print("Proxy normalization stats missing -- fitting them now (train-family only)...")
            from ab.gpt.util.method.fit_proxy_normalization import fit_and_save_norm_stats
            fit_and_save_norm_stats(
                raw_path=input_path,
                cache_path=PROXY_CACHE_PATH,
                stats_path=PROXY_NORM_STATS_PATH,
                family_fn=infer_arch_family,
                train_families=TRAIN_ARCH_FAMILIES,
            )
        proxy_cache = _load_proxy_cache(PROXY_CACHE_PATH)
        proxy_norm_stats = _load_proxy_norm_stats(PROXY_NORM_STATS_PATH)
        print(f"Loaded zero-cost proxy cache: {len(proxy_cache)} architectures")

    semantic_index = None
    if USE_SEMANTIC_RETRIEVAL:
        from ab.gpt.semantic_retrieval import SemanticIndex
        semantic_index = SemanticIndex.load()
        print(f"Loaded semantic embedding index: {len(semantic_index.all_meta)} architectures, "
              f"{len(semantic_index._train_keys)} in the training (retrievable) pool")

    nn_stats_cache: dict = {}
    if USE_NN_STATS:
        if not NN_STATS_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"USE_NN_STATS is True but {NN_STATS_CACHE_PATH} is missing. "
                "Run ab.gpt.build_nn_stat_cache first."
            )
        with open(NN_STATS_CACHE_PATH, encoding="utf-8") as fh:
            nn_stats_cache = json.load(fh)
        print(f"Loaded nn_stat FLOPs cache: {len(nn_stats_cache)} architectures")

    records: list[dict] = []
    skipped = 0
    for raw in tqdm(_stream_jsonl(input_path), desc="Processing"):
        if USE_ZERO_COST_PROXIES:
            _attach_proxy_features(raw, proxy_cache, proxy_norm_stats)
        if USE_NN_STATS:
            _attach_nn_stats(raw, nn_stats_cache)
        if USE_SEMANTIC_RETRIEVAL:
            _attach_similar_architectures(raw, semantic_index, SEMANTIC_RETRIEVAL_K)
        chatml = _record_to_chatml(raw)
        nn = raw.get("nn")
        if chatml is None or not raw.get("task") or not raw.get("dataset") or not nn:
            skipped += 1
            continue
        records.append({"nn": str(nn), "chatml": chatml, "raw": raw})

    if not records:
        raise ValueError(f"No valid records loaded from {input_path} (skipped {skipped})")

    train_data, val_data, test_data = _split_by_architecture_family(records)
    train_path = output_dir / "train_llm_dataset.jsonl"
    val_path = output_dir / "val_llm_dataset.jsonl"
    test_path = output_dir / "test_llm_dataset.jsonl"

    def _with_arch_id(rec: dict) -> dict:
        out = dict(rec["chatml"])
        out["architecture_id"] = rec["nn"]
        return out

    _write_jsonl(train_path, [_with_arch_id(r) for r in train_data])
    _write_jsonl(val_path, [_with_arch_id(r) for r in val_data])
    _write_jsonl(test_path, [_with_arch_id(r) for r in test_data])
    return train_path, val_path, test_path


def _load_messages(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict] = []
    for row in _stream_jsonl(path):
        if "messages" in row and row.get("architecture_id"):
            examples.append({"messages": row["messages"], "architecture_id": str(row["architecture_id"])})
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def _load_model(model_name: str, max_seq_len: int):
    candidates = [model_name, *(m for m in MODEL_FALLBACKS if m != model_name)]
    last_error: Exception | None = None

    for model_id in candidates:
        try:
            return FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_len,
                dtype=None,
                load_in_4bit=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load any Qwen3-8B checkpoint. Tried: {candidates}") from last_error


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


# --- model testing (from early_stopping_full_code_early_epochs_qwen8/test_model.py) ---


def _test_load_checkpoint(model_path: Path, max_seq_len: int):
    return FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )


def _test_load_data(path: Path, limit: int | None = None) -> list[dict]:
    data: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def _test_parse_assistant_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("Ġ", " ")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _test_get_ground_truth(messages: list[dict]) -> dict | None:
    for message in messages:
        if message.get("role") == "assistant":
            return _test_parse_assistant_json(message.get("content", ""))
    return None


def _test_build_prompt_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") in ("system", "user")]


def _test_extract_run_id(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            match = re.search(r"id:\s*(\S+)", message.get("content", ""))
            return match.group(1).strip() if match else ""
    return ""


def _test_extract_dataset(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            match = re.search(r"^dataset:\s*(.+)$", message.get("content", ""), re.MULTILINE)
            return match.group(1).strip() if match else ""
    return ""


MIN_N_FOR_DATASET_METRICS = 5  # below this, correlation is too noisy to report meaningfully


def _extended_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Pearson r, Spearman rho, R^2 alongside the existing RMSE/MAE -- these are
    the paper's actual headline metrics (Table 2/3), which this script did
    not compute before (see out/IMPROVEMENT_PLAN_Zero_Cost_Proxies.md §6).
    Returns None for any statistic that can't be computed (n<2 or zero
    variance in either array -- pearsonr/spearmanr are undefined there)
    rather than raising, since this is called per-dataset too, where small
    or degenerate groups are expected.
    """
    n = len(pred)
    result = {"n": n, "rmse": None, "mae": None, "pearson_r": None, "spearman_r": None, "r2": None}
    if n == 0:
        return result

    result["rmse"] = float(np.sqrt(np.mean((pred - gt) ** 2)))
    result["mae"] = float(np.mean(np.abs(pred - gt)))

    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    if ss_tot > 1e-12:
        result["r2"] = float(1 - ss_res / ss_tot)

    if n >= 2 and np.std(pred) > 1e-12 and np.std(gt) > 1e-12:
        result["pearson_r"] = float(pearsonr(pred, gt)[0])
        result["spearman_r"] = float(spearmanr(pred, gt)[0])

    return result


def _per_dataset_metrics(valid: list[dict]) -> dict:
    """Per-dataset best_accuracy breakdown, reproducing the paper's Table 3
    style. Groups with fewer than MIN_N_FOR_DATASET_METRICS rows are still
    reported (n and RMSE/MAE) but flagged low_n=True since Pearson/Spearman
    are unreliable at very small n."""
    by_dataset: dict[str, list[dict]] = {}
    for p in valid:
        by_dataset.setdefault(p["dataset"], []).append(p)

    out: dict[str, dict] = {}
    for dataset, rows in sorted(by_dataset.items()):
        pred = np.array([r["pred_best_accuracy"] for r in rows])
        gt = np.array([r["gt_best_accuracy"] for r in rows])
        m = _extended_metrics(pred, gt)
        m["low_n"] = m["n"] < MIN_N_FOR_DATASET_METRICS
        out[dataset] = m
    return out


def evaluate_checkpoint(
    model_path: Path,
    data_path: Path,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_new_tokens: int = TEST_MAX_NEW_TOKENS,
    limit: int | None = None,
    show_progress: bool = True,
    temperature: float = TEST_TEMPERATURE,
    predictions_csv: Path | None = None,
    show_raw: bool = False,
) -> dict:
    """
    Run greedy generation on test JSONL and compare predictions to ground truth.

    Returns a metric dict with RMSE/MAE for best_accuracy and best_epoch.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, tokenizer = _test_load_checkpoint(model_path, max_seq_len)
    FastLanguageModel.for_inference(model)

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    stop_token_ids = []
    for tok_str in ("}", "<|im_end|>", "<|endoftext|>"):
        ids = text_tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])
    if text_tokenizer.eos_token_id is not None:
        stop_token_ids.append(text_tokenizer.eos_token_id)
    stop_token_ids = list(set(stop_token_ids))

    raw_data = _test_load_data(data_path, limit)
    predictions: list[dict] = []
    parse_failures = 0

    iterator = tqdm(raw_data, desc="Inference", disable=not show_progress)
    for i, item in enumerate(iterator):
        messages = item.get("messages", [])
        gt = _test_get_ground_truth(messages)
        if gt is None:
            parse_failures += 1
            continue

        prompt_messages = _test_build_prompt_messages(messages)
        text = _apply_chat_template_for_inference(text_tokenizer, prompt_messages)
        input_ids = text_tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids_t)
        inputs = {"input_ids": input_ids_t, "attention_mask": attention_mask}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else None,
                use_cache=False,
            )

        generated = text_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        if show_raw:
            print(f"\n--- raw model completion [sample {i}] ---\n{generated}\n--- end ---\n")
        pred = _test_parse_assistant_json(generated)

        if pred is None:
            parse_failures += 1
            pred = {"best_accuracy": float("nan"), "best_epoch": float("nan")}

        pred_best = pred.get("best_accuracy")
        pred_prog = pred.get("best_epoch")
        if pred_best is None:
            pred_best = float("nan")
        if pred_prog is None:
            pred_prog = float("nan")
        try:
            pred_prog = float(pred_prog)
            if pred_prog < 1:
                pred_prog = float("nan")
        except (TypeError, ValueError):
            pred_prog = float("nan")

        gt_best = gt.get("best_accuracy", float("nan"))
        gt_prog = gt.get("best_epoch", float("nan"))
        try:
            gt_prog = float(gt_prog)
        except (TypeError, ValueError):
            gt_prog = float("nan")

        predictions.append(
            {
                "idx": i,
                "run_id": _test_extract_run_id(messages),
                "dataset": _test_extract_dataset(messages),
                "pred_best_accuracy": pred_best,
                "pred_best_epoch": pred_prog,
                "gt_best_accuracy": gt_best,
                "gt_best_epoch": gt_prog,
            }
        )

    valid = [
        p
        for p in predictions
        if not (np.isnan(p["pred_best_accuracy"]) or np.isnan(p["pred_best_epoch"]))
    ]

    metrics: dict = {
        "n_samples": len(raw_data),
        "n_rows_scored": len(predictions),
        "n_valid": len(valid),
        "parse_failures": parse_failures,
        "best_accuracy_rmse": None,
        "best_accuracy_mae": None,
        "best_accuracy_pearson_r": None,
        "best_accuracy_spearman_r": None,
        "best_accuracy_r2": None,
        "best_epoch_rmse": None,
        "best_epoch_mae": None,
        "best_epoch_pearson_r": None,
        "best_epoch_spearman_r": None,
        "best_epoch_r2": None,
        "per_dataset": {},
    }

    if valid:
        pred_best_arr = np.array([p["pred_best_accuracy"] for p in valid])
        pred_prog_arr = np.array([p["pred_best_epoch"] for p in valid], dtype=float)
        gt_best_arr = np.array([p["gt_best_accuracy"] for p in valid])
        gt_prog_arr = np.array([p["gt_best_epoch"] for p in valid], dtype=float)

        acc_m = _extended_metrics(pred_best_arr, gt_best_arr)
        epoch_m = _extended_metrics(pred_prog_arr, gt_prog_arr)

        metrics["best_accuracy_rmse"] = acc_m["rmse"]
        metrics["best_accuracy_mae"] = acc_m["mae"]
        metrics["best_accuracy_pearson_r"] = acc_m["pearson_r"]
        metrics["best_accuracy_spearman_r"] = acc_m["spearman_r"]
        metrics["best_accuracy_r2"] = acc_m["r2"]

        metrics["best_epoch_rmse"] = epoch_m["rmse"]
        metrics["best_epoch_mae"] = epoch_m["mae"]
        metrics["best_epoch_pearson_r"] = epoch_m["pearson_r"]
        metrics["best_epoch_spearman_r"] = epoch_m["spearman_r"]
        metrics["best_epoch_r2"] = epoch_m["r2"]

        metrics["per_dataset"] = _per_dataset_metrics(valid)

    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_csv, "w", encoding="utf-8") as handle:
            handle.write(
                "idx,run_id,dataset,pred_best_accuracy,pred_best_epoch,"
                "gt_best_accuracy,gt_best_epoch\n"
            )
            for p in predictions:
                handle.write(
                    f"{p['idx']},{p['run_id']},{p['dataset']},{p['pred_best_accuracy']},"
                    f"{p['pred_best_epoch']},{p['gt_best_accuracy']},{p['gt_best_epoch']}\n"
                )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def _test_write_metrics_log(
    metrics: dict,
    predictions_path: Path,
    metrics_path: Path = DEFAULT_TEST_METRICS_PATH,
) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write(f"test_run {predictions_path}\n")
        handle.write(f"total_samples {metrics['n_samples']}\n")
        handle.write(f"valid_predictions {metrics['n_valid']}\n")
        handle.write(f"parse_failures {metrics['parse_failures']}\n")
        for key, label in (
            ("best_accuracy_rmse", "best_accuracy_RMSE"),
            ("best_accuracy_mae", "best_accuracy_MAE"),
            ("best_accuracy_pearson_r", "best_accuracy_Pearson_r"),
            ("best_accuracy_spearman_r", "best_accuracy_Spearman_r"),
            ("best_accuracy_r2", "best_accuracy_R2"),
            ("best_epoch_rmse", "best_epoch_RMSE"),
            ("best_epoch_mae", "best_epoch_MAE"),
            ("best_epoch_pearson_r", "best_epoch_Pearson_r"),
            ("best_epoch_spearman_r", "best_epoch_Spearman_r"),
            ("best_epoch_r2", "best_epoch_R2"),
        ):
            value = metrics[key]
            if value is None:
                handle.write(f"{label} nan\n")
            else:
                handle.write(f"{label} {value:.4f}\n")

        per_dataset = metrics.get("per_dataset") or {}
        if per_dataset:
            handle.write("\nper_dataset_best_accuracy (dataset n pearson_r spearman_r r2 rmse mae [low_n]):\n")
            for dataset, m in per_dataset.items():
                flag = " low_n" if m.get("low_n") else ""
                fmt = lambda v: f"{v:.4f}" if v is not None else "nan"
                handle.write(
                    f"  {dataset} {m['n']} {fmt(m['pearson_r'])} {fmt(m['spearman_r'])} "
                    f"{fmt(m['r2'])} {fmt(m['rmse'])} {fmt(m['mae'])}{flag}\n"
                )
    return metrics_path


def test_model(
    model_path: Path = DEFAULT_OUTPUT_DIR,
    data_path: Path = DEFAULT_TEST_PATH,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_new_tokens: int = TEST_MAX_NEW_TOKENS,
    limit: int | None = None,
    output_path: Path = DEFAULT_TEST_OUTPUT_PATH,
    metrics_path: Path = DEFAULT_TEST_METRICS_PATH,
    show_raw: bool = False,
) -> dict:
    """Evaluate fine-tuned checkpoint on test JSONL; save predictions CSV and metrics log."""
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Checkpoint: {model_path}")
    print(f"Test data:  {data_path}")
    print(f"Limit:      {limit or 'all'}")

    metrics = evaluate_checkpoint(
        model_path,
        data_path,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        limit=limit,
        show_progress=True,
        predictions_csv=output_path,
        show_raw=show_raw,
    )

    print(f"Predictions saved to: {output_path}")
    print(f"Total samples:      {metrics['n_samples']}")
    print(f"Valid predictions:  {metrics['n_valid']}")
    print(f"Parse failures:     {metrics['parse_failures']}")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    if metrics["best_accuracy_rmse"] is None:
        print("No valid predictions to compute metrics (CSV saved anyway).")
    else:
        print("best_accuracy:")
        print(f"  RMSE:       {_fmt(metrics['best_accuracy_rmse'])}")
        print(f"  MAE:        {_fmt(metrics['best_accuracy_mae'])}")
        print(f"  Pearson r:  {_fmt(metrics['best_accuracy_pearson_r'])}")
        print(f"  Spearman r: {_fmt(metrics['best_accuracy_spearman_r'])}")
        print(f"  R2:         {_fmt(metrics['best_accuracy_r2'])}")
        print("best_epoch:")
        print(f"  RMSE:       {_fmt(metrics['best_epoch_rmse'])}")
        print(f"  MAE:        {_fmt(metrics['best_epoch_mae'])}")
        print(f"  Pearson r:  {_fmt(metrics['best_epoch_pearson_r'])}")
        print(f"  Spearman r: {_fmt(metrics['best_epoch_spearman_r'])}")
        print(f"  R2:         {_fmt(metrics['best_epoch_r2'])}")

        per_dataset = metrics.get("per_dataset") or {}
        if per_dataset:
            print("per-dataset best_accuracy (Pearson r):")
            for dataset, m in per_dataset.items():
                flag = " (low n)" if m.get("low_n") else ""
                print(f"  {dataset:<16} n={m['n']:<5} r={_fmt(m['pearson_r'])}{flag}")

    written_metrics_path = _test_write_metrics_log(metrics, output_path, metrics_path)
    print(f"Metrics logged to: {written_metrics_path}")
    return metrics


def main() -> None:
    print("=" * 80)
    print("Step 1/4: Data preprocessing (nn_dataset → JSONL)")
    print("=" * 80)
    data_preprocessing(
        output_jsonl_path=RAW_INPUT_PATH,
        output_csv_path=RAW_CSV_PATH,
    )

    print("\n" + "=" * 80)
    print("Step 2/4: Prepare LLM training datasets")
    print("=" * 80)
    train_path, val_path, test_path = prepare_llm_datasets()
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")

    print("\n" + "=" * 80)
    print("Step 3/4: Fine-tune model")
    print("=" * 80)
    train_model(train_path=train_path, val_path=val_path)
    print(f"Model saved to {DEFAULT_OUTPUT_DIR}")

    print("\n" + "=" * 80)
    print("Step 4/4: Test model on held-out test set")
    print("=" * 80)
    test_model(
        model_path=DEFAULT_OUTPUT_DIR,
        data_path=test_path,
        output_path=DEFAULT_TEST_OUTPUT_PATH,
    )
    print(f"\nDone. Pipeline complete.")


if __name__ == "__main__":
    main()


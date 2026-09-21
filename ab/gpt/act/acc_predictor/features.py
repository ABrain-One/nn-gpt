"""Prompt-feature construction: arch-family inference, per-block formatting, user message."""
from __future__ import annotations

import ast
import re
from typing import Optional

from . import config
from .common import _safe_float, _safe_int




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
        if config.USE_ZERO_COST_PROXIES else ""
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

def _format_layer_stats_lines(record: dict) -> list[str]:
    table = record.get("layer_stats")

    if not table:
        return []

    return [
        "LAYER_STATS_MEAN_BY_EPOCH",
        table,
        "",
    ]


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
    for name in config.ACTIVE_PROXY_NAMES:
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
    if not config.USE_NN_STATS:
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
    if not config.USE_SOURCE_CODE:
        return []
    return ["NEURAL_NETWORK_CODE", "```python", (nn_code or "").strip(), "```", ""]


def _build_user_message(record: dict, nn_code: str) -> str:
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    architecture_lines = ["- Architecture depth and complexity"] if config.USE_SOURCE_CODE else []
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
        *([f"epoch_3_accuracy: {round(_safe_float(record.get('accuracy_epoch_3'), 0.0), 6)}"] if config.USE_THIRD_EARLY_EPOCH else []),
        "",
        *_format_layer_stats_lines(record),
        *_format_proxy_lines(record),
        *_format_nn_stats_lines(record),
        *_format_code_lines(nn_code),
        *_format_similar_architectures_lines(record),
        "Analyze the training dynamics"
        + (", architecture complexity, and" if config.USE_SOURCE_CODE else " and")
        + " optimization hyperparameters to estimate the final training outcome.",
        "",
        "Important signals to consider:",
        "- Per-epoch layer-statistics averages, if present",
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
            if config.USE_CHAIN_OF_THOUGHT
            else "OUTPUT (JSON ONLY):"
        ),
    ]
    return "\n".join(lines)

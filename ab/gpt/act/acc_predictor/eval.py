"""Stage 4: evaluate a checkpoint on the held-out test set; metrics and predictions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from unsloth import FastLanguageModel

from .config import *  # noqa: F401,F403
from .common import _apply_chat_template_for_inference




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

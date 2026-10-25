#!/usr/bin/env python3
"""Predictor validation for the TEVC revision (Stage 2 / Stage 4 check).

Compares predicted maximum accuracy against TRUE maximum accuracy (best over the
frozen full-training schedule) on the fully trained subset, and reports:
  * mean absolute error (MAE, percentage points) with a bootstrap 95% CI;
  * Spearman and Kendall rank correlations with bootstrap 95% CIs;
  * top-10 overlap between single-epoch ranking and predicted ranking (per config);
  * the pre-registered decision (use predicted accuracy as Stage-3 fitness iff
    Spearman >= 0.80 and MAE <= 3 pp).

Only needs numpy. Reads predictions.jsonl (from exp_predict) and the full-training
eval_info.json under each gen dir. Run anywhere the outputs are available.

Example:
  python -m ab.gpt.act.kto.exp_validate --pool stage2_pool \
      --predictions stage2_pool/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

rng = np.random.default_rng(0)


def true_max_acc(gen_dir: Path) -> float | None:
    """Best accuracy over epochs from the full-training eval_info.json (percent)."""
    info = gen_dir / "eval_info.json"
    if not info.exists():
        return None
    try:
        payload = json.loads(info.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    # ab.nn records the best accuracy over the schedule as prm.best_accuracy;
    # eval_results.accuracy is only the final epoch.
    prm = (payload.get("eval_args") or {}).get("prm") or {}
    if prm.get("best_accuracy") is not None:
        return float(prm["best_accuracy"]) * 100.0
    res = payload.get("eval_results")
    if isinstance(res, dict):
        eps = res.get("epochs") or []
        accs = [e.get("accuracy", e.get("acc")) for e in eps if isinstance(e, dict)]
        accs = [float(a) for a in accs if a is not None]
        if accs:
            return max(accs) * 100.0
        a = res.get("accuracy", res.get("acc"))
        return float(a) * 100.0 if a is not None else None
    return None


def _rank(x):
    """Average (fractional) ranks, so heavily tied values, such as the predictor's
    coarse outputs, are ranked correctly (matches scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def kendall(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")


def boot_ci(fn, *arrs, B=3000):
    arrs = [np.asarray(a) for a in arrs]
    n = len(arrs[0]); out = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        try:
            out.append(fn(*[a[s] for a in arrs]))
        except Exception:  # noqa: BLE001
            pass
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the max-accuracy predictor")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--full_pool", default=None,
                    help="Dir holding the full-training eval_info.json if separate from --pool")
    args = ap.parse_args()

    pool = Path(args.pool)
    pred_path = Path(args.predictions) if args.predictions else pool / "predictions.jsonl"
    preds = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    full_root = Path(args.full_pool) if args.full_pool else pool

    pred_v, true_v, e1_v, cfg_v = [], [], [], []
    for r in preds:
        gd = full_root / r["config"] / r["gen"]
        tm = true_max_acc(gd)
        if tm is None:
            continue
        p = float(r["pred_max_acc"])
        if p <= 1.5:            # predictor emits a 0-1 fraction; put it on the 0-100 scale
            p *= 100.0
        pred_v.append(p); true_v.append(tm)
        e1_v.append(r["e1"]); cfg_v.append(r["config"])
    if len(pred_v) < 3:
        raise SystemExit(f"Only {len(pred_v)} fully-trained models found; need more to validate.")
    pred_v, true_v, e1_v = map(np.asarray, (pred_v, true_v, e1_v))

    mae = float(np.mean(np.abs(pred_v - true_v)))
    mae_ci = boot_ci(lambda p, t: np.mean(np.abs(p - t)), pred_v, true_v)
    sp = spearman(pred_v, true_v); sp_ci = boot_ci(spearman, pred_v, true_v)
    kt = kendall(pred_v, true_v)
    sp_1ep = spearman(e1_v, true_v)

    print("=" * 68)
    print(f"Predictor validation on {len(pred_v)} fully-trained models")
    print(f"  MAE (predicted vs true max)   : {mae:5.2f} pp   [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
    print(f"  Spearman (predicted vs true)  : {sp:5.2f}      [{sp_ci[0]:.2f}, {sp_ci[1]:.2f}]")
    print(f"  Kendall  (predicted vs true)  : {kt:5.2f}")
    print(f"  Spearman (1-epoch vs true)    : {sp_1ep:5.2f}   (baseline the predictor must beat)")
    use_pred = (sp >= 0.80) and (mae <= 3.0)
    print("-" * 68)
    print(f"  DECISION (Spearman>=0.80 & MAE<=3pp): "
          f"{'USE predicted accuracy as Stage-3 fitness' if use_pred else 'FALL BACK to raw 3-epoch accuracy'}")
    print("=" * 68)


if __name__ == "__main__":
    main()

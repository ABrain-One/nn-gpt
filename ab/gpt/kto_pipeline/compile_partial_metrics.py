#!/usr/bin/env python3
"""Compile card-style metrics from a (possibly still-running) generation benchmark.

NNEval writes a per-model eval_info.json the moment each architecture finishes, but
the aggregate metrics.json / all_cycles_results.json are only written at the end of a
cycle. This tool reads whatever per-model eval_info.json files exist so far under a
run folder and prints the card block — so you can summarize a live run without
waiting for or interrupting it.

Usage:
    python -m ab.gpt.kto_pipeline.compile_partial_metrics --run_dir out_2777004
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── stats (self-contained: no numpy / scipy / matplotlib, so it runs anywhere) ──
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
        22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
        120: 1.980}


def _t_crit(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T95:
        return _T95[df]
    if df > 120:
        return 1.960
    keys = [k for k in _T95 if k <= df]
    return _T95[max(keys)] if keys else 12.706


def mean_t_ci(values: List[float]):
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = (var ** 0.5) / (n ** 0.5)
    t = _t_crit(n - 1)
    return (mean, mean - t * se, mean + t * se)


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _acc_from_eval_info(payload: Dict[str, Any]) -> Optional[float]:
    res = payload.get("eval_results")
    a = None
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        a = res[1]
    elif isinstance(res, dict):
        a = res.get("accuracy", res.get("acc"))
        if a is None:
            eps = res.get("epochs", [])
            if eps and isinstance(eps[0], dict):
                a = eps[0].get("accuracy", eps[0].get("acc"))
    try:
        return float(a) if a is not None else None
    except (TypeError, ValueError):
        return None


def _read_code(model_dir: Path) -> Optional[str]:
    for name in ("new_nn.py", "new_nn.notnovel.py"):
        f = model_dir / name
        if f.exists():
            try:
                return f.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return None
    return None


def _count_generated(run_dir: Path) -> int:
    total = 0
    for rec in run_dir.rglob("generation_records.jsonl"):
        try:
            with open(rec, encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        except Exception:  # noqa: BLE001
            pass
    return total


def _structural_unique(codes: List[str]) -> Optional[int]:
    """Count structurally-distinct architectures via the repo's NoveltyChecker."""
    try:
        from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
    except Exception:  # noqa: BLE001
        return None
    try:
        nc = NoveltyChecker(None)  # fresh — only among these generations
        u = 0
        for c in codes:
            if nc.is_novel(c):
                u += 1
                nc.mark_as_seen(c)
        return u
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Compile partial card-style metrics from a live run")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run folder (e.g. out_2777004) or an experiment subdir")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Accuracy threshold for the '>=T%%' quality stat")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")
    thr = args.threshold

    accs: List[float] = []
    codes: List[str] = []
    n_processed = 0
    for info in run_dir.rglob("eval_info.json"):
        # keep only per-model files: .../nneval/<model>/eval_info.json
        if info.parent.parent.name != "nneval":
            continue
        n_processed += 1
        try:
            payload = json.loads(info.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a file being written right now
            continue
        a = _acc_from_eval_info(payload)
        if a is not None:
            accs.append(a)
            c = _read_code(info.parent)
            if c:
                codes.append(c)

    generated = _count_generated(run_dir)
    valid = len(accs)
    unique = _structural_unique(codes) if codes else None

    lines = [
        "=" * 72,
        f"PARTIAL metrics (live) — {run_dir}",
        f"  generated         : {generated}",
        f"  evaluated so far  : {n_processed}"
        + (f"  ({100.0 * n_processed / generated:.1f}% of generated)" if generated else ""),
    ]
    card = ""
    if valid:
        m, lo, hi = mean_t_ci(accs)
        ge = 100.0 * sum(1 for a in accs if a >= thr) / valid
        denom = generated or valid
        card = (f"Valid: {valid}/{denom} ({100.0 * valid / denom:.1f}%) · "
                f"Average (all valid): {m * 100:.2f}% [95% CI {lo * 100:.2f}-{hi * 100:.2f}] · "
                f"Median: {_median(accs) * 100:.2f}% · "
                f"Best: {max(accs) * 100:.2f}% · "
                f"≥{thr * 100:.0f}%: {ge:.2f}%")
        if unique is not None:
            card += f" · Unique: {unique}/{valid} ({100.0 * unique / valid:.1f}%)"
        lines.append("  " + card)
    else:
        lines.append("  (no evaluated models with an accuracy yet)")
    lines.append("=" * 72)

    out = "\n".join(lines)
    try:
        print(out)
    except UnicodeEncodeError:
        print(out.encode("ascii", "replace").decode())

    try:
        (run_dir / "partial_metrics_summary.txt").write_text(
            (card + "\n") if card else (out + "\n"), encoding="utf-8")
        print(f"  written: {run_dir / 'partial_metrics_summary.txt'}")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()

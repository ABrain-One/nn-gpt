#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402


def _parse_cycle(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Flatten one cycle record (from the aggregate OR a per-cycle metrics.json)."""
    if "bucketing" not in c:
        return None  # failed cycle with no metrics
    b = c.get("bucketing", {})
    bd = b.get("undesirable_breakdown", {})
    gen = int(c.get("generated", 0)) or 1
    new_des = int(b.get("new_desirable", 0))
    not_novel = int(b.get("not_novel_skipped", bd.get("not_novel", 0)))
    return {
        "cycle": int(c.get("cycle", 0)),
        "generated": gen,
        "evaluated": int(b.get("evaluated_accuracies", gen)),
        "best": float(b.get("best_accuracy", 0.0) or 0.0),
        "avg": float(b.get("avg_accuracy", 0.0) or 0.0),
        "effective_threshold": float(b.get("effective_threshold", 0.0) or 0.0),
        "new_desirable": new_des,
        "new_undesirable": int(b.get("new_undesirable", 0)),
        "not_novel": not_novel,
        "low_accuracy": int(bd.get("low_accuracy", 0)),
        "non_compiling": int(bd.get("non_compiling", 0)),
        "runtime_error": int(bd.get("runtime_error", 0)),
        "unparseable": int(bd.get("unparseable", 0)),
        "desirable_total": int(b.get("desirable_total", 0)),
        "undesirable_total": int(b.get("undesirable_total", 0)),
        "pass_acc": new_des + not_novel,  # cleared the accuracy bar (novel + duplicate)
        "trained": bool(c.get("training", {}).get("success", False)),
    }


def load_cycles(results_path: Path, cycles_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Build the per-cycle list, keyed by cycle number, from:
      1. the aggregate all_cycles_results.json (current session's cycles), then
      2. every cycle_<n>/metrics.json found under cycles_dir — these PERSIST across
         resumes, so they give the FULL history even when the aggregate is partial.
    """
    by_cycle: Dict[int, Dict[str, Any]] = {}

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for c in data.get("cycles", []):
            rec = _parse_cycle(c)
            if rec is not None:
                by_cycle[rec["cycle"]] = rec
    except Exception as e:  # noqa: BLE001
        print(f"[plot][warn] could not read aggregate {results_path}: {e}")

    if cycles_dir and cycles_dir.exists():
        for mfile in sorted(cycles_dir.glob("cycle_*/metrics.json")):
            try:
                rec = _parse_cycle(json.loads(mfile.read_text(encoding="utf-8")))
                if rec is not None:
                    by_cycle[rec["cycle"]] = rec  # authoritative per-cycle metrics
            except Exception:  # noqa: BLE001
                continue

    return [by_cycle[k] for k in sorted(by_cycle)]


def _acc_from_eval_info(payload: Dict[str, Any]) -> Optional[float]:
    """Pull accuracy from a per-model eval_info.json (eval_results may be tuple or dict)."""
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


def load_per_model_accuracies(cycles_dir: Optional[Path]) -> Dict[int, List[float]]:
    """{cycle: [per-model accuracies]} from cycle_<n>/nneval/*/eval_info.json (if present)."""
    accs: Dict[int, List[float]] = {}
    if not cycles_dir or not cycles_dir.exists():
        return accs
    for cycle_dir in sorted(cycles_dir.glob("cycle_*")):
        try:
            n = int(cycle_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        vals: List[float] = []
        for info in (cycle_dir / "nneval").glob("*/eval_info.json"):
            try:
                a = _acc_from_eval_info(json.loads(info.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                a = None
            if a is not None:
                vals.append(a)
        if vals:
            accs[n] = vals
    return accs


def apply_pass_average(cycles: List[Dict[str, Any]], per_model: Dict[int, List[float]],
                       run_threshold: float) -> None:
    """Redefine 'avg' as the mean over models that cleared the threshold (card-style).

    Uses per-model eval_info.json when available (corrects already-finished runs);
    otherwise leaves the stored avg untouched.
    """
    for r in cycles:
        accs = per_model.get(r["cycle"])
        if not accs:
            continue
        thr = r["effective_threshold"] or run_threshold
        passed = [a for a in accs if a >= thr]
        r["avg"] = (sum(passed) / len(passed)) if passed else float("nan")


def _line_plot(xs, ys, title, ylabel, label, color, path, ylim=None) -> Path:
    """One image, one plotted line (single legend entry)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, "o-", color=color, label=label)
    ax.set_title(title); ax.set_xlabel("Cycle"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_separate(cycles: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """One image per metric, each a single-line trend."""
    xs = [r["cycle"] for r in cycles]
    paths = [
        _line_plot(xs, [r["avg"] * 100 for r in cycles],
                   "Average Accuracy per Cycle (models ≥ threshold)", "Accuracy (%)",
                   "Average accuracy (≥ threshold)", "#ff7f0e",
                   out_dir / "kto_avg_accuracy.png"),
        _line_plot(xs, [r["best"] * 100 for r in cycles],
                   "Best Accuracy per Cycle", "Accuracy (%)", "Best accuracy",
                   "#1f77b4", out_dir / "kto_best_accuracy.png"),
        _line_plot(xs, [100 * r["evaluated"] / r["generated"] for r in cycles],
                   "Valid Models per Cycle (compiled+trained)", "% of generated",
                   "Valid (compiled+trained)", "#17becf",
                   out_dir / "kto_valid_trained.png", ylim=(0, 100)),
        _line_plot(xs, [100 * r["pass_acc"] / r["generated"] for r in cycles],
                   "Valid Models per Cycle (cleared threshold)", "% of generated",
                   "Cleared threshold", "#9467bd",
                   out_dir / "kto_valid_cleared_threshold.png", ylim=(0, 100)),
    ]

    # Bucket counts: desirable + undesirable (the two requested series).
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.4
    ax.bar([x - w / 2 for x in xs], [r["new_desirable"] for r in cycles], w,
           label="Desirable", color="#2ca02c")
    ax.bar([x + w / 2 for x in xs], [r["new_undesirable"] for r in cycles], w,
           label="Undesirable", color="#d62728", alpha=0.85)
    ax.set_title("Per-Cycle Bucketing Counts"); ax.set_xlabel("Cycle")
    ax.set_ylabel("Count"); ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=10)
    bpath = out_dir / "kto_bucket_counts.png"
    fig.tight_layout(); fig.savefig(bpath, dpi=130); plt.close(fig)
    paths.append(bpath)
    return paths


def save_csv(cycles: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "kto_cycle_summary.csv"
    cols = ["cycle", "generated", "evaluated", "best", "avg", "new_desirable",
            "new_undesirable", "low_accuracy", "not_novel", "desirable_total",
            "undesirable_total", "trained"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in cycles:
            w.writerow(r)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot KTO per-cycle analysis")
    p.add_argument("--results", type=str, default="all_cycles_results.json",
                   help="Path to the KTO all_cycles_results.json")
    p.add_argument("--out_dir", type=str, default="kto_plots")
    p.add_argument("--cycles_dir", type=str, default=None,
                   help="Optional dir with cycle_<n>/metrics.json for full history (default: results dir)")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect cycles_dir next to the results file if not given — per-cycle
    # metrics.json persist across resumes and give the full history.
    cycles_dir = Path(args.cycles_dir) if args.cycles_dir else results_path.parent
    cycles = load_cycles(results_path, cycles_dir)
    if not cycles:
        raise SystemExit("No cycles with metrics found in the results file.")

    # Average accuracy is reported over models that cleared the threshold (card-style).
    try:
        run_threshold = float(json.loads(results_path.read_text()).get("accuracy_threshold", 0.40))
    except Exception:  # noqa: BLE001
        run_threshold = 0.40
    per_model = load_per_model_accuracies(cycles_dir)
    if per_model:
        apply_pass_average(cycles, per_model, run_threshold)
    else:
        print("[plot][warn] no per-model eval_info.json found — 'avg' uses the stored "
              "metric (above-threshold only if produced by the updated pipeline).")

    figs = plot_separate(cycles, out_dir)
    csv_path = save_csv(cycles, out_dir)

    # Console summary (handy for the report).
    best_overall = max(cycles, key=lambda r: r["best"])
    print("=" * 64)
    print(f"KTO run: {len(cycles)} cycles ({cycles[0]['cycle']}..{cycles[-1]['cycle']})")
    print(f"  best accuracy this run : {best_overall['best']*100:.2f}%  (cycle {best_overall['cycle']})")
    print(f"  last-cycle avg accuracy: {cycles[-1]['avg']*100:.2f}%")
    print(f"  desirable accumulated  : {cycles[-1]['desirable_total']}")
    print("-" * 64)
    for f in figs:
        print(f"  figure : {f}")
    print(f"  csv    : {csv_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()

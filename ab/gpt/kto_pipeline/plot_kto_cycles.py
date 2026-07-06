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
        # Novelty read-out: among models that cleared the bar, how many were
        # structurally new vs duplicates of an already-accepted design.
        "novelty_rate": (new_des / (new_des + not_novel)) if (new_des + not_novel) else float("nan"),
        # Similarity-penalty telemetry (present only on sim-penalty runs).
        "sim_penalty_nonzero": int(b.get("sim_penalty_nonzero", 0)),
        "sim_penalty_mean": float(b.get("sim_penalty_mean", 0.0) or 0.0),
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


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n → (lo, hi) in [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# Two-sided 95% Student-t critical values by degrees of freedom (→ 1.96 for large df).
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
    """Sample mean and t-based 95% CI → (mean, lo, hi)."""
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
        r["pass_accs"] = passed  # sample for the t-based CI on the mean
        # Card-style stats over ALL valid (evaluated) models this cycle.
        r["valid"] = len(accs)
        r["avg_all"] = sum(accs) / len(accs)
        r["median"] = _median(accs)
        r["ge_threshold_pct"] = 100.0 * sum(1 for a in accs if a >= thr) / len(accs)


def _ci_plot(xs, ys, lo, hi, title, ylabel, label, color, path, ylim=None) -> Path:
    """One image, one metric, with 95% CI error bars per point."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, ecolor=color,
                capsize=3, label=label)
    ax.set_title(title); ax.set_xlabel("Cycle"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_separate(cycles: List[Dict[str, Any]], out_dir: Path,
                  run_threshold: float = 0.40) -> List[Path]:
    """Per-cycle plots: combined avg+best accuracy (with #evaluated), valid count,
    Wilson-CI rates, and bucketing counts."""
    xs = [r["cycle"] for r in cycles]

    # ── Combined accuracy: average (t-CI) + best, with #models-evaluated bars ──
    avg_y, avg_lo, avg_hi = [], [], []
    for r in cycles:
        accs = r.get("pass_accs") or []
        if len(accs) >= 2:
            m, lo, hi = mean_t_ci(accs)
            avg_y.append(m * 100); avg_lo.append((m - lo) * 100); avg_hi.append((hi - m) * 100)
        else:
            avg_y.append(r["avg"] * 100); avg_lo.append(0.0); avg_hi.append(0.0)
    best_y = [r["best"] * 100 for r in cycles]
    evaluated = [int(r["evaluated"]) for r in cycles]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax2 = ax.twinx()
    ax2.bar(xs, evaluated, width=0.6, color="#b0b0b0", alpha=0.35,
            label="Models evaluated", zorder=1)
    ax2.set_ylabel("Models evaluated / cycle")
    ax.errorbar(xs, avg_y, yerr=[avg_lo, avg_hi], fmt="o-", color="#ff7f0e",
                ecolor="#ff7f0e", capsize=3, label="Average (≥ threshold)", zorder=3)
    ax.plot(xs, best_y, "s-", color="#1f77b4", label="Best", zorder=3)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per Cycle — average (95% CI) & best, with #evaluated")
    ax.grid(True, alpha=0.3)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)  # lines over bars
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    acc_path = out_dir / "kto_accuracy.png"
    fig.tight_layout(); fig.savefig(acc_path, dpi=130); plt.close(fig)

    # ── Valid count: compiled + >=threshold models per cycle (novelty-agnostic) ──
    counts = [int(r["pass_acc"]) for r in cycles]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs, counts, width=0.6, color="#2ca02c",
           label=f"Valid (compiled + ≥{run_threshold*100:.0f}%)")
    for x, c in zip(xs, counts):
        ax.text(x, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Valid Models per Cycle (compiled + ≥{run_threshold*100:.0f}%, novelty-agnostic)")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=10)
    vcount_path = out_dir / "kto_valid_count.png"
    fig.tight_layout(); fig.savefig(vcount_path, dpi=130); plt.close(fig)

    # ── Wilson-CI rate plots (kept) ──
    vt_y, vt_lo, vt_hi, ct_y, ct_lo, ct_hi = [], [], [], [], [], []
    for r in cycles:
        n = r["generated"]
        for k, ys, los, his in ((r["evaluated"], vt_y, vt_lo, vt_hi),
                                (r["pass_acc"], ct_y, ct_lo, ct_hi)):
            p = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            ys.append(p * 100); los.append((p - lo) * 100); his.append((hi - p) * 100)

    paths = [
        acc_path,
        vcount_path,
        _ci_plot(xs, vt_y, vt_lo, vt_hi,
                 "Valid Generation Rate per Cycle (compiled+trained) — Wilson 95% CI",
                 "Valid generation rate (%)", "Valid (compiled+trained)", "#17becf",
                 out_dir / "kto_valid_trained.png", ylim=(0, 100)),
        _ci_plot(xs, ct_y, ct_lo, ct_hi,
                 "Cleared-Threshold Rate per Cycle — Wilson 95% CI",
                 "Cleared-threshold rate (%)", "Cleared threshold", "#9467bd",
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

    # ── Novelty per cycle: novel vs duplicate among threshold-clearing models ──
    # Direct read-out of whether the similarity penalty is pushing generation
    # toward structurally new architectures. Under the penalty, non-novel passers
    # still enter training, but a working penalty should slow their growth / lift
    # the novelty rate over cycles relative to the no-penalty baseline.
    novel = [int(r["new_desirable"]) for r in cycles]
    dup = [int(r["not_novel"]) for r in cycles]
    rate = [100.0 * n / (n + d) if (n + d) else float("nan")
            for n, d in zip(novel, dup)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(xs, novel, width=0.6, color="#2ca02c", label="Novel (unique)")
    ax.bar(xs, dup, width=0.6, bottom=novel, color="#ff7f0e", alpha=0.85,
           label="Duplicate (non-novel)")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Threshold-clearing models")
    ax.set_title("Novel vs Duplicate Architectures per Cycle (with novelty rate)")
    ax.grid(True, alpha=0.3, axis="y")
    ax2 = ax.twinx()
    ax2.plot(xs, rate, "o-", color="#1f77b4", label="Novelty rate")
    ax2.set_ylabel("Novelty rate (%)"); ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    npath = out_dir / "kto_novelty.png"
    fig.tight_layout(); fig.savefig(npath, dpi=130); plt.close(fig)
    paths.append(npath)
    return paths


def save_csv(cycles: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "kto_cycle_summary.csv"
    cols = ["cycle", "generated", "evaluated", "valid", "best", "avg", "avg_all",
            "median", "ge_threshold_pct", "new_desirable", "new_undesirable",
            "low_accuracy", "not_novel", "novelty_rate", "sim_penalty_nonzero",
            "sim_penalty_mean", "desirable_total", "undesirable_total", "trained"]
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

    figs = plot_separate(cycles, out_dir, run_threshold)
    csv_path = save_csv(cycles, out_dir)

    # ── Card-style summary (over ALL valid models, matching the HF model card) ──
    best_overall = max(cycles, key=lambda r: r["best"])
    generated_total = sum(int(r.get("generated", 0)) for r in cycles)
    all_valid = [a for accs in per_model.values() for a in accs]
    above = [a for r in cycles for a in (r.get("pass_accs") or [])]

    print("=" * 64)
    print(f"KTO run: {len(cycles)} cycles ({cycles[0]['cycle']}..{cycles[-1]['cycle']})")
    if all_valid:
        n_valid = len(all_valid)
        denom = generated_total or n_valid
        m, lo, hi = mean_t_ci(all_valid)
        ge = 100.0 * sum(1 for a in all_valid if a >= run_threshold) / n_valid
        # Novelty as a pure metric: structurally distinct vs duplicate generations.
        uniq = sum(int(r.get("new_desirable", 0)) for r in cycles)
        dup = sum(int(r.get("not_novel", 0)) for r in cycles)
        checked = uniq + dup
        uniq_str = (f" · Unique: {uniq}/{checked} ({100.0 * uniq / checked:.1f}%)"
                    if checked else "")
        card = (f"Valid: {n_valid}/{denom} ({100.0 * n_valid / denom:.1f}%) · "
                f"Average (all valid): {m*100:.2f}% [95% CI {lo*100:.2f}-{hi*100:.2f}] · "
                f"Median: {_median(all_valid)*100:.2f}% · "
                f"Best: {max(all_valid)*100:.2f}% · "
                f"≥{run_threshold*100:.0f}%: {ge:.2f}%" + uniq_str)
        try:
            print("  " + card)
        except UnicodeEncodeError:
            print("  " + card.encode("ascii", "replace").decode())
        try:
            (out_dir / "kto_card_summary.txt").write_text(card + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    else:
        print(f"  best accuracy this run : {best_overall['best']*100:.2f}%  "
              f"(cycle {best_overall['cycle']})")
        print("  [card block needs per-model eval_info.json — median/all-valid avg/>=thr skipped]")
    # Above-threshold average (the per-cycle plot's definition), kept for reference.
    if above:
        m, lo, hi = mean_t_ci(above)
        print(f"  Average (>= threshold) : {m*100:.2f}%  (95% CI {lo*100:.2f}-{hi*100:.2f}, n={len(above)})")
    print(f"  desirable accumulated  : {cycles[-1]['desirable_total']}")
    print("-" * 64)
    for f in figs:
        print(f"  figure : {f}")
    print(f"  csv    : {csv_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()

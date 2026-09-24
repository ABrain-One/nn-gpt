"""
Phase 1 analysis for the NNGPT token-budget sweep.

Walks every out_phaseA_{dataset}_{budget}/ directory and produces:
  phase1_output/phase1_master.csv   — one row per (dataset, budget, batch)
  phase1_output/phase1_stats.md     — per-budget summary table + key findings
  phase1_output/phase1_plots/       — overthinking-curve plots

Run from /shared/ssd/home/b-r-duvvuri/nngpt2/nn-gpt:
    python3 analyze_phase1.py [--out_dir OUT_DIR] [--no_plots]

Primary DVs (from Phase 0 reframe):
  - structural_validity_rate    (static checker, decoupled from infra)
  - code_fraction               (nn_chars / output_chars)
  - tokens_per_valid            (output_chars / n_structurally_valid)
  - eval_success_rate           (batches that reached LEMUR eval)
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- optional imports -------------------------------------------------------
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# Wire in static validity checker
sys.path.insert(0, str(Path(__file__).parent))
from ab.gpt.util.eval.static_validity import check_static_validity

# ---------------------------------------------------------------------------

BUDGETS = [512, 1024, 1536, 2048, 3072, 6144]
DATASETS = ["mnist", "cifar10", "cifar100"]
CHARS_PER_TOKEN = 4.2


# ---------------------------------------------------------------------------
# Error classification (matches Phase 0)
# ---------------------------------------------------------------------------

LEVEL_FAILED     = 0
LEVEL_BAD_CODE   = 1
LEVEL_INFRA_FAIL = 2
LEVEL_VALID      = 3
LEVEL_EVALUATED  = 4

LEVEL_NAMES = {0: "no_output", 1: "bad_code", 2: "infra_fail", 3: "duplicate", 4: "evaluated"}


def classify_error(error_txt_path: str):
    if not os.path.isfile(error_txt_path):
        return LEVEL_EVALUATED, "no_error"
    with open(error_txt_path, encoding="utf-8", errors="replace") as f:
        first = f.read(500)
    if "NN already exists" in first:
        return LEVEL_VALID, "duplicate"
    if "FileNotFoundError" in first:
        return LEVEL_INFRA_FAIL, "infra_fnf"
    if "lacks the required function" in first:
        return LEVEL_BAD_CODE, "missing_func"
    if "is not used in the code" in first:
        return LEVEL_BAD_CODE, "unused_param"
    if "SyntaxError" in first or "IndentationError" in first:
        return LEVEL_BAD_CODE, "syntax_error"
    if "AccuracyException" in first or "ValueError" in first:
        return LEVEL_BAD_CODE, "accuracy_error"
    return LEVEL_BAD_CODE, "other_error"


# ---------------------------------------------------------------------------
# Per-batch analysis
# ---------------------------------------------------------------------------

def analyze_batch(batch_dir: str, dataset: str, budget: int, epoch: int, batch_id: str) -> dict:
    nn_file    = os.path.join(batch_dir, "new_nn.py")
    out_file   = os.path.join(batch_dir, "full_output.txt")
    error_file = os.path.join(batch_dir, "error.txt")

    row = {
        "dataset":    dataset,
        "budget":     budget,
        "epoch":      epoch,
        "batch_id":   batch_id,
        "has_output": os.path.isfile(out_file),
        "has_nn":     os.path.isfile(nn_file),
        "output_chars":  0,
        "nn_chars":      0,
        "code_fraction": None,
        "est_tokens":    None,
        "validity_level":   LEVEL_FAILED,
        "error_type":       "no_output",
        "static_valid":     False,
        "static_score":     0.0,
        "static_reason":    "no_output",
    }

    if row["has_output"]:
        try:
            with open(out_file, encoding="utf-8", errors="replace") as f:
                txt = f.read()
            row["output_chars"] = len(txt)
            row["est_tokens"]   = len(txt) / CHARS_PER_TOKEN
        except Exception:
            pass

    if row["has_nn"]:
        try:
            with open(nn_file, encoding="utf-8", errors="replace") as f:
                code = f.read()
            row["nn_chars"] = len(code)
        except Exception:
            pass
        level, etype = classify_error(error_file)
        row["validity_level"] = level
        row["error_type"]     = etype

        sv = check_static_validity(nn_file)
        row["static_valid"]  = sv.valid
        row["static_score"]  = sv.score
        row["static_reason"] = sv.reason
    else:
        row["validity_level"] = LEVEL_FAILED
        row["error_type"]     = "no_output"

    if row["output_chars"] > 0 and row["nn_chars"] > 0:
        row["code_fraction"] = row["nn_chars"] / row["output_chars"]

    return row


# ---------------------------------------------------------------------------
# Walk out_phaseA_* directories
# ---------------------------------------------------------------------------

def collect_rows(base_dir: str) -> list[dict]:
    rows = []
    pattern = os.path.join(base_dir, "out_phaseA_*")
    sweep_dirs = sorted(glob.glob(pattern))

    if not sweep_dirs:
        print(f"[WARN] No out_phaseA_* directories found under {base_dir}")
        return rows

    for sweep_dir in sweep_dirs:
        name = os.path.basename(sweep_dir)
        # name pattern: out_phaseA_{dataset}_{budget}
        m = re.match(r"out_phaseA_(.+?)_(\d+)$", name)
        if not m:
            print(f"[SKIP] Unexpected dir name: {name}")
            continue
        dataset = m.group(1)
        budget  = int(m.group(2))

        batch_dirs = glob.glob(os.path.join(sweep_dir, "llm", "epoch", "A*", "synth_nn", "B*"))
        for batch_dir in sorted(batch_dirs):
            parts = Path(batch_dir).parts
            # ...epoch/A{n}/synth_nn/B{m}
            epoch_str = [p for p in parts if re.match(r"A\d+", p)]
            batch_str = [p for p in parts if re.match(r"B\d+", p)]
            epoch_num = int(epoch_str[0][1:]) if epoch_str else -1
            batch_id  = batch_str[0]          if batch_str else "?"
            rows.append(analyze_batch(batch_dir, dataset, budget, epoch_num, batch_id))

    return rows


# ---------------------------------------------------------------------------
# Per-(dataset, budget) summary
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (ds, budget), g in df.groupby(["dataset", "budget"]):
        n_total   = len(g)
        n_has_nn  = g["has_nn"].sum()
        n_sv      = g["static_valid"].sum()
        n_eval    = (g["validity_level"] >= LEVEL_INFRA_FAIL).sum()

        sv_rate   = n_sv / n_total if n_total else 0.0
        gen_rate  = n_has_nn / n_total if n_total else 0.0
        eval_rate = n_eval / n_total if n_total else 0.0

        out_chars = g.loc[g["output_chars"] > 0, "output_chars"]
        nn_chars  = g.loc[g["nn_chars"] > 0, "nn_chars"]
        cf        = g["code_fraction"].dropna()

        records.append({
            "dataset":              ds,
            "budget":               budget,
            "n_total":              n_total,
            "n_has_nn":             int(n_has_nn),
            "n_static_valid":       int(n_sv),
            "n_eval_attempted":     int(n_eval),
            "generation_rate":      round(gen_rate, 3),
            "static_validity_rate": round(sv_rate, 3),
            "eval_success_rate":    round(eval_rate, 3),
            "mean_output_chars":    round(out_chars.mean(), 0) if len(out_chars) else None,
            "mean_nn_chars":        round(nn_chars.mean(), 0) if len(nn_chars) else None,
            "mean_code_fraction":   round(cf.mean(), 3) if len(cf) else None,
            "tokens_per_valid":     round(out_chars.sum() / max(n_sv, 1) / CHARS_PER_TOKEN, 0),
        })
    return pd.DataFrame(records).sort_values(["dataset", "budget"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(summary: pd.DataFrame, plot_dir: str, no_plots: bool):
    if no_plots or not HAS_PLT:
        return
    os.makedirs(plot_dir, exist_ok=True)

    for ds in summary["dataset"].unique():
        sub = summary[summary["dataset"] == ds].sort_values("budget")
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Phase 1 budget sweep — {ds}", fontsize=13)

        axes[0].plot(sub["budget"], sub["static_validity_rate"], "o-", color="steelblue")
        axes[0].set_xlabel("Token budget (max_output_tokens)")
        axes[0].set_ylabel("Static validity rate")
        axes[0].set_title("Validity vs Budget")
        axes[0].set_xscale("log", base=2)

        axes[1].plot(sub["budget"], sub["mean_code_fraction"], "o-", color="darkorange")
        axes[1].set_xlabel("Token budget")
        axes[1].set_ylabel("Code fraction (nn_chars / output_chars)")
        axes[1].set_title("Code Fraction vs Budget")
        axes[1].set_xscale("log", base=2)

        axes[2].plot(sub["budget"], sub["tokens_per_valid"], "o-", color="green")
        axes[2].set_xlabel("Token budget")
        axes[2].set_ylabel("Tokens per valid NN")
        axes[2].set_title("Efficiency vs Budget")
        axes[2].set_xscale("log", base=2)

        plt.tight_layout()
        path = os.path.join(plot_dir, f"phase1_{ds}_sweep.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  saved {path}")

    # Cross-dataset: static validity vs budget
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in summary["dataset"].unique():
        sub = summary[summary["dataset"] == ds].sort_values("budget")
        ax.plot(sub["budget"], sub["static_validity_rate"], "o-", label=ds)
    ax.set_xlabel("Token budget (max_output_tokens)")
    ax.set_ylabel("Static validity rate")
    ax.set_title("Phase 1: Structural validity across datasets")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.axvline(6144, ls="--", color="gray", alpha=0.6, label="Expt 109 budget")
    plt.tight_layout()
    path = os.path.join(plot_dir, "phase1_all_datasets_validity.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Stats report
# ---------------------------------------------------------------------------

def write_stats(summary: pd.DataFrame, master: pd.DataFrame, out_path: str):
    lines = [
        "# Phase 1 — Token Budget Sweep Analysis",
        "",
        "## Primary hypothesis",
        "Forcing a smaller token budget squeezes out prose preamble, raising code fraction",
        "and structural validity rate (the real signal from Phase 0).",
        "",
        "## Summary table (per dataset × budget)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Key metrics",
    ]

    for ds in summary["dataset"].unique():
        sub = summary[summary["dataset"] == ds].sort_values("budget")
        if sub.empty:
            continue
        best_sv = sub.loc[sub["static_validity_rate"].idxmax()]
        lines += [
            f"",
            f"### {ds}",
            f"- Best static_validity_rate: {best_sv['static_validity_rate']:.1%} at budget={int(best_sv['budget'])}",
            f"- Best mean_code_fraction:   {sub['mean_code_fraction'].max():.3f}",
            f"- Lowest tokens_per_valid:   {sub['tokens_per_valid'].min():.0f}",
        ]

        if HAS_SCIPY and len(sub) >= 3:
            cf_vals = sub["mean_code_fraction"].dropna()
            sv_vals = sub.loc[cf_vals.index, "static_validity_rate"]
            if len(cf_vals) >= 3:
                r, p = stats.pearsonr(sub["budget"].iloc[:len(cf_vals)], cf_vals)
                lines.append(f"- budget vs code_fraction Pearson r={r:.3f} p={p:.3f}")

    lines += [
        "",
        "## Phase 0 baseline (Expt 109, budget=6144)",
        "- A0 static validity rate: 35/52 = 67.3% (after fixing static_validity.py)",
        "- A0 mean code fraction: ~0.123 (3,070 / 25,023 chars)",
        "- Tokens per valid (est): ~5,958 chars / 4.2 = ~1,418 tokens",
        "",
        "---",
        "_Generated by analyze_phase1.py_",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".", help="Directory containing out_phaseA_* dirs")
    ap.add_argument("--out_dir", default="phase1_output", help="Output directory")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "phase1_plots")

    print("Collecting batch rows...")
    rows = collect_rows(args.base_dir)
    if not rows:
        print("No data found. Have the Phase 1 jobs completed?")
        sys.exit(0)

    master = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "phase1_master.csv")
    master.to_csv(csv_path, index=False)
    print(f"  {len(master)} rows → {csv_path}")

    print("Computing per-(dataset, budget) summary...")
    summary = summarize(master)
    summary_path = os.path.join(out_dir, "phase1_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  {len(summary)} rows → {summary_path}")

    print("Writing stats report...")
    stats_path = os.path.join(out_dir, "phase1_stats.md")
    write_stats(summary, master, stats_path)

    print("Making plots...")
    make_plots(summary, plot_dir, args.no_plots)

    print("\nDone.")
    print(summary[["dataset", "budget", "n_total", "static_validity_rate",
                   "mean_code_fraction", "tokens_per_valid"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Phase 0 analysis for the NNGPT token-efficiency study.

Walks every out_109_* batch directory and produces:
  phase0_master.csv   — one row per batch
  phase0_stats.md     — correlations + key findings
  phase0_plots/       — three correlation scatter plots

Run from the repo root (nn-gpt/):
    python3 ab/gpt/util/analize/analyze_phase0.py [--out_dir OUT_DIR] [--no_plots]

Without torch/fvcore the script still runs; nn_params and nn_flops columns
are populated only when executed inside the training container.
"""

import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = [
    "mnist", "svhn", "cifar10", "cifar100",
    "celeba_gender", "imagenette", "places365",
]

# Dataset names as they appear in dir names vs LEMUR dataset= key
DS_ALIAS = {
    "cifar10": "cifar-10",
    "cifar100": "cifar-100",
    "celeba_gender": "celeba-gender",
}

# Chars per token — empirical estimate for mixed code+prose (OlympicCoder-7B)
CHARS_PER_TOKEN = 4.2

# Validity levels
LEVEL_FAILED = 0       # no new_nn.py
LEVEL_BAD_CODE = 1     # new_nn.py exists but structural error (missing func, NameError, etc.)
LEVEL_INFRA_FAIL = 2   # new_nn.py exists, infra error (FileNotFoundError / stale tmp)
LEVEL_VALID = 3        # new_nn.py exists, "NN already exists" (code correct, in LEMUR)
LEVEL_EVALUATED = 4    # new_nn.py exists, no error or AccuracyException (full eval attempted)


def classify_error(error_txt_path: str) -> tuple[int, str]:
    if not os.path.isfile(error_txt_path):
        return LEVEL_EVALUATED, "no_error"
    with open(error_txt_path, encoding="utf-8", errors="replace") as f:
        first = f.read(500)
    if "NN already exists" in first:
        return LEVEL_VALID, "duplicate"
    if "FileNotFoundError" in first:
        return LEVEL_INFRA_FAIL, "infra_fnf"
    if "Code is missing" in first:
        return LEVEL_BAD_CODE, "code_missing"
    if "lacks the required function" in first:
        return LEVEL_BAD_CODE, "missing_func"
    if "not used in the code" in first:
        return LEVEL_BAD_CODE, "unused_param"
    if "NameError" in first:
        return LEVEL_BAD_CODE, "name_error"
    if "TypeError" in first:
        return LEVEL_BAD_CODE, "type_error"
    if "AccuracyException" in first:
        return LEVEL_EVALUATED, "low_accuracy"
    if "LearnTimeException" in first or "time limit" in first.lower():
        return LEVEL_BAD_CODE, "time_limit"
    return LEVEL_BAD_CODE, "other_error"


def extract_gen_accuracy(error_txt_path: str) -> Optional[float]:
    """Pull accuracy from AccuracyException line if present."""
    if not os.path.isfile(error_txt_path):
        return None
    with open(error_txt_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"AccuracyException.*?\((\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    return None


def read_src_info(batch_dir: str) -> dict:
    """Read source NN info from dataframe.df (pandas Series)."""
    df_path = os.path.join(batch_dir, "dataframe.df")
    if not os.path.isfile(df_path):
        # Fall back to original_*.py file size
        originals = glob.glob(os.path.join(batch_dir, "original_*.py"))
        if originals:
            src_chars = os.path.getsize(originals[0])
            src_name = os.path.basename(originals[0]).replace("original_", "").rsplit(".", 1)[0]
            return {"src_chars": src_chars, "src_nn": src_name, "src_acc": None}
        return {"src_chars": None, "src_nn": None, "src_acc": None}
    try:
        with open(df_path, "rb") as f:
            row = pickle.load(f)
        if isinstance(row, pd.Series):
            src_code = row.get("nn_code", "") or ""
            return {
                "src_chars": len(src_code),
                "src_nn": row.get("nn", None),
                "src_acc": row.get("accuracy", None),
            }
        elif isinstance(row, pd.DataFrame):
            r = row.iloc[0]
            src_code = r.get("nn_code", "") or ""
            return {
                "src_chars": len(src_code),
                "src_nn": r.get("nn", None),
                "src_acc": r.get("accuracy", None),
            }
    except Exception:
        pass
    # Last resort: size of original_*.py
    originals = glob.glob(os.path.join(batch_dir, "original_*.py"))
    if originals:
        return {
            "src_chars": os.path.getsize(originals[0]),
            "src_nn": os.path.basename(originals[0]).replace("original_", "").rsplit(".", 1)[0],
            "src_acc": None,
        }
    return {"src_chars": None, "src_nn": None, "src_acc": None}


def count_chars_in_tags(text: str, open_tag: str, close_tag: str) -> int:
    total = 0
    start = 0
    while True:
        a = text.find(open_tag, start)
        if a == -1:
            break
        b = text.find(close_tag, a + len(open_tag))
        if b == -1:
            total += len(text) - a - len(open_tag)
            break
        total += b - a - len(open_tag)
        start = b + len(close_tag)
    return total


def analyze_output_file(path: str) -> dict:
    if not os.path.isfile(path):
        return {
            "output_chars": 0, "output_lines": 0,
            "code_chars": 0, "prose_chars": 0,
            "has_hp_tag": False, "has_nn_tag": False,
        }
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    output_chars = len(text)
    output_lines = text.count("\n")
    code_chars = (
        count_chars_in_tags(text, "<hp>", "</hp>")
        + count_chars_in_tags(text, "<tr>", "</tr>")
        + count_chars_in_tags(text, "<nn>", "</nn>")
    )
    prose_chars = output_chars - code_chars
    return {
        "output_chars": output_chars,
        "output_lines": output_lines,
        "code_chars": code_chars,
        "prose_chars": prose_chars,
        "has_hp_tag": "<hp>" in text,
        "has_nn_tag": "<nn>" in text,
    }


def get_nn_cost_safe(nn_file: str, dataset: str) -> tuple:
    try:
        from ab.gpt.util.eval.nn_cost import get_nn_cost
        return get_nn_cost(nn_file, dataset)
    except Exception:
        return None, None


def walk_dataset(ds_dir: str, dataset: str) -> list[dict]:
    rows = []
    epoch_dirs = sorted(glob.glob(os.path.join(ds_dir, "llm", "epoch", "A*")))
    for epoch_dir in epoch_dirs:
        epoch = os.path.basename(epoch_dir)  # e.g. A0
        batch_dirs = sorted(glob.glob(os.path.join(epoch_dir, "synth_nn", "B*")),
                            key=lambda p: int(os.path.basename(p)[1:]))
        for batch_dir in batch_dirs:
            batch = os.path.basename(batch_dir)

            output_path = os.path.join(batch_dir, "full_output.txt")
            out_info = analyze_output_file(output_path)

            new_nn_path = os.path.join(batch_dir, "new_nn.py")
            nn_chars = os.path.getsize(new_nn_path) if os.path.isfile(new_nn_path) else 0
            gen_valid = nn_chars > 0

            error_path = os.path.join(batch_dir, "error.txt")
            if gen_valid:
                validity_level, error_type = classify_error(error_path)
            else:
                validity_level, error_type = LEVEL_FAILED, "no_generation"

            gen_acc = extract_gen_accuracy(error_path) if gen_valid else None

            src = read_src_info(batch_dir)

            # Verbosity ratio: full output vs generated code
            verbosity_ratio = (
                out_info["output_chars"] / nn_chars if nn_chars > 0 else None
            )

            # Token estimates
            output_tokens_est = out_info["output_chars"] / CHARS_PER_TOKEN

            # Try to get nn_params / nn_flops if torch available
            nn_params, nn_flops = (None, None)
            if gen_valid:
                nn_params, nn_flops = get_nn_cost_safe(new_nn_path, dataset)

            src_params, src_flops = (None, None)
            originals = glob.glob(os.path.join(batch_dir, "original_*.py"))
            if originals:
                src_params, src_flops = get_nn_cost_safe(originals[0], dataset)

            rows.append({
                "dataset": dataset,
                "epoch": epoch,
                "batch": batch,
                "output_chars": out_info["output_chars"],
                "output_lines": out_info["output_lines"],
                "output_tokens_est": round(output_tokens_est, 1),
                "code_chars_in_output": out_info["code_chars"],
                "prose_chars_in_output": out_info["prose_chars"],
                "has_hp_tag": out_info["has_hp_tag"],
                "has_nn_tag": out_info["has_nn_tag"],
                "nn_chars": nn_chars,
                "gen_valid": gen_valid,
                "validity_level": validity_level,
                "error_type": error_type,
                "verbosity_ratio": verbosity_ratio,
                "gen_acc": gen_acc,
                "src_nn": src["src_nn"],
                "src_chars": src["src_chars"],
                "src_acc": src["src_acc"],
                "nn_params": nn_params,
                "nn_flops": nn_flops,
                "src_params": src_params,
                "src_flops": src_flops,
            })
    return rows


def pearson_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Pearson r with Fisher-z 95% CI. Returns dict."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return {"r": None, "ci_lo": None, "ci_hi": None, "p": None, "n": n}
    r, p = stats.pearsonr(x, y)
    # Fisher z-transform
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = float(np.tanh(z - z_crit * se))
    ci_hi = float(np.tanh(z + z_crit * se))
    return {"r": float(r), "ci_lo": ci_lo, "ci_hi": ci_hi, "p": float(p), "n": int(n)}


def biserial_with_ci(x: np.ndarray, y_binary: np.ndarray) -> dict:
    """Point-biserial correlation for continuous x vs binary y."""
    mask = np.isfinite(x)
    x, y = x[mask], y_binary[mask]
    n = len(x)
    if n < 4:
        return {"r": None, "ci_lo": None, "ci_hi": None, "p": None, "n": n}
    r, p = stats.pointbiserialr(y, x)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.975)
    ci_lo = float(np.tanh(z - z_crit * se))
    ci_hi = float(np.tanh(z + z_crit * se))
    return {"r": float(r), "ci_lo": ci_lo, "ci_hi": ci_hi, "p": float(p), "n": int(n)}


def make_plots(df: pd.DataFrame, plot_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # (a) output_tokens vs validity (binary)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {True: "#2ecc71", False: "#e74c3c"}
    for v in [False, True]:
        sub = df[df["gen_valid"] == v]
        ax.scatter(sub["output_tokens_est"], [int(v)] * len(sub),
                   c=colors[v], alpha=0.4, s=30,
                   label="Valid generation" if v else "Failed generation")
    ax.set_xlabel("Output tokens (estimated)")
    ax.set_ylabel("Generation succeeded")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Failed", "Valid"])
    ax.legend()
    ax.set_title("Output token count vs generation validity")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "tokens_vs_validity.png"), dpi=150)
    plt.close(fig)

    # (b) verbosity ratio vs validity
    sub = df[df["verbosity_ratio"].notna()]
    if len(sub) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        for v in [False, True]:
            s = sub[sub["gen_valid"] == v]
            ax.scatter(s["verbosity_ratio"], [int(v)] * len(s),
                       c=colors[v], alpha=0.5, s=30,
                       label="Valid" if v else "Failed")
        ax.set_xlabel("Verbosity ratio (output_chars / nn_chars)")
        ax.set_ylabel("Generation succeeded")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Failed", "Valid"])
        ax.legend()
        ax.set_title("Verbosity ratio vs generation validity")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "verbosity_vs_validity.png"), dpi=150)
        plt.close(fig)

    # (c) per-epoch validity rate
    epoch_order = sorted(df["epoch"].unique())
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        rates = [sub[sub["epoch"] == e]["gen_valid"].mean() for e in epoch_order]
        ax.bar(epoch_order, rates, color="#3498db", alpha=0.8)
        ax.set_title(ds, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Generation rate" if ax == axes[0] else "")
    fig.suptitle("Generation validity rate by epoch and dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "validity_rate_by_epoch.png"), dpi=150)
    plt.close(fig)

    print(f"[PLOTS] Saved 3 plots to {plot_dir}/")


def write_stats_md(df: pd.DataFrame, corrs: dict, out_path: str):
    lines = [
        "# Phase 0 Statistical Summary",
        "",
        f"**Batches analysed:** {len(df)}  ",
        f"**Datasets:** {', '.join(sorted(df['dataset'].unique()))}  ",
        f"**Epochs:** {', '.join(sorted(df['epoch'].unique()))}  ",
        "",
        "## Generation Validity Rate",
        "",
        "| Dataset | Epoch | Total | Gen Valid | Valid% |",
        "|---------|-------|-------|-----------|--------|",
    ]
    for ds in sorted(df["dataset"].unique()):
        for ep in sorted(df["epoch"].unique()):
            sub = df[(df["dataset"] == ds) & (df["epoch"] == ep)]
            tot = len(sub)
            val = sub["gen_valid"].sum()
            lines.append(f"| {ds} | {ep} | {tot} | {val} | {val/tot*100:.0f}% |")

    lines += [
        "",
        "## Error Type Breakdown",
        "",
        "| Error Type | Count | % |",
        "|------------|-------|---|",
    ]
    ec = df["error_type"].value_counts()
    total_batches = len(df)
    for err_type, cnt in ec.items():
        lines.append(f"| {err_type} | {cnt} | {cnt/total_batches*100:.1f}% |")

    lines += [
        "",
        "## Output Size Statistics by Epoch",
        "",
        "| Epoch | Mean output_chars | Median | Std | Mean output_tokens_est |",
        "|-------|-------------------|--------|-----|------------------------|",
    ]
    for ep in sorted(df["epoch"].unique()):
        sub = df[df["epoch"] == ep]["output_chars"]
        tok = df[df["epoch"] == ep]["output_tokens_est"]
        lines.append(
            f"| {ep} | {sub.mean():.0f} | {sub.median():.0f} | {sub.std():.0f} | {tok.mean():.0f} |"
        )

    lines += [
        "",
        "## Correlations with 95% CI",
        "",
        "### (b) output_tokens vs validity (point-biserial r)",
        "",
        "| Dataset | r | 95% CI | p | n |",
        "|---------|---|--------|---|---|",
    ]
    for ds, c in corrs.get("tokens_vs_validity_by_ds", {}).items():
        if c["r"] is not None:
            lines.append(
                f"| {ds} | {c['r']:.3f} | [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] | {c['p']:.4f} | {c['n']} |"
            )

    overall = corrs.get("tokens_vs_validity_overall", {})
    if overall.get("r") is not None:
        lines += [
            "",
            f"**Overall (all datasets pooled):** r = {overall['r']:.3f}, "
            f"95% CI [{overall['ci_lo']:.3f}, {overall['ci_hi']:.3f}], p = {overall['p']:.4f}, n = {overall['n']}",
        ]

    lines += [
        "",
        "### (c) verbosity_ratio vs validity (point-biserial r)",
        "",
    ]
    vratio = corrs.get("verbosity_vs_validity_overall", {})
    if vratio.get("r") is not None:
        lines.append(
            f"**Overall:** r = {vratio['r']:.3f}, "
            f"95% CI [{vratio['ci_lo']:.3f}, {vratio['ci_hi']:.3f}], p = {vratio['p']:.4f}, n = {vratio['n']}"
        )

    lines += [
        "",
        "### src_chars vs nn_chars (Pearson r, valid batches only)",
        "",
    ]
    src_nn = corrs.get("src_chars_vs_nn_chars", {})
    if src_nn.get("r") is not None:
        lines.append(
            f"**r = {src_nn['r']:.3f}**, 95% CI [{src_nn['ci_lo']:.3f}, {src_nn['ci_hi']:.3f}], "
            f"p = {src_nn['p']:.4f}, n = {src_nn['n']}"
        )

    lines += [
        "",
        "## Key Findings",
        "",
        "1. **Generation rate** = fraction of batches where a new_nn.py was extracted.",
        "2. **Validity level 3** (LEMUR duplicate) = code is correct; accuracy already in LEMUR DB.",
        "3. **tokens_vs_validity** correlation: "
        + (f"r = {overall['r']:.3f}" if overall.get("r") else "insufficient data")
        + " — positive r means longer outputs correlate with higher generation success.",
        "4. **nn_params / nn_flops**: populated only when run inside container (torch required).",
        "",
        "_Generated by analyze_phase0.py_",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[MD] Stats written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--out_dir", default="phase0_output")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    base = args.base_dir
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for ds in DATASETS:
        ds_dir = os.path.join(base, f"out_109_{ds}")
        if not os.path.isdir(ds_dir):
            print(f"[SKIP] {ds_dir} not found")
            continue
        print(f"[SCAN] {ds} ...")
        lemur_dataset = DS_ALIAS.get(ds, ds)
        rows = walk_dataset(ds_dir, lemur_dataset)
        all_rows.extend(rows)
        print(f"  → {len(rows)} batches")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "phase0_master.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Master CSV: {csv_path}  ({len(df)} rows)")

    # Correlations
    corrs = {}

    # tokens vs validity (binary), per dataset
    tv_by_ds = {}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        tv_by_ds[ds] = biserial_with_ci(
            sub["output_tokens_est"].values.astype(float),
            sub["gen_valid"].values.astype(float),
        )
    corrs["tokens_vs_validity_by_ds"] = tv_by_ds

    corrs["tokens_vs_validity_overall"] = biserial_with_ci(
        df["output_tokens_est"].values.astype(float),
        df["gen_valid"].values.astype(float),
    )

    corrs["verbosity_vs_validity_overall"] = biserial_with_ci(
        df["verbosity_ratio"].fillna(np.nan).values.astype(float),
        df["gen_valid"].values.astype(float),
    )

    valid_df = df[df["gen_valid"] & df["src_chars"].notna()]
    corrs["src_chars_vs_nn_chars"] = pearson_with_ci(
        valid_df["src_chars"].values.astype(float),
        valid_df["nn_chars"].values.astype(float),
    )

    md_path = os.path.join(out_dir, "phase0_stats.md")
    write_stats_md(df, corrs, md_path)

    if not args.no_plots:
        make_plots(df, os.path.join(out_dir, "phase0_plots"))

    # Print summary table
    print("\n" + "=" * 72)
    print("PHASE 0 SUMMARY")
    print("=" * 72)
    print(f"{'Dataset':<16} {'Epoch':<6} {'Total':<6} {'GenValid':<9} {'Rate%':<7} {'AvgOutChars':<13} {'AvgVerbosity'}")
    print("-" * 72)
    for ds in sorted(df["dataset"].unique()):
        for ep in sorted(df["epoch"].unique()):
            sub = df[(df["dataset"] == ds) & (df["epoch"] == ep)]
            tot = len(sub)
            val = sub["gen_valid"].sum()
            avg_chars = sub["output_chars"].mean()
            avg_verb = sub[sub["verbosity_ratio"].notna()]["verbosity_ratio"].mean()
            avg_verb_s = f"{avg_verb:.1f}x" if not np.isnan(avg_verb) else "—"
            print(f"{ds:<16} {ep:<6} {tot:<6} {val:<9} {val/tot*100:<7.0f} {avg_chars:<13.0f} {avg_verb_s}")

    print("\nCorrelations (overall):")
    ov = corrs["tokens_vs_validity_overall"]
    if ov["r"] is not None:
        print(f"  output_tokens vs validity:   r={ov['r']:+.3f}  95%CI [{ov['ci_lo']:+.3f},{ov['ci_hi']:+.3f}]  p={ov['p']:.4f}  n={ov['n']}")
    vv = corrs["verbosity_vs_validity_overall"]
    if vv["r"] is not None:
        print(f"  verbosity_ratio vs validity: r={vv['r']:+.3f}  95%CI [{vv['ci_lo']:+.3f},{vv['ci_hi']:+.3f}]  p={vv['p']:.4f}  n={vv['n']}")
    sc = corrs["src_chars_vs_nn_chars"]
    if sc["r"] is not None:
        print(f"  src_chars vs nn_chars:       r={sc['r']:+.3f}  95%CI [{sc['ci_lo']:+.3f},{sc['ci_hi']:+.3f}]  p={sc['p']:.4f}  n={sc['n']}")

    print(f"\nOutputs: {out_dir}/")
    print("  phase0_master.csv  — full row-per-batch table")
    print("  phase0_stats.md    — this summary in markdown")
    print("  phase0_plots/      — 3 scatter/bar plots")
    print()
    print("NOTE: nn_params and nn_flops columns are None until run inside the")
    print("      training container where torch + fvcore/ptflops are available.")


if __name__ == "__main__":
    main()

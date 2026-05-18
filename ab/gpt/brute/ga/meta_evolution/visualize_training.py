"""
visualize_training.py
---------------------
Generates and saves plots for:
  1. GA evolution progress  (from stats/ JSON files)
  2. LLM fine-tuning progress (from LLM-evolution-logs.jsonl)

Output is saved into a timestamped folder:
  meta_evolution/visualizations/run_<YYYY-MM-DD_HH-MM-SS>/
      ga_evolution/
          generation_accuracy.png
          population_diversity.png
          best_vs_avg_accuracy.png
      fine_tuning/
          reward_over_iterations.png
          syntax_success_rate.png
          score_improvement.png

Usage:
    python3 visualize_training.py
"""

import os
import json
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
STATS_DIR      = os.path.join(BASE_DIR, "stats")
JSONL_LOG      = os.path.join(BASE_DIR, "LLM-evolution-logs.jsonl")
VIZ_ROOT       = os.path.join(BASE_DIR, "visualizations")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3f5c",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#b0b0b0",
    "ytick.color":      "#b0b0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "legend.facecolor": "#1a1d2e",
    "legend.edgecolor": "#3a3f5c",
}

ACCENT1 = "#7c83fd"   # blue-purple
ACCENT2 = "#fd7c83"   # coral
ACCENT3 = "#7cfd83"   # green
BAR_COLOR = "#3a4a8a"


def _apply_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(colors="#b0b0b0")


def _save(fig, path, saved_files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(path)
    print(f"  [saved] {os.path.relpath(path, BASE_DIR)}")


def _warn(msg):
    print(f"  [WARN]  {msg}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats_records():
    """
    Scan stats/ for model JSON files.
    Returns a list of dicts sorted by folder mtime (proxy for eval order).
    Each dict contains: accuracy, best_accuracy, train_accuracy, train_loss,
                        test_loss, gradient_norm, samples_per_second, uid, mtime
    """
    records = []
    if not os.path.isdir(STATS_DIR):
        _warn(f"stats/ directory not found at {STATS_DIR}")
        return records
    for folder in sorted(os.listdir(STATS_DIR)):
            folder_path = os.path.join(STATS_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            mtime = os.path.getmtime(folder_path)
            # Read the most recent epoch JSON (highest numbered file)
            json_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".json")],
                key=lambda x: int(x.replace(".json", "")) if x.replace(".json", "").isdigit() else 0
            )
            if not json_files:
                continue
            json_path = os.path.join(folder_path, json_files[-1])
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception as e:
                _warn(f"Could not read {json_path}: {e}")
                continue

            hp = data.get("hyperparameters", {})
            ts = data.get("training_summary", {})

            def _get(*keys):
                """Try data → hyperparameters → training_summary for each key."""
                for k in keys:
                    for src in [data, hp, ts]:
                        if k in src and src[k] is not None:
                            try:
                                return float(src[k])
                            except (TypeError, ValueError):
                                pass
                return None

            rec = {
                "uid":               data.get("uid", folder),
                "mtime":             mtime,
                "accuracy":          _get("accuracy"),
                "best_accuracy":     _get("best_accuracy"),
                "train_accuracy":    _get("train_accuracy"),
                "train_loss":        _get("train_loss"),
                "test_loss":         _get("test_loss"),
                "gradient_norm":     _get("gradient_norm"),
                "samples_per_second":_get("samples_per_second"),
            }
            records.append(rec)

    # Sort by folder mtime → chronological eval order
    records.sort(key=lambda r: r["mtime"])
    return records


def load_llm_logs():
    """
    Read LLM-evolution-logs.jsonl. Returns list of dicts.
    Expected fields: method, score, reward, valid_syntax, timestamp.
    """
    if not os.path.exists(JSONL_LOG):
        return []
    entries = []
    with open(JSONL_LOG) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                _warn(f"Skipping malformed JSONL line {i+1}: {e}")
    return entries


# ---------------------------------------------------------------------------
# GA Evolution plots
# ---------------------------------------------------------------------------

def plot_generation_accuracy(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping generation_accuracy.png")
        return

    x = list(range(1, len(records) + 1))
    acc     = [r["accuracy"]     * 100 if r["accuracy"]     is not None else None for r in records]
    best    = [r["best_accuracy"] * 100 if r["best_accuracy"] is not None else None for r in records]

    # Filter out None positions for plotting
    def _clean(vals):
        xs, ys = [], []
        for xi, yi in zip(x, vals):
            if yi is not None:
                xs.append(xi); ys.append(yi)
        return xs, ys

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        xb, yb = _clean(best)
        xa, ya = _clean(acc)
        if xb:
            ax.plot(xb, yb, color=ACCENT1, linewidth=2, label="Best Accuracy", marker="o", markersize=3)
        if xa:
            ax.plot(xa, ya, color=ACCENT2, linewidth=1.5, linestyle="--", label="Accuracy (test)", marker="s", markersize=2)
        _apply_style(ax, "Model Accuracy Over Evaluation Order", "Model Index", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "generation_accuracy.png"), saved_files)


def plot_population_diversity(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping population_diversity.png")
        return

    accs = [r["accuracy"] * 100 for r in records if r["accuracy"] is not None]
    if not accs:
        _warn("No accuracy values found — skipping population_diversity.png")
        return

    # Group into batches of ~10 to simulate generation-level diversity
    batch_size = 10
    batches, labels = [], []
    for i in range(0, len(accs), batch_size):
        chunk = accs[i:i + batch_size]
        if chunk:
            batches.append(chunk)
            labels.append(f"Gen {i // batch_size + 1}")

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(batches) * 1.2), 5))
        bp = ax.boxplot(
            batches,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="#2a3a6e", color=ACCENT1),
            medianprops=dict(color=ACCENT2, linewidth=2),
            whiskerprops=dict(color="#6a7aad"),
            capprops=dict(color="#6a7aad"),
            flierprops=dict(marker="o", color=ACCENT3, alpha=0.5, markersize=4),
        )
        _apply_style(ax, "Population Diversity (Accuracy Spread per Generation Batch)",
                     "Generation Batch", "Accuracy (%)")
        _save(fig, os.path.join(out_dir, "population_diversity.png"), saved_files)


def plot_best_vs_avg_accuracy(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping best_vs_avg_accuracy.png")
        return

    accs = [r["accuracy"] * 100 for r in records if r["accuracy"] is not None]
    bests = [r["best_accuracy"] * 100 for r in records if r["best_accuracy"] is not None]
    if not accs:
        _warn("No accuracy values — skipping best_vs_avg_accuracy.png")
        return

    batch_size = 10
    avg_per_batch, best_per_batch, gen_labels = [], [], []
    for i in range(0, max(len(accs), len(bests)), batch_size):
        chunk_acc  = accs[i:i + batch_size]
        chunk_best = bests[i:i + batch_size] if bests else []
        if not chunk_acc:
            break
        avg_per_batch.append(sum(chunk_acc) / len(chunk_acc))
        best_per_batch.append(max(chunk_best) if chunk_best else max(chunk_acc))
        gen_labels.append(f"Gen {i // batch_size + 1}")

    xs = list(range(len(gen_labels)))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.2), 5))
        bars = ax.bar(xs, avg_per_batch, color=BAR_COLOR, alpha=0.8, label="Avg Accuracy", zorder=2)
        ax.plot(xs, best_per_batch, color=ACCENT1, linewidth=2.5, marker="D",
                markersize=6, label="Best Accuracy", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(gen_labels)
        _apply_style(ax, "Best vs Average Accuracy per Generation Batch",
                     "Generation Batch", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "best_vs_avg_accuracy.png"), saved_files)


# ---------------------------------------------------------------------------
# LLM fine-tuning plots
# ---------------------------------------------------------------------------

def plot_reward_over_iterations(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping reward_over_iterations.png")
        return

    rewards = [e.get("reward", 0.0) for e in entries]
    xs = list(range(1, len(rewards) + 1))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [ACCENT3 if r > 0 else ACCENT2 for r in rewards]
        ax.bar(xs, rewards, color=colors, alpha=0.85, zorder=2)
        ax.axhline(0, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.5, label="y = 0")
        ax.plot(xs, rewards, color=ACCENT1, linewidth=1.5, alpha=0.7)
        _apply_style(ax, "RL Reward Over Meta-Evolution Iterations",
                     "Iteration", "Reward")
        pos_patch = mpatches.Patch(color=ACCENT3, label="Positive reward")
        neg_patch = mpatches.Patch(color=ACCENT2, label="Penalty")
        ax.legend(handles=[pos_patch, neg_patch])
        _save(fig, os.path.join(out_dir, "reward_over_iterations.png"), saved_files)


def plot_syntax_success_rate(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping syntax_success_rate.png")
        return

    valid = [1 if e.get("valid_syntax", False) else 0 for e in entries]
    xs = list(range(1, len(valid) + 1))
    window = 10

    # Rolling success rate
    rolling = []
    for i in range(len(valid)):
        start = max(0, i - window + 1)
        chunk = valid[start:i + 1]
        rolling.append(sum(chunk) / len(chunk) * 100)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(xs, rolling, alpha=0.2, color=ACCENT1)
        ax.plot(xs, rolling, color=ACCENT1, linewidth=2, label=f"Rolling success rate (window={window})")
        ax.axhline(50, color=ACCENT2, linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
        ax.set_ylim(0, 105)
        _apply_style(ax, "Syntax Success Rate Over Iterations",
                     "Iteration", "Success Rate (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "syntax_success_rate.png"), saved_files)


def plot_score_improvement(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping score_improvement.png")
        return

    # Only use entries that have both score fields
    filtered = [e for e in entries if "score" in e]
    if not filtered:
        _warn("No 'score' fields in LLM logs — skipping score_improvement.png")
        return

    xs      = list(range(1, len(filtered) + 1))
    scores  = [e.get("score", 0.0)         for e in filtered]
    rewards = [e.get("reward", 0.0)         for e in filtered]
    # Derive baseline: baseline = score - reward (since reward = score - baseline in meta_evolver)
    baselines = [max(0.0, s - r) for s, r in zip(scores, rewards)]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, baselines, color=ACCENT2, linewidth=2, linestyle="--",
                marker="o", markersize=4, label="Baseline Score")
        ax.plot(xs, scores,   color=ACCENT3, linewidth=2,
                marker="s", markersize=4, label="New Score")
        ax.fill_between(xs, baselines, scores,
                        where=[s > b for s, b in zip(scores, baselines)],
                        alpha=0.2, color=ACCENT3, label="Improvement region")
        ax.fill_between(xs, baselines, scores,
                        where=[s <= b for s, b in zip(scores, baselines)],
                        alpha=0.15, color=ACCENT2, label="Regression region")
        _apply_style(ax, "Score Improvement per Iteration (LLM Fine-Tuning)",
                     "Iteration", "Score")
        ax.legend()
        _save(fig, os.path.join(out_dir, "score_improvement.png"), saved_files)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir    = os.path.join(VIZ_ROOT, f"run_{timestamp}")
    ga_dir     = os.path.join(run_dir, "ga_evolution")
    ft_dir     = os.path.join(run_dir, "fine_tuning")

    os.makedirs(ga_dir, exist_ok=True)
    os.makedirs(ft_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  visualize_training.py — run: {timestamp}")
    print(f"  Output root: {os.path.relpath(run_dir, BASE_DIR)}")
    print(f"{'='*60}\n")

    saved_files = []

    # ── GA evolution ────────────────────────────────────────────────────────
    print("[1/2] Loading stats records …")
    records = load_stats_records()
    print(f"      Found {len(records)} evaluated model(s).\n")

    print("  Generating GA evolution plots …")
    plot_generation_accuracy(records,     ga_dir, saved_files)
    plot_population_diversity(records,    ga_dir, saved_files)
    plot_best_vs_avg_accuracy(records,    ga_dir, saved_files)

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    print("\n[2/2] Loading LLM evolution logs …")
    entries = load_llm_logs()
    print(f"      Found {len(entries)} log entry(ies).\n")

    print("  Generating fine-tuning plots …")
    plot_reward_over_iterations(entries, ft_dir, saved_files)
    plot_syntax_success_rate(entries,   ft_dir, saved_files)
    plot_score_improvement(entries,     ft_dir, saved_files)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Saved {len(saved_files)} plot(s):")
    for p in saved_files:
        print(f"    • {os.path.relpath(p, BASE_DIR)}")
    if not saved_files:
        print("    (none — check warnings above)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
meta_visualization.py
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
    python3 meta_visualization.py
"""

import os
import json
import warnings
import glob
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
STATS_DIR      = os.path.join(BASE_DIR, "stats")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
PIPELINE_DIR   = os.environ.get("PIPELINE_DIR", os.path.join(BASE_DIR, "meta_evolution"))
VIZ_ROOT       = os.path.join(PIPELINE_DIR, "visualizations")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# PLOT_STYLE = {
#     "figure.facecolor": "#0f1117",
#     "axes.facecolor":   "#1a1d2e",
#     "axes.edgecolor":   "#3a3f5c",
#     "axes.labelcolor":  "#e0e0e0",
#     "xtick.color":      "#b0b0b0",
#     "ytick.color":      "#b0b0b0",
#     "text.color":       "#e0e0e0",
#     "grid.color":       "#2a2d3e",
#     "legend.facecolor": "#1a1d2e",
#     "legend.edgecolor": "#3a3f5c",
# }
PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "text.color":       "black",
    "grid.color":       "gray",
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
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
    # ax.grid(True, linestyle="--", alpha=0.4)
    # ax.tick_params(colors="#b0b0b0")
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(colors="black")


def _save(fig, path, saved_files, suffix=""):
    if suffix:
        base, ext = os.path.splitext(path)
        path = f"{base}_{suffix}{ext}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor='white', transparent=False)
    plt.close(fig)
    saved_files.append(path)
    print(f"  [saved] {os.path.relpath(path, BASE_DIR)}")


def _warn(msg):
    print(f"  [WARN]  {msg}")

def _determine_dataset(filename):
    """
    Determine the dataset ('cifar10' or 'cifar100') from the filename using explicit conditions.
    """
    if "imagenet100" in filename:
        return "imagenet100"
    elif "cifar100" in filename:
        return "cifar100"
    elif "cifar10" in filename:
        return "cifar10"
    else:
        return "cifar10"

def _extract_log_timestamp(target_ts=None):
    """
    Extract the experiment timestamp from source log filenames.
    Priority: target_ts > GA_EVAL_LOG > glob search fallback.
    Determines dataset by checking if 'cifar100' or 'cifar10' is in the log file name.
    """
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')

    # 1. Check if the environment variable points directly to the active log
    env_log = os.environ.get("GA_EVAL_LOG")
    if env_log and os.path.exists(env_log):
        if not target_ts or target_ts in env_log:
            base = os.path.basename(env_log)
            dataset = _determine_dataset(base)
            match = ts_pattern.search(base)
            ts = match.group(1) if match else target_ts
            
            model_name = ""
            remainder = base.replace(f"ga_evaluations_{dataset}_", "").replace(f"LLM-evolution-logs_{dataset}_", "").replace(".jsonl", "")
            parts = remainder.split("_")
            if len(parts) > 2 and "-" in remainder:
                model_name = "_".join(parts[:-2])
                
            return ts, dataset, model_name

    # 2. If a specific timestamp is passed, find the most recent file matching it
    if target_ts:
        search_dirs = [LOGS_DIR, os.path.join(BASE_DIR, "logs_cifar10"), os.path.join(BASE_DIR, "logs_cifar100")]
        all_files = []
        for d in search_dirs:
            all_files.extend(glob.glob(os.path.join(d, "ga_evaluations_*.jsonl")))
            
        for f in all_files:
            if target_ts and target_ts in f:
                base = os.path.basename(f)
                dataset = _determine_dataset(base)
                
                model_name = ""
                remainder = base.replace(f"ga_evaluations_{dataset}_", "").replace(f"LLM-evolution-logs_{dataset}_", "").replace(".jsonl", "")
                parts = remainder.split("_")
                if len(parts) > 2 and "-" in remainder:
                    model_name = "_".join(parts[:-2])
                    
                return target_ts, dataset, model_name
    
    # 3. Last fallback: return target_ts and empty dataset/model
    return target_ts, "cifar10", ""
    
    # 3. No target specified -> find the absolute most recent log file
    search_patterns = [
        os.path.join(BASE_DIR, "logs_cifar10", "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar100", "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar10", "LLM-evolution-logs_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar100", "LLM-evolution-logs_*.jsonl"),
        os.path.join(LOGS_DIR, "ga_evaluations_*.jsonl"),
        os.path.join(LOGS_DIR, "LLM-evolution-logs_*.jsonl"),
        os.path.join(LOGS_DIR, "pod_*.log"),
        os.path.join(BASE_DIR, "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "LLM-evolution-logs_*.jsonl"),
    ]
    
    all_files = []
    for pattern in search_patterns:
        all_files.extend(glob.glob(pattern))
        
    if all_files:
        latest = max(all_files, key=os.path.getmtime)
        match = ts_pattern.search(os.path.basename(latest))
        if match:
            dataset = _determine_dataset(os.path.basename(latest))
            return match.group(1), dataset
    
    # Final fallback: current wall-clock time
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "cifar10"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats_records(target_ts=None):
    """
    Read the ga_evaluations*.jsonl to get exact chronological evaluation order for the current run.
    """
    records = []
    if target_ts:
        log_files = glob.glob(os.path.join(BASE_DIR, "*_pipeline", "logs_*", "*", "*", f"ga_evaluations*{target_ts}.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_*", "*", "*", f"ga_evaluations*{target_ts}.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, f"ga_evaluations*{target_ts}.jsonl"))
        log_files = [f for f in log_files if os.path.exists(f)]
    else:
        log_files = glob.glob(os.path.join(BASE_DIR, "*_pipeline", "logs_*", "*", "*", "ga_evaluations*.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_*", "*", "*", "ga_evaluations*.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, "ga_evaluations*.jsonl"))
        if not log_files:
            log_files = glob.glob(os.path.join(BASE_DIR, "ga_evaluations*.jsonl"))
            
    if not log_files:
        _warn(f"No ga_evaluations{'* ' if not target_ts else '_'+target_ts}.jsonl files found")
        return records
        
    latest_log = log_files[0] if target_ts else max(log_files, key=os.path.getmtime)
    print(f"  [Info] Loading GA eval logs from: {os.path.basename(latest_log)}")
    
    with open(latest_log) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                records.append(data)
            except Exception as e:
                _warn(f"Could not read line {i+1} in {latest_log}: {e}")

    return records


def group_by_meta_iteration(records, llm_entries):
    attempts = []
    for d in llm_entries:
        ts_str = d.get("fine_tune_start_time")
        if ts_str:
            attempts.append({
                "attempt": d.get("attempt"),
                "end_time": datetime.fromisoformat(ts_str),
                "evals": []
            })
    attempts.append({"attempt": "Final", "end_time": datetime.max, "evals": []})
    
    for r in records:
        if "timestamp" not in r: continue
        ev_time = datetime.fromisoformat(r["timestamp"])
        for a in attempts:
            if ev_time <= a["end_time"]:
                a["evals"].append(r)
                break
                
    return [a for a in attempts if a["evals"]]


def load_llm_logs(target_ts=None):
    """
    Read the LLM-evolution-logs*.jsonl. Returns list of dicts.
    Expected fields: method, score, reward, valid_syntax, timestamp.
    """
    if target_ts:
        log_files = glob.glob(os.path.join(BASE_DIR, "*_pipeline", "logs_*", "*", "*", f"LLM-evolution-logs*{target_ts}.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_*", "*", "*", f"LLM-evolution-logs*{target_ts}.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, f"LLM-evolution-logs*{target_ts}.jsonl"))
        log_files = [f for f in log_files if os.path.exists(f)]
    else:
        log_files = glob.glob(os.path.join(BASE_DIR, "*_pipeline", "logs_*", "*", "*", "LLM-evolution-logs*.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_*", "*", "*", "LLM-evolution-logs*.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, "LLM-evolution-logs*.jsonl"))
        if not log_files:
            log_files = glob.glob(os.path.join(BASE_DIR, "LLM-evolution-logs*.jsonl"))
            
    if not log_files:
        _warn(f"No LLM-evolution-logs{'* ' if not target_ts else '_'+target_ts}.jsonl files found")
        return []
        
    latest_log = log_files[0] if target_ts else max(log_files, key=os.path.getmtime)
    print(f"  [Info] Loading LLM logs from: {os.path.basename(latest_log)}")
    
    entries = []
    with open(latest_log) as f:
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

def group_meta_generations(records):
    generations = []
    idx = 0
    if idx < len(records):
        generations.append({"generation": len(generations)+1, "evals": records[idx:idx+20]})
        idx += 20
    if idx < len(records):
        generations.append({"generation": len(generations)+1, "evals": records[idx:idx+20]})
        idx += 20
    while idx < len(records):
        generations.append({"generation": len(generations)+1, "evals": records[idx:idx+15]})
        idx += 15
    # Remove empty generations just in case
    return [g for g in generations if g["evals"]]


def plot_generation_accuracy(records, llm_entries, out_dir, saved_files, suffix=""):
    generations = group_meta_generations(records)

    if not generations:
        _warn("No grouped records found — skipping generation_accuracy.png")
        return
        
    gen_numbers, avg_accuracies, peak_accuracies, running_peaks = [], [], [], []
    running_peak = 0.0
    
    for g in generations:
        accs = [e.get("accuracy", 0) for e in g["evals"] if "accuracy" in e]
        if not accs: continue
        avg_acc = np.mean(accs)
        peak_acc = max(accs)
        running_peak = max(running_peak, peak_acc)
        
        gen_numbers.append(g["generation"])
        avg_accuracies.append(avg_acc)
        peak_accuracies.append(peak_acc)
        running_peaks.append(running_peak)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(gen_numbers) * 0.4), 7))
        ax.plot(gen_numbers, avg_accuracies, label="Average Accuracy",
                color="#3b82f6", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
        ax.plot(gen_numbers, peak_accuracies, label="Peak Accuracy",
                color="#f97316", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
        ax.plot(gen_numbers, running_peaks, label="Running Best (cumulative)",
                color="#10b981", linewidth=2.5, linestyle="--")

        ax.set_xlabel("Number of Generation", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title("LLM-Guided GA: Accuracy per Generation", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(colors="black")
        
        max_x = max(gen_numbers) if gen_numbers else 5
        xticks = list(range(0, max_x + 5, 5))
        if 1 not in xticks: xticks.insert(1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right')
            
        if running_peaks:
            upper_limit = min(100, max(running_peaks) + 5)
            lower_limit = max(0, min(min(avg_accuracies), min(peak_accuracies)) - 5)
            ax.set_ylim(lower_limit, upper_limit)
        else:
            ax.set_ylim(0, 100)
            
        if gen_numbers:
            ax.set_xlim(1, max(gen_numbers))

        if running_peaks:
            ax.annotate(f"{running_peaks[-1]:.2f}%",
                        xy=(gen_numbers[-1], running_peaks[-1]),
                        xytext=(-60, 15), textcoords="offset points",
                        fontsize=11, fontweight="bold", color="#10b981",
                        arrowprops=dict(arrowstyle="->", color="#10b981"))
                        
        plt.tight_layout()
        path = os.path.join(out_dir, "generation_accuracy.png")
        _save(fig, path, saved_files, suffix)


def plot_population_diversity(records, llm_entries, out_dir, saved_files, suffix=""):
    generations = group_meta_generations(records)

    if not generations:
        _warn("No records found — skipping population_diversity.png")
        return

    batches, positions = [], []
    for g in generations:
        accs = [e["accuracy"] for e in g["evals"] if e.get("accuracy") is not None]
        if accs:
            batches.append(accs)
            positions.append(g["generation"])

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(batches) * 0.5), 5))
        bp = ax.boxplot(
            batches,
            positions=positions,
            patch_artist=True,
            boxprops=dict(facecolor="#2a3a6e", color=ACCENT1),
            medianprops=dict(color=ACCENT2, linewidth=2),
            whiskerprops=dict(color="#6a7aad"),
            capprops=dict(color="#6a7aad"),
            flierprops=dict(marker="o", color=ACCENT3, alpha=0.5, markersize=4),
        )
        
        max_x = max(positions) if positions else 5
        xticks = list(range(0, max_x + 5, 5))
        if 1 not in xticks: xticks.insert(1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right')
            
        _apply_style(ax, "Population Diversity per Generation",
                     "Number of Generation", "Accuracy (%)")
        _save(fig, os.path.join(out_dir, "population_diversity.png"), saved_files, suffix)


def plot_best_vs_avg_accuracy(records, llm_entries, out_dir, saved_files, suffix=""):
    generations = group_meta_generations(records)

    if not generations:
        _warn("No records found — skipping best_vs_avg_accuracy.png")
        return

    avg_per_batch, best_per_batch, median_per_batch, ci_per_batch, xs = [], [], [], [], []
    
    for g in generations:
        chunk_acc = [e["accuracy"] for e in g["evals"] if e.get("accuracy") is not None]
        chunk_best = [e.get("best_accuracy") for e in g["evals"] if e.get("best_accuracy") is not None]
        if not chunk_acc: continue
        
        avg = sum(chunk_acc) / len(chunk_acc)
        avg_per_batch.append(avg)
        best_per_batch.append(max(chunk_best) if chunk_best else max(chunk_acc))
        median_per_batch.append(np.median(chunk_acc))
        
        std = np.std(chunk_acc, ddof=1) if len(chunk_acc) > 1 else 0
        ci = 1.96 * (std / np.sqrt(len(chunk_acc)))
        ci_per_batch.append(ci)
        
        xs.append(g["generation"])

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.5), 5))
        if xs:
            ax.set_xlim(1, max(xs))
            min_y = min([x for x in np.array(avg_per_batch) - np.array(ci_per_batch)])
            ax.set_ylim(max(0, min_y - 5), min(100, max(best_per_batch) + 5))
        
        ax.plot(xs, avg_per_batch, color=BAR_COLOR, alpha=0.8, label="Avg Accuracy", zorder=2, linewidth=2)
        
        lower_bound = np.array(avg_per_batch) - np.array(ci_per_batch)
        upper_bound = np.array(avg_per_batch) + np.array(ci_per_batch)
        ax.fill_between(xs, lower_bound, upper_bound, alpha=0.2, color=BAR_COLOR, zorder=1, label="95% CI (Avg)")
        
        ax.plot(xs, median_per_batch, color="#2ca02c", alpha=0.9, linestyle="--", label="Median Accuracy", zorder=2, linewidth=2)
        ax.plot(xs, best_per_batch, color=ACCENT1, linewidth=2.5, marker="D", markersize=4, label="Best Accuracy", zorder=3)
                
        max_x = max(xs) if xs else 5
        xticks = list(range(0, max_x + 5, 5))
        if 1 not in xticks: xticks.insert(1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right')
            
        _apply_style(ax, "Best vs Average Accuracy per Generation",
                     "Number of Generation", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "best_vs_avg_accuracy.png"), saved_files, suffix)


def plot_time_per_generation(records, llm_entries, out_dir, saved_files, suffix=""):
    generations = group_meta_generations(records)

    if not generations:
        _warn("No records found — skipping time_per_generation.png")
        return

    gen_numbers = []
    gen_times = []
    
    for g in generations:
        gen_numbers.append(g["generation"])
        times = [datetime.fromisoformat(e["timestamp"]) for e in g["evals"] if "timestamp" in e]
        if times:
            gen_duration = (times[-1] - times[0]).total_seconds() / 60.0
            if len(times) > 1:
                avg_model_time = (times[-1] - times[0]).total_seconds() / (len(times) - 1)
                gen_duration += avg_model_time / 60.0
            gen_times.append(gen_duration)
        else:
            gen_times.append(0.0)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(gen_numbers, gen_times, label="Time Taken (per Generation)",
                color="#a855f7", linewidth=2.0, alpha=0.9, marker="s", markersize=5)
        
        ax.set_xlabel("Number of Generation", fontsize=13)
        ax.set_ylabel("Time Taken (Minutes)", fontsize=13)
        ax.set_title("LLM-Guided GA: Compute Time per Generation", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(colors="black")
        
        max_x = max(gen_numbers) if gen_numbers else 5
        xticks = list(range(0, max_x + 5, 5))
        if 1 not in xticks: xticks.insert(1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right')
            
        plt.tight_layout()
        path = os.path.join(out_dir, "time_per_generation.png")
        _save(fig, path, saved_files, suffix)


# ---------------------------------------------------------------------------
# LLM fine-tuning plots
# ---------------------------------------------------------------------------

def plot_reward_over_iterations(entries, out_dir, saved_files, suffix=""):
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
        _save(fig, os.path.join(out_dir, "reward_over_iterations.png"), saved_files, suffix)


def plot_syntax_success_rate(entries, out_dir, saved_files, suffix=""):
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
        _save(fig, os.path.join(out_dir, "syntax_success_rate.png"), saved_files, suffix)


def plot_score_improvement(entries, out_dir, saved_files, suffix=""):
    if not entries:
        _warn("No LLM log entries — skipping score_improvement.png")
        return

    # Only use entries that have both score fields
    filtered = [e for e in entries if "score" in e and e.get("reward", 0.0) > 0]
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
        _save(fig, os.path.join(out_dir, "score_improvement.png"), saved_files, suffix)


def plot_peak_accuracy_over_iterations(entries, out_dir, saved_files, suffix=""):
    if not entries:
        _warn("No LLM log entries — skipping meta_peak_accuracy.png")
        return
    accs = [e.get("peak_accuracy", 0.0) for e in entries if e.get("reward", 0.0) > 0]
    xs = list(range(1, len(accs) + 1))
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, accs, color=ACCENT1, linewidth=2.5, marker="o", markersize=5, label="Peak Accuracy")
        ax.fill_between(xs, accs, alpha=0.15, color=ACCENT1)
        _apply_style(ax, "Peak GA Accuracy Over Meta-Iterations", "Meta-Iteration", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "meta_peak_accuracy.png"), saved_files, suffix)


def plot_modification_success_rate(entries, out_dir, saved_files, suffix=""):
    if not entries:
        _warn("No LLM log entries — skipping llm_success_rates.png")
        return
    syntax_valid = [1 if e.get("valid_syntax", False) else 0 for e in entries]
    improved = [1 if e.get("reward", 0.0) > 0 else 0 for e in entries]
    xs = list(range(1, len(syntax_valid) + 1))
    window = min(10, max(2, len(entries) // 5))  # Dynamic window, capped
    roll_syntax = [sum(syntax_valid[max(0, i-window+1):i+1]) / len(syntax_valid[max(0, i-window+1):i+1]) * 100 for i in range(len(syntax_valid))]
    roll_improve = [sum(improved[max(0, i-window+1):i+1]) / len(improved[max(0, i-window+1):i+1]) * 100 for i in range(len(improved))]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, roll_syntax, color=ACCENT1, linewidth=2, label=f"Syntax Success (rolling)")
        ax.plot(xs, roll_improve, color=ACCENT3, linewidth=2, linestyle="--", label=f"Improvement Success (rolling)")
        ax.axhline(50, color="#ffffff", linestyle=":", linewidth=1, alpha=0.3)
        ax.set_ylim(-5, 105)
        _apply_style(ax, "LLM Modification Success Rates", "Meta-Iteration", "Success Rate (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "llm_success_rates.png"), saved_files, suffix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target_ts=None, target_dataset=None):
    if len(sys.argv) > 1:
        target_ts = sys.argv[1]
    if len(sys.argv) > 2:
        target_dataset = sys.argv[2]
        
    # Use source log timestamp so visualizations correlate with their experiment
    timestamp, dataset_name, model_name = _extract_log_timestamp(target_ts)
    if target_dataset:
        dataset_name = target_dataset
        
    if model_name:
        suffix = f"{dataset_name}_{model_name}_{timestamp}"
        run_dir = os.path.join(VIZ_ROOT, f"meta_visualization_{suffix}")
    else:
        suffix = f"{dataset_name}_{timestamp}"
        run_dir = os.path.join(VIZ_ROOT, f"meta_visualization_{suffix}")
    ga_dir     = os.path.join(run_dir, "ga_evolution")
    ft_dir     = os.path.join(run_dir, "fine_tuning")

    os.makedirs(ga_dir, exist_ok=True)
    os.makedirs(ft_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  meta_visualization.py — run: {timestamp}")
    print(f"  Output root: {os.path.relpath(run_dir, BASE_DIR)}")
    print(f"{'='*60}\n")

    saved_files = []

    # ── Fine-tuning (Load First) ────────────────────────────────────────────
    print("\n[1/2] Loading LLM evolution logs …")
    entries = load_llm_logs(timestamp)
    print(f"      Found {len(entries)} log entry(ies).\n")

    # ── GA evolution ────────────────────────────────────────────────────────
    print("[2/2] Loading stats records …")
    records = load_stats_records(timestamp)
    print(f"      Found {len(records)} evaluated model(s).\n")

    print("  Generating GA evolution plots …")
    plot_generation_accuracy(records, entries, ga_dir, saved_files, suffix)
    plot_time_per_generation(records, entries, ga_dir, saved_files, suffix)
    plot_population_diversity(records, entries, ga_dir, saved_files, suffix)
    plot_best_vs_avg_accuracy(records, entries, ga_dir, saved_files, suffix)

    print("  Generating fine-tuning plots …")
    plot_reward_over_iterations(entries, ft_dir, saved_files, suffix)
    plot_score_improvement(entries,     ft_dir, saved_files, suffix)
    plot_peak_accuracy_over_iterations(entries, ft_dir, saved_files, suffix)
    plot_modification_success_rate(entries, ft_dir, saved_files, suffix)

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

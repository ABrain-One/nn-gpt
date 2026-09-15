import json
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", os.path.join(BASE_DIR, "cifar10_pipeline"))
VIZ_ROOT = os.path.join(PIPELINE_DIR, "visualizations")

# if len(sys.argv) > 1:
#     LOG_FILE = sys.argv[1]
# else:
#     log_files = glob.glob(os.path.join(LOGS_DIR, "baseline_evaluations_cifar10_*.jsonl")) + \
#                 glob.glob(os.path.join(LOGS_DIR, "baseline_evaluations_*.jsonl"))
#     if not log_files:
#         raise FileNotFoundError(f"No baseline_evaluations*.jsonl found in {LOGS_DIR}")
#     LOG_FILE = max(log_files, key=os.path.getmtime)

GEN1_SIZE = 20
REST_SIZE = 15

def load_evaluations(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def split_into_generations(entries, gen1_size=20, rest_size=15):
    generations = []
    # Generation 1
    generations.append(entries[:gen1_size])
    idx = gen1_size
    # Remaining generations
    while idx < len(entries):
        generations.append(entries[idx:idx + rest_size])
        idx += rest_size
    return generations

# def main():
#     print(f"Loading: {LOG_FILE}")
# def main(dataset=None, log_file_override=None):
#     if dataset:
#         target_logs_dir = os.path.join(BASE_DIR, f"logs_{dataset}")
#     else:
#         target_logs_dir = os.path.join(BASE_DIR, "logs_cifar10")
# 
#     if log_file_override:
#         LOG_FILE = log_file_override
#     elif len(sys.argv) > 1:
#         LOG_FILE = sys.argv[1]
#     else:
#         if dataset:
#             log_files = glob.glob(os.path.join(target_logs_dir, f"baseline_evaluations_{dataset}_*.jsonl"))
#         else:
#             log_files = glob.glob(os.path.join(target_logs_dir, "baseline_evaluations_cifar10_*.jsonl")) + \
#                         glob.glob(os.path.join(target_logs_dir, "baseline_evaluations_*.jsonl"))
#         if not log_files:
#             raise FileNotFoundError(f"No baseline_evaluations*.jsonl found in {target_logs_dir}")
#         LOG_FILE = max(log_files, key=os.path.getmtime)
def main(dataset=None, log_file_override=None):
    global VIZ_ROOT
    # if not dataset:
    #     dataset = os.environ.get("DATASET", "cifar10")
    # 
    # target_pipeline_dir = os.environ.get("PIPELINE_DIR", os.path.join(BASE_DIR, f"{dataset}_pipeline"))
    # VIZ_ROOT = os.path.join(target_pipeline_dir, "visualizations")
    # 
    # if log_file_override and os.path.exists(log_file_override):
    #     LOG_FILE = log_file_override
    # elif os.environ.get("GA_EVAL_LOG") and os.path.exists(os.environ.get("GA_EVAL_LOG")):
    #     LOG_FILE = os.environ.get("GA_EVAL_LOG")
    # elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    #     LOG_FILE = sys.argv[1]

    # Resolve candidate log file first
    candidate_log = None
    if log_file_override and os.path.exists(log_file_override):
        candidate_log = log_file_override
    elif os.environ.get("GA_EVAL_LOG") and os.path.exists(os.environ.get("GA_EVAL_LOG")):
        candidate_log = os.environ.get("GA_EVAL_LOG")
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        candidate_log = sys.argv[1]

    # Infer dataset from candidate log if not provided
    if not dataset and candidate_log:
        base_log = os.path.basename(candidate_log)
        if "imagenet100" in base_log:
            dataset = "imagenet100"
        elif "cifar100" in base_log:
            dataset = "cifar100"
        elif "cifar10" in base_log:
            dataset = "cifar10"

    if not dataset:
        dataset = os.environ.get("DATASET", "cifar10")

    target_pipeline_dir = os.environ.get("PIPELINE_DIR", os.path.join(BASE_DIR, f"{dataset}_pipeline"))
    VIZ_ROOT = os.path.join(target_pipeline_dir, "visualizations")

    if candidate_log:
        LOG_FILE = candidate_log
    else:
        search_dirs = [
            os.path.join(target_pipeline_dir, f"logs_{dataset}", "Baseline"),
            os.path.join(target_pipeline_dir, f"logs_{dataset}"),
            os.path.join(BASE_DIR, f"logs_{dataset}"),
            os.path.join(BASE_DIR, "logs_cifar10"),
            LOGS_DIR
        ]
        log_files = []
        for sdir in search_dirs:
            if os.path.exists(sdir):
                log_files.extend(glob.glob(os.path.join(sdir, "**", f"baseline_evaluations_{dataset}_*.jsonl"), recursive=True))
                log_files.extend(glob.glob(os.path.join(sdir, f"baseline_evaluations_{dataset}_*.jsonl")))
                log_files.extend(glob.glob(os.path.join(sdir, "**", "baseline_evaluations_*.jsonl"), recursive=True))
                log_files.extend(glob.glob(os.path.join(sdir, "baseline_evaluations_*.jsonl")))
        log_files = list(set([os.path.abspath(f) for f in log_files if os.path.isfile(f)]))
        if not log_files:
            raise FileNotFoundError(f"No baseline_evaluations*.jsonl found in search dirs: {search_dirs}")
        LOG_FILE = max(log_files, key=os.path.getmtime)

    print(f"Loading: {LOG_FILE}")
    entries = load_evaluations(LOG_FILE)
    print(f"Total models evaluated: {len(entries)}")

    generations = split_into_generations(entries, GEN1_SIZE, REST_SIZE)
    print(f"Total generations: {len(generations)}")

    gen_numbers = []
    avg_accuracies = []
    peak_accuracies = []
    running_peak = 0.0
    running_peaks = []
    gen_times = []
    prev_end_time = None

    for i, gen in enumerate(generations):
        gen_num = i + 1
        accuracies = [e["accuracy"] for e in gen]
        avg_acc = np.mean(accuracies)
        peak_acc = max(accuracies)
        running_peak = max(running_peak, peak_acc)

        gen_numbers.append(gen_num)
        avg_accuracies.append(avg_acc)
        peak_accuracies.append(peak_acc)
        running_peaks.append(running_peak)

        # Calculate time taken
        times = [datetime.fromisoformat(e["timestamp"]) for e in gen if "timestamp" in e]
        if times:
            gen_start_time = prev_end_time if prev_end_time else times[0]
            gen_end_time = times[-1]
            gen_duration = (gen_end_time - gen_start_time).total_seconds() / 60.0 # in minutes
            
            # For the first generation, if we use times[0], we miss the time of the very first model.
            # We approximate it by adding the average time per model in this generation.
            if prev_end_time is None and len(times) > 1:
                avg_model_time = (times[-1] - times[0]).total_seconds() / (len(times) - 1)
                gen_duration += avg_model_time / 60.0
                
            gen_times.append(gen_duration)
            prev_end_time = gen_end_time
        else:
            gen_times.append(0.0)

        if gen_num <= 5 or gen_num % 10 == 0 or gen_num == len(generations):
            print(f"  Gen {gen_num:3d}: {len(gen):2d} models | Avg: {avg_acc:.2f}% | Peak: {peak_acc:.2f}% | Running Best: {running_peak:.2f}% | Time: {gen_times[-1]:.1f} min")

    # --- Plot ---
    log_basename = os.path.basename(LOG_FILE)
    title_prefix = "Baseline GA" if "baseline_evaluations_" in log_basename else "LLM-Guided GA"

    # fig, ax = plt.subplots(figsize=(14, 7))
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(gen_numbers, avg_accuracies, label="Average Accuracy (per gen)",
            color="#3b82f6", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
    ax.plot(gen_numbers, peak_accuracies, label="Peak Accuracy (per gen)",
            color="#f97316", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
    ax.plot(gen_numbers, running_peaks, label="Running Best (cumulative)",
            color="#10b981", linewidth=2.5, linestyle="--")

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    # ax.set_title("Baseline GA: Accuracy per Generation", fontsize=15, fontweight="bold")
    ax.set_title(f"{title_prefix}: Accuracy per Generation", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    # ax.grid(True, alpha=0.3)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(1, len(generations))

    # Annotate final running best
    ax.annotate(f"{running_peaks[-1]:.2f}%",
                xy=(gen_numbers[-1], running_peaks[-1]),
                xytext=(-60, 15), textcoords="offset points",
                fontsize=11, fontweight="bold", color="#10b981",
                arrowprops=dict(arrowstyle="->", color="#10b981"))

    plt.tight_layout()
    # Extract timestamp from log filename (e.g., "baseline_evaluations_2026-06-10_17-13-19.jsonl" -> "2026-06-10_17-13-19")
    # log_basename = os.path.basename(LOG_FILE)
    # timestamp = log_basename.replace("baseline_evaluations_", "").replace(".jsonl", "")
    # plot_dir = os.path.join(BASE_DIR, "visualizations", f"baseline_{timestamp}")
    if "baseline_evaluations_" in log_basename:
        if "imagenet100" in log_basename: dataset = "imagenet100"
        elif "cifar100" in log_basename: dataset = "cifar100"
        elif "cifar10" in log_basename: dataset = "cifar10"
        else: dataset = ""
        
        # Remove prefix
        remainder = log_basename.replace(f"baseline_evaluations_{dataset}_", "") if dataset else log_basename.replace("baseline_evaluations_", "")
        remainder = remainder.replace(".jsonl", "")
        
        # Parse model name if present
        parts = remainder.split("_")
        if len(parts) > 2 and "-" in remainder:
            # We assume model name contains '-' or is the first part before date
            timestamp = "_".join(parts[-2:])
            model_name = "_".join(parts[:-2])
            suffix = f"{dataset}_{model_name}_{timestamp}"
        else:
            timestamp = remainder
            suffix = f"{dataset}_{timestamp}" if dataset else f"{timestamp}"
        
        plot_dir = os.path.join(VIZ_ROOT, f"baseline_visualization_{suffix}")
            
    elif "ga_evaluations_" in log_basename:
        if "imagenet100" in log_basename: dataset = "imagenet100"
        elif "cifar100" in log_basename: dataset = "cifar100"
        elif "cifar10" in log_basename: dataset = "cifar10"
        else: dataset = ""
        
        remainder = log_basename.replace(f"ga_evaluations_{dataset}_", "") if dataset else log_basename.replace("ga_evaluations_", "")
        remainder = remainder.replace(".jsonl", "")
        
        parts = remainder.split("_")
        if len(parts) > 2 and "-" in remainder:
            timestamp = "_".join(parts[-2:])
            model_name = "_".join(parts[:-2])
            suffix = f"{dataset}_{model_name}_{timestamp}"
        else:
            timestamp = remainder
            suffix = f"{dataset}_{timestamp}" if dataset else f"{timestamp}"
            
        plot_dir = os.path.join(VIZ_ROOT, f"baseline_visualization_{suffix}")
    else:
        timestamp = log_basename.replace(".jsonl", "")
        suffix = timestamp
        plot_dir = os.path.join(VIZ_ROOT, f"baseline_visualization_{suffix}")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"baseline_accuracy_per_generation_{suffix}.png")
    # plt.savefig(plot_path, dpi=150)
    plt.savefig(plot_path, dpi=150, facecolor='white', transparent=False)
    print(f"\nAccuracy plot saved to: {plot_path}")

    # --- Plot 2: Time per generation ---
    # fig2, ax2 = plt.subplots(figsize=(14, 7))
    fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor='white')
    ax2.set_facecolor('white')
    ax2.plot(gen_numbers, gen_times, label="Time Taken (per gen)",
            color="#a855f7", linewidth=2.0, alpha=0.9, marker="s", markersize=5)
    
    ax2.set_xlabel("Generation", fontsize=13)
    ax2.set_ylabel("Time Taken (Minutes)", fontsize=13)
    # ax2.set_title("Baseline GA: Compute Time per Generation", fontsize=15, fontweight="bold")
    ax2.set_title(f"{title_prefix}: Compute Time per Generation", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=11, loc="upper right")
    # ax2.grid(True, alpha=0.3)
    ax2.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xlim(1, len(generations))
    
    plt.tight_layout()
    plot_time_path = os.path.join(plot_dir, f"baseline_time_per_generation_{suffix}.png")
    # fig2.savefig(plot_time_path, dpi=150)
    fig2.savefig(plot_time_path, dpi=150, facecolor='white', transparent=False)
    print(f"Time plot saved to: {plot_time_path}")

    # --- Plot 3: Population Diversity per Generation ---
    fig3, ax3 = plt.subplots(figsize=(14, 7), facecolor='white')
    ax3.set_facecolor('white')

    gen_acc_batches = []
    unique_counts = []

    for gen in generations:
        accs = [e["accuracy"] for e in gen if "accuracy" in e and e["accuracy"] is not None]
        gen_acc_batches.append(accs if accs else [0.0])

        uids = [e.get("uid", e.get("checksum", "")) for e in gen]
        unique_uids = set(uids) - {""}
        unique_counts.append(len(unique_uids) if unique_uids else len(gen))

    bp = ax3.boxplot(
        gen_acc_batches,
        positions=gen_numbers,
        patch_artist=True,
        boxprops=dict(facecolor="#2a3a6e", color="#3b82f6", alpha=0.7),
        medianprops=dict(color="#f97316", linewidth=2),
        whiskerprops=dict(color="#6a7aad"),
        capprops=dict(color="#6a7aad"),
        flierprops=dict(marker="o", color="#ef4444", alpha=0.5, markersize=4),
    )

    ax3.set_xlabel("Generation", fontsize=13)
    ax3.set_ylabel("Accuracy Distribution (%)", fontsize=13)
    ax3.set_title(f"{title_prefix}: Generational Population Diversity", fontsize=15, fontweight="bold")
    ax3.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax3.set_xlim(0.5, len(generations) + 0.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(gen_numbers, unique_counts, label="Unique Architectures Evaluated",
                 color="#10b981", linewidth=2.0, linestyle="--", marker="o", markersize=4)
    ax3_twin.set_ylabel("Unique Architectures Count", fontsize=13, color="#10b981")
    ax3_twin.tick_params(axis='y', labelcolor="#10b981")

    plt.tight_layout()
    plot_div_path = os.path.join(plot_dir, f"baseline_population_diversity_{suffix}.png")
    fig3.savefig(plot_div_path, dpi=150, facecolor='white', transparent=False)
    print(f"Diversity plot saved to: {plot_div_path}")

if __name__ == "__main__":
    main()

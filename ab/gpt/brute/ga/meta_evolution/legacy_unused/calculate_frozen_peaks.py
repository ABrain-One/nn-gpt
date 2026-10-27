import os
import json
import glob

base_dir = "/shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/logs_cifar100/frozen"

peak_accs = []

# Find all meta log files in the subdirectories
for run_dir in glob.glob(os.path.join(base_dir, "run*_seed_*")):
    meta_logs = glob.glob(os.path.join(run_dir, "frozen_meta_log_*.jsonl"))
    if not meta_logs:
        continue
        
    run_peak = 0.0
    for meta_log in meta_logs:
        with open(meta_log, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                acc = data.get("true_peak_accuracy", 0.0)
                if acc > run_peak:
                    run_peak = acc
                    
    if run_peak > 0:
        peak_accs.append(run_peak)
        print(f"Run {os.path.basename(run_dir)} peak accuracy: {run_peak:.2f}%")

if peak_accs:
    avg_acc = sum(peak_accs) / len(peak_accs)
    print(f"\nAverage Peak Accuracy across {len(peak_accs)} runs: {avg_acc:.2f}%")
else:
    print("No data found.")

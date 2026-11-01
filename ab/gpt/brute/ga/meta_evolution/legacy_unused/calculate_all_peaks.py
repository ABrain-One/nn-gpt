import os
import json
import glob

base_dir = "/shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/logs_cifar100/frozen"

results = []

for run_dir in sorted(glob.glob(os.path.join(base_dir, "run*_seed_*"))):
    run_name = os.path.basename(run_dir)
    
    # Files
    true_files = glob.glob(os.path.join(run_dir, "*_true.jsonl"))
    epoch1_files = glob.glob(os.path.join(run_dir, "*_1_epoch.jsonl"))
    
    # The LLM predictor file is the one without _true or _1_epoch
    all_evals = glob.glob(os.path.join(run_dir, "frozen_evaluations_*.jsonl"))
    llm_files = [f for f in all_evals if not f.endswith("_true.jsonl") and not f.endswith("_1_epoch.jsonl")]

    def get_peak(files):
        peak = 0.0
        for fname in files:
            with open(fname, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        acc = data.get("accuracy", 0.0)
                        if acc > peak:
                            peak = acc
                    except:
                        pass
        return peak

    peak_3epoch = get_peak(true_files)
    peak_1epoch = get_peak(epoch1_files)
    peak_llm = get_peak(llm_files)

    results.append({
        "run": run_name,
        "1_epoch": peak_1epoch,
        "3_epoch": peak_3epoch,
        "llm_pred": peak_llm
    })

print(f"{'Run':<15} | {'1-Epoch Peak':<12} | {'3-Epoch Peak':<12} | {'LLM Predictor Peak':<18}")
print("-" * 65)

avg_1 = 0
avg_3 = 0
avg_llm = 0

for res in results:
    print(f"{res['run']:<15} | {res['1_epoch']:<12.2f} | {res['3_epoch']:<12.2f} | {res['llm_pred']:<18.2f}")
    avg_1 += res['1_epoch']
    avg_3 += res['3_epoch']
    avg_llm += res['llm_pred']
    
if results:
    avg_1 /= len(results)
    avg_3 /= len(results)
    avg_llm /= len(results)
    print("-" * 65)
    print(f"{'Average':<15} | {avg_1:<12.2f} | {avg_3:<12.2f} | {avg_llm:<18.2f}")

import os
import json
import subprocess

source_dir = "/shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/logs_cifar100/frozen/04-10-26_20-26"
target_dir = "/shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/logs_cifar100/frozen"

# Mapping of seeds to runs and pods
seed_map = {
    1001: {"run_name": "run1", "pod": "nngpt-fractal-frozen-cifar100-run1-b8c5n"},
    1002: {"run_name": "run2", "pod": "nngpt-fractal-frozen-cifar100-run2-s4lts"},
    1003: {"run_name": "run3", "pod": "nngpt-fractal-frozen-cifar100-run3-vptz2"},
    1004: {"run_name": "run4", "pod": "nngpt-fractal-frozen-cifar100-run4-v5hzt"},
}

# Create output directories in the target_dir
for seed, info in seed_map.items():
    os.makedirs(os.path.join(target_dir, f"{info['run_name']}_seed_{seed}"), exist_ok=True)

jsonl_files = [
    "frozen_evaluations_cifar100_2026-10-04_20-26-21.jsonl",
    "frozen_evaluations_cifar100_2026-10-04_20-26-21_1_epoch.jsonl",
    "frozen_evaluations_cifar100_2026-10-04_20-26-21_true.jsonl",
    "frozen_meta_log_cifar100_2026-10-04_20-26-21.jsonl"
]

# Separate the JSONL files
for jfile in jsonl_files:
    source_path = os.path.join(source_dir, jfile)
    if not os.path.exists(source_path):
        continue
    
    # Open file handlers for each seed
    handlers = {}
    for seed, info in seed_map.items():
        out_path = os.path.join(target_dir, f"{info['run_name']}_seed_{seed}", jfile)
        handlers[seed] = open(out_path, 'w')
        
    with open(source_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                seed = data.get("seed")
                if seed in handlers:
                    handlers[seed].write(line + "\n")
            except Exception as e:
                print(f"Error parsing line in {jfile}: {e}")
                
    # Close handlers
    for h in handlers.values():
        h.close()

# Fetch Pod Logs using kubectl
for seed, info in seed_map.items():
    pod_name = info['pod']
    out_path = os.path.join(target_dir, f"{info['run_name']}_seed_{seed}", "pod_logs.txt")
    print(f"Fetching logs for {pod_name}...")
    try:
        # Run kubectl logs
        with open(out_path, "w") as f_out:
            subprocess.run(["kubectl", "logs", pod_name], stdout=f_out, stderr=subprocess.STDOUT, check=True)
        print(f"Successfully saved pod logs to {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch logs for {pod_name}: {e}")

print("Separation complete!")

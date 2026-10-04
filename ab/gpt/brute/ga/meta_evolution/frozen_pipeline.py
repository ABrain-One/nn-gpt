"""
Frozen-Operator Control Pipeline
=================================
This is a copy of meta_evolver.py with the LLM completely removed.
The GA runs the same 5 meta-iterations × 5 generations schedule,
but the handcrafted operators stay fixed (no LLM, no reward, no fine-tuning).

Purpose: Isolate the effect of the restart schedule from the LLM's code rewrites.
"""
import os
import re
import random
import subprocess
import shutil
import time
import json
import sys
import glob
import gc
import torch
from ab.gpt.util.Eval import Eval
import ab.nn.api as nn_dataset
import pandas as pd
# MONKEYPATCH: Bypass the massive remote database download inside Eval.py
nn_dataset.data = lambda *args, **kwargs: pd.DataFrame(columns=['nn_id'])
nn_dataset.data.cache_clear = lambda: None
from datetime import datetime

from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable_backbone import SEARCH_SPACE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", BASE_DIR)
DATASET = os.environ.get("DATASET", "cifar100")
DATASET_DASH = "cifar-100" if DATASET == "cifar100" else "cifar-10"

# --- Frozen control uses "frozen" as the model name ---
_model_name = "frozen"

# The frozen control uses the BASELINE GA code directly — no modified_GA folder needed.
# We still point TARGET_FILE at a copy so the runner can import it,
# but we never modify it.
MODIFIED_GA_DIR = os.path.join(PIPELINE_DIR, f"modified_GA_{_model_name}")
os.makedirs(MODIFIED_GA_DIR, exist_ok=True)
TARGET_FILE = os.path.join(MODIFIED_GA_DIR, "genetic_algorithm_evolved.py")

# Always reset to pristine baseline at the start of each run
baseline_file = os.path.join(BASE_DIR, "genetic_algorithm_baseline.py")
if os.path.exists(baseline_file):
    shutil.copy(baseline_file, TARGET_FILE)
    print("[Frozen] Copied pristine baseline GA to target file.")

# Ensure __init__.py exists for importability
init_file = os.path.join(MODIFIED_GA_DIR, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write("")

# No checkpoint saving
CHECKPOINT_FILE = None

# Use run_fractal_frozen.py as the inner GA runner
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_fractal_frozen.py")
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FOLDER_TIMESTAMP = datetime.now().strftime("%d-%m-%y_%H-%M")

# Logging directories: logs_cifar100/frozen/DD-MM-YY_HH-MM/
LOGS_DIR = os.path.join(PIPELINE_DIR, f"logs_{DATASET}", _model_name, FOLDER_TIMESTAMP)
os.makedirs(LOGS_DIR, exist_ok=True)
GA_EVAL_LOG_FILE = os.path.join(LOGS_DIR, f"frozen_evaluations_{DATASET}_{RUN_TIMESTAMP}.jsonl")

# Match the LLM runs: 5 generations per inner run, population 20
BENCH_GENS = int(os.environ.get("GENERATIONS", 5))
BENCH_POP = int(os.environ.get("POPULATION_SIZE", 20))

# Random seed for reproducibility
SEED = int(os.environ.get("FROZEN_SEED", random.randint(1, 999999)))
random.seed(SEED)
print(f"[Frozen] Random seed: {SEED}")


class FrozenPipeline:
    """Runs the meta-evolution schedule with handcrafted operators frozen (no LLM)."""

    def __init__(self):
        self.baseline_score = 0.0
        best_info_path = os.path.join(PIPELINE_DIR, f"best_baseline_info_{DATASET}.json")
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    best_data = json.load(f)
                    self.baseline_score = float(best_data.get("fitness", 0.0))
                    print(f"[Frozen] Loaded Baseline Score: {self.baseline_score:.2f}%")
            except Exception:
                pass

        if self.baseline_score == 0.0:
            print("[Frozen] Running initial Baseline Benchmark...")
            self.baseline_score = self.run_benchmark(gens=1)["peak_accuracy"]
            print(f"[Frozen] Calculated Baseline: {self.baseline_score:.4f}%")

        self.global_best_score = 0.0
        self.global_archive_size = 0

    def _cuda_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
        print("[Frozen] GPU cleanup attempted (gc + empty_cache)")

    def run_benchmark(self, gens=None, meta_iteration=0):
        """Run the inner GA benchmark. Identical to meta_evolver.run_benchmark()."""
        import statistics
        if gens is None:
            gens = BENCH_GENS

        cmd = [sys.executable, RUNNER_SCRIPT, "--gens", str(gens), "--pop", str(BENCH_POP)]
        env = os.environ.copy()
        env["GA_EVAL_LOG"] = GA_EVAL_LOG_FILE
        # Separate stats directory to prevent cache cross-contamination
        env["STATS_SUBDIR"] = f"meta_{_model_name}"
        # Pass meta-iteration number and seed to the inner runner for JSONL logging
        env["META_ITERATION"] = str(meta_iteration)
        env["FROZEN_SEED"] = str(SEED)
        print(f"[Frozen] Setting STATS_SUBDIR=meta_{_model_name} for subprocess → stats/meta_{_model_name}/")

        runs = 1
        scores = []
        peak_accs = []
        true_peak_accs = []
        archive_size = 0
        error_trace = ""

        for i in range(runs):
            print(f"\n[Frozen] Running Benchmark {i+1}/{runs} (meta-iteration {meta_iteration})...")
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

                full_output = ""
                for line in process.stdout:
                    print(line, end="")
                    full_output += line

                BENCH_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", 1800))
                process.wait(timeout=BENCH_TIMEOUT)

                if process.returncode != 0:
                    lines = full_output.strip().splitlines()
                    error_trace = "\n".join(lines[-5:]) if lines else "Process failed with no output"

                score_match = re.search(r"META_SCORE:\s*([\d\.]+)", full_output)
                score = float(score_match.group(1)) if score_match else 0.0

                peak_match = re.search(r"PEAK_ACCURACY:\s*([\d\.]+)", full_output)
                peak_acc = float(peak_match.group(1)) if peak_match else 0.0

                true_peak_match = re.search(r"TRUE_PEAK_ACCURACY:\s*([\d\.]+)", full_output)
                true_peak_acc = float(true_peak_match.group(1)) if true_peak_match else peak_acc

                top3_match = re.search(r"TOP3_MEAN:\s*([\d\.]+)", full_output)
                top3_acc = float(top3_match.group(1)) if top3_match else peak_acc

                archive_match = re.search(r"ARCHIVE_SIZE:\s*(\d+)", full_output)
                archive_size = int(archive_match.group(1)) if archive_match else 0

                scores.append(top3_acc)
                peak_accs.append(peak_acc)
                true_peak_accs.append(true_peak_acc)
            except Exception as e:
                print(f"[Frozen] Benchmark Exception: {e}")
                scores.append(0.0)
                peak_accs.append(0.0)
                true_peak_accs.append(0.0)
                archive_size = 0
                error_trace = str(e)

        median_top3 = statistics.median(scores) if scores else 0.0
        median_peak = statistics.median(peak_accs) if peak_accs else 0.0
        median_true_peak = statistics.median(true_peak_accs) if true_peak_accs else 0.0
        print(f"[Frozen] Median Top-3 Quality over {runs} runs: {median_top3:.4f}")
        print(f"[Frozen] Median True Peak Accuracy over {runs} runs: {median_true_peak:.4f}")

        # Cleanup
        scores.clear()
        peak_accs.clear()
        true_peak_accs.clear()
        self._cuda_cleanup()

        # Read fitness_history from the trajectory file
        fitness_history = []
        trajectory_pattern = os.path.join(PIPELINE_DIR, f"evolved_results_{DATASET}_*.json")
        for traj_file in glob.glob(trajectory_pattern):
            try:
                with open(traj_file) as f:
                    traj_data = json.load(f)
                if "fitness_history" in traj_data and traj_data["fitness_history"]:
                    fitness_history = traj_data["fitness_history"]
            except Exception:
                pass

        return {
            "top3_mean": median_top3,
            "peak_accuracy": median_peak,
            "true_peak_accuracy": median_true_peak,
            "archive_size": archive_size,
            "error_trace": error_trace,
            "fitness_history": fitness_history
        }

    def run_frozen_iteration(self, meta_iteration, total_iterations):
        """
        Run one meta-iteration with frozen (unchanged) operators.
        This is the identity step: no LLM, no code rewrite, just run the GA benchmark.
        """
        print(f"\n[Frozen] === Meta-Iteration {meta_iteration}/{total_iterations} ===")
        print(f"[Frozen] Identity step: operators unchanged (handcrafted baseline).")

        bench_stats = self.run_benchmark(meta_iteration=meta_iteration)
        new_score = bench_stats["true_peak_accuracy"]
        top3_mean = bench_stats["top3_mean"]
        new_archive_size = bench_stats["archive_size"]

        # Track global best (same as meta_evolver)
        if new_score > self.global_best_score:
            print(f"[Frozen] --> NEW BEST! {new_score:.2f}% (was {self.global_best_score:.2f}%)")
            self.global_best_score = new_score
        if new_archive_size > self.global_archive_size:
            print(f"[Frozen] --> ARCHIVE EXPANDED! {new_archive_size} cells (was {self.global_archive_size})")
            self.global_archive_size = new_archive_size

        # Log this iteration (simplified — no LLM fields)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "meta_iteration": meta_iteration,
            "total_iterations": total_iterations,
            "seed": SEED,
            "method": "frozen_identity",
            "valid_syntax": True,
            "score": new_score,
            "peak_accuracy": bench_stats["peak_accuracy"],
            "true_peak_accuracy": bench_stats["true_peak_accuracy"],
            "top3_mean": top3_mean,
            "archive_size": new_archive_size,
            "error_trace": bench_stats.get("error_trace", ""),
            "fitness_history": bench_stats.get("fitness_history", []),
            # No LLM fields
            "llm_loaded": False,
            "reward": 0.0,
            "fine_tune_expected": False,
            "fine_tune_completed": False,
        }

        log_file = os.path.join(LOGS_DIR, f"frozen_meta_log_{DATASET}_{RUN_TIMESTAMP}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[Frozen] Logged iteration {meta_iteration} to {log_file}")

        return True


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  FROZEN-OPERATOR CONTROL PIPELINE")
    print(f"  Dataset: {DATASET} ({DATASET_DASH})")
    print(f"  Seed: {SEED}")
    print(f"  Schedule: 5 meta-iterations × {BENCH_GENS} generations × {BENCH_POP} population")
    print(f"  Logs: {LOGS_DIR}")
    print(f"  operator LLM: none (frozen); fitness predictor: identical to main arm")
    print(f"{'='*60}\n")

    pipeline = FrozenPipeline()

    META_ITERATIONS = int(os.environ.get("META_ATTEMPTS", 5))

    print(f"\n[Frozen] Starting {META_ITERATIONS} meta-iterations with frozen operators.")

    for iteration in range(1, META_ITERATIONS + 1):
        pipeline.run_frozen_iteration(iteration, META_ITERATIONS)
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"  FROZEN CONTROL COMPLETE")
    print(f"  Global Best Score: {pipeline.global_best_score:.2f}%")
    print(f"  Archive Size: {pipeline.global_archive_size}")
    print(f"  Seed: {SEED}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"{'='*60}")

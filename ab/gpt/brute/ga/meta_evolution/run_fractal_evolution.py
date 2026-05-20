import os
import argparse
import hashlib
import json
import time
import shutil
from datetime import datetime
from contextlib import contextmanager
import sys

# FIX MODULE PATH: Add repo root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

import torch
from ab.gpt.brute.ga.meta_evolution.genetic_algorithm import GeneticAlgorithm
from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable import SEARCH_SPACE, generate_model_code_string
from ab.gpt.util.Eval import Eval

import logging

# Configure logging to be simpler (remove timestamps for cleaner output)
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This is the folder where unique fractal models will be saved
ARCH_DIR = os.path.join(BASE_DIR, 'ga_fractal_arch') 
STATS_DIR = os.path.join(BASE_DIR, 'stats')
CHECKPOINT = 'fractal_ga_ckpt.pkl'
BEST_STATS_DIR = os.path.join(BASE_DIR, 'best_fractal_stats')

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

seen_checksums = set()

# Persist checksums across runs: load checksums from existing stats folders
def _load_existing_checksums():
    """Scan stats/ directory for previously evaluated models to avoid re-evaluation."""
    count = 0
    # prefix = "img-classification_cifar_FractalNet-"   # BUG: missing '-10', never matched any folder
    prefix = "img-classification_cifar-10_FractalNet-"
    if os.path.isdir(STATS_DIR):
        for name in os.listdir(STATS_DIR):
            if name.startswith(prefix):
                checksum = name[len(prefix):]
                seen_checksums.add(checksum)
                count += 1
    if count:
        print(f"[Init] Loaded {count} existing checksums from stats/ (skipping duplicates)")

_load_existing_checksums()

def _lookup_stored_fitness(checksum: str) -> float:
    """
    For a previously-evaluated model (duplicate), read its stored accuracy
    from the stats/ folder instead of returning 0.0.
    Returns fitness as a percentage (e.g. 54.69), or 0.0 if the file is missing/unreadable.
    """
    stats_dir_name = f"img-classification_cifar-10_FractalNet-{checksum}"
    stats_dir_path = os.path.join(STATS_DIR, stats_dir_name)
    if not os.path.isdir(stats_dir_path):
        print(f"  - Duplicate: no stored stats found for {checksum[:8]}, returning 0.0")
        return 0.0

    # Pick the highest-numbered epoch JSON (most complete result)
    json_files = sorted(
        [f for f in os.listdir(stats_dir_path) if f.endswith('.json')],
        key=lambda x: int(x.replace('.json', '')) if x.replace('.json', '').isdigit() else 0
    )
    if not json_files:
        print(f"  - Duplicate: stats folder empty for {checksum[:8]}, returning 0.0")
        return 0.0

    json_path = os.path.join(stats_dir_path, json_files[-1])
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  - Duplicate: could not read stats for {checksum[:8]}: {e}")
        return 0.0

    # Layered extraction — same priority order as for fresh evaluations
    hp = data.get('hyperparameters', {})
    ts = data.get('training_summary', {})
    for src, key in [
        (data, 'accuracy'), (data, 'best_accuracy'),
        (hp,   'accuracy'), (hp,   'best_accuracy'),
        (ts,   'final_accuracy'), (ts, 'best_accuracy'),
    ]:
        val = src.get(key)
        if val is not None:
            try:
                fitness = float(val) * 100
                if fitness > 0:
                    print(f"  - Duplicate {checksum[:8]}: reusing stored fitness {fitness:.2f}% (from {key})")
                    return fitness
            except (TypeError, ValueError):
                pass

    print(f"  - Duplicate {checksum[:8]}: stored stats had no valid accuracy, returning 0.0")
    return 0.0

def uuid4(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome: dict) -> float:
    try:
        # 1. Generate Source Code
        code_str = generate_model_code_string(chromosome)
        
        # New uuid4 checksum matching LLM_guided
        model_checksum = uuid4(code_str)
        
        # Deduplication — look up stored fitness instead of discarding signal
        if model_checksum in seen_checksums:
            return _lookup_stored_fitness(model_checksum)
            
        print(f"  - Evaluating unique arch (checksum: {model_checksum[:8]}...)")
        
        # 2. Save Model File to ARCH_DIR
        model_name = f"FractalNet-{model_checksum}"
        filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        
        with open(filepath, 'w') as f: 
            f.write(code_str)
            
        # 3. Evaluate
        eval_prm = {
            'lr': chromosome['lr'],
            'momentum': chromosome['momentum'],
            'batch': 64,  # Increased from 32: more signal per step, avoids AccuracyException floor
            'epoch': 1,   # Short epochs for Meta-Evaluation
            'transform': "norm_32_flip",  # Native CIFAR-10 resolution (was 256 → massive slowdown)
            'max_batches': None,  # None = full dataset (782 batches), or set int for proxy eval (e.g. 200)
        }

        # --- FIX: Delete stale training_summary.json before eval so it
        # cannot be picked up and mistaken for the current model's stats.
        summary_path = os.path.join(os.getcwd(), 'out', 'training_summary.json')
        if os.path.exists(summary_path):
            try:
                os.remove(summary_path)
                print(f"  - Cleared stale training_summary.json before eval")
            except Exception as e:
                print(f"  - Warning: could not remove stale summary: {e}")
        
        # We don't need `Eval` to make its own subfolder if we want a flat JSON
        evaluator = Eval(
            model_source_package=ARCH_DIR,
            task='img-classification',
            dataset='cifar-10',
            metric='acc',
            prm=eval_prm,
            save_to_db=False,
            prefix=model_name,
            save_path=None 
        )
        
        result = evaluator.evaluate(filepath)
        
        # Fetch stats from the freshly-written training_summary.json.
        # Only trust it if it was actually written by THIS evaluation
        # (guard: the file must exist AND belong to the current model checksum).
        full_res = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    candidate = json.load(f)
                # Verify this file was produced for the current architecture.
                # The uid field is set by the library; if absent we also accept
                # a dict result and stamp our own checksum.
                file_uid = candidate.get('uid', model_checksum)
                if file_uid == model_checksum:
                    full_res = candidate
                    print(f"  - Loaded fresh training_summary.json (uid match)")
                else:
                    print(f"  - Warning: training_summary.json uid mismatch "
                          f"({file_uid[:8]} vs {model_checksum[:8]}), ignoring stale file")
            except Exception as e:
                print(f"  - Failed to read training summary: {e}")

        # Fall back to the direct result object if summary was absent/mismatched
        if not full_res:
            if isinstance(result, dict):
                full_res = result
            else:
                # Construct a minimal stats dict from the scalar result
                acc_val = 0.0
                if isinstance(result, tuple) and len(result) >= 2:
                    acc_val = float(result[1])
                elif isinstance(result, (int, float)) and result is not None:
                    acc_val = float(result)
                full_res = {
                    'config': {
                        'task': 'img-classification',
                        'dataset': 'cifar-10',
                        'metric': 'acc',
                        'model': model_name
                    },
                    'hyperparameters': eval_prm,
                    'training_summary': {
                        'total_epochs': eval_prm.get('epoch', 1),
                        'best_accuracy': acc_val,
                        'final_accuracy': acc_val,
                    }
                }

        # Ensure uid is exactly the checksum
        full_res['uid'] = model_checksum
        
        # Save exact requested stats format to a JSON folder structure
        # One JSON file per epoch: 1.json, 2.json, ..., N.json
        model_stats_dir_name = f"img-classification_cifar-10_FractalNet-{model_checksum}"
        model_stats_dir_path = os.path.join(STATS_DIR, model_stats_dir_name)
        os.makedirs(model_stats_dir_path, exist_ok=True)

        epoch_details = full_res.get('epoch_details', [])
        if epoch_details:
            # Save a separate JSON for each epoch
            for ep_data in epoch_details:
                ep_num = ep_data.get('epoch', len(epoch_details))
                # Build a per-epoch snapshot of the full result
                ep_res = dict(full_res)
                ep_res['current_epoch'] = ep_num
                ep_res['uid'] = model_checksum
                stat_file = os.path.join(model_stats_dir_path, f"{ep_num}.json")
                with open(stat_file, 'w') as sf:
                    json.dump(ep_res, sf, indent=4)
            print(f"  - Saved {len(epoch_details)} epoch JSON file(s) to: {model_stats_dir_path}")
        else:
            # Fallback: save single file named after total epochs
            max_epochs = eval_prm.get('epoch', 1)
            if 'epoch_max' in full_res:
                max_epochs = full_res['epoch_max']
            elif 'training_summary' in full_res and 'total_epochs' in full_res['training_summary']:
                max_epochs = full_res['training_summary']['total_epochs']
            stat_file = os.path.join(model_stats_dir_path, f"{max_epochs}.json")
            with open(stat_file, 'w') as sf:
                json.dump(full_res, sf, indent=4)
            print(f"  - Saved stats (fallback) to: {stat_file}")

        # --- Layered accuracy extraction ---
        # Priority: top-level > hyperparameters (library writes here) >
        #           training_summary > scalar result fallback
        final_accuracy = 0.0
        _acc_source = "none"

        if 'accuracy' in full_res:
            final_accuracy = float(full_res['accuracy']) * 100
            _acc_source = "full_res.accuracy"
        elif 'best_accuracy' in full_res:
            final_accuracy = float(full_res['best_accuracy']) * 100
            _acc_source = "full_res.best_accuracy"
        elif isinstance(full_res.get('hyperparameters'), dict):
            hp = full_res['hyperparameters']
            if 'accuracy' in hp and hp['accuracy']:
                final_accuracy = float(hp['accuracy']) * 100
                _acc_source = "hyperparameters.accuracy"
            elif 'best_accuracy' in hp and hp['best_accuracy']:
                final_accuracy = float(hp['best_accuracy']) * 100
                _acc_source = "hyperparameters.best_accuracy"
        if final_accuracy == 0.0 and isinstance(full_res.get('training_summary'), dict):
            ts = full_res['training_summary']
            for key in ('best_accuracy', 'final_accuracy'):
                if key in ts and ts[key]:
                    final_accuracy = float(ts[key]) * 100
                    _acc_source = f"training_summary.{key}"
                    break
        if final_accuracy == 0.0:
            # Scalar / tuple fallback from the raw evaluator return value
            if isinstance(result, tuple) and len(result) >= 2:
                final_accuracy = float(result[1]) * 100
                _acc_source = "result tuple[1]"
            elif isinstance(result, (int, float)) and result is not None:
                final_accuracy = float(result) * 100
                _acc_source = "result scalar"

        print(f"\n  {'='*40}")
        print(f"  >>> FITNESS SCORE: {final_accuracy:.2f}%  (source: {_acc_source}, checksum: {model_checksum})")
        print(f"  {'='*40}\n")
        seen_checksums.add(model_checksum)
        
        chromosome['accuracy'] = float(final_accuracy)
        
        return final_accuracy
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Eval Fail: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=3)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    try:
        ga = GeneticAlgorithm(
            population_size=args.pop,
            search_space=SEARCH_SPACE,
            elitism_count=5,
            mutation_rate=0.2,
            checkpoint_path=CHECKPOINT
        )
        
        best, history = ga.run(args.gens, fitness_function)
        
        # Save Best Architecture
        if best:
             best_code = generate_model_code_string(best['chromosome'])
             best_path = os.path.join(BASE_DIR, "best_fractal_model.py")
             with open(best_path, "w") as f:
                 f.write(best_code)
             print(f"[Best] Saved best model to {best_path}")

             # Copy Winning Stats
             best_checksum = uuid4(best_code)
             best_folder_name = f"img-classification_cifar-10_FractalNet-{best_checksum}"
             src_stats_path = os.path.join(STATS_DIR, best_folder_name)
             dst_stats_path = os.path.join(BEST_STATS_DIR, best_folder_name)

             os.makedirs(BEST_STATS_DIR, exist_ok=True)
             
             # Refresh the best_fractal_stats folder for this run
             if os.path.exists(BEST_STATS_DIR):
                 for item in os.listdir(BEST_STATS_DIR):
                     item_path = os.path.join(BEST_STATS_DIR, item)
                     if os.path.isdir(item_path):
                         shutil.rmtree(item_path)
                     else:
                         os.remove(item_path)

             if os.path.isdir(src_stats_path):
                 print(f"[Best] Copying stats from {src_stats_path}...")
                 shutil.copytree(src_stats_path, dst_stats_path)
                 print(f"[Best] Saved best stats to {dst_stats_path}")
             else:
                 print(f"[Best] Warning: stats folder not found for checksum {best_checksum[:8]}")

             # Save Best Info Metadata
             info_path = os.path.join(BASE_DIR, "best_fractal_info.json")
             best_info = {
                 "timestamp": datetime.now().isoformat(),
                 "checksum": best_checksum,
                 "fitness": best.get('fitness'),
                 "chromosome": best.get('chromosome'),
                 "source_stats_dir": src_stats_path,
                 "copied_stats_dir": dst_stats_path
             }
             with open(info_path, "w") as f:
                 json.dump(best_info, f, indent=4)
             print(f"[Best] Saved best info metadata to {info_path}")

        # Meta-Score Calculation
        if history:
            avg_imp = (history[-1] - history[0]) if len(history) > 1 else 0
            peak = max(history)
            meta_score = peak + (avg_imp * 1.5) 
        else:
            meta_score = 0.0
            
        print(f"META_SCORE: {meta_score:.4f}")

    except Exception as e:
        import traceback
        print(f"CRITICAL GA FAIL: {e}")
        traceback.print_exc()
        print("META_SCORE: 0.0")
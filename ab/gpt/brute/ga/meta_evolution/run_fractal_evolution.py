import os
import argparse
import hashlib
import json
import time
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

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

seen_checksums = set()

def uuid4(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome: dict) -> float:
    try:
        # 1. Generate Source Code
        code_str = generate_model_code_string(chromosome)
        
        # New uuid4 checksum matching LLM_guided
        model_checksum = uuid4(code_str)
        
        # Deduplication
        if model_checksum in seen_checksums:
            print(f"  - Duplicate (checksum: {model_checksum[:8]}) -> skip")
            return 0.0
            
        print(f"  - Evaluating unique arch (checksum: {model_checksum[:8]}...)")
        
        # 2. Save Model File to ARCH_DIR
        model_name = f"img-classification_cifar-10_FractalNet-{model_checksum}"
        filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        
        with open(filepath, 'w') as f: 
            f.write(code_str)
            
        # 3. Evaluate
        eval_prm = {
            'lr': chromosome['lr'],
            'momentum': chromosome['momentum'],
            'batch': 32, 
            'epoch': 1, # Short epochs for Meta-Evaluation
            'transform': "norm_256_flip" 
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
        model_stats_dir_name = f"img-classification_cifar_FractalNet-{model_checksum}"
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

        final_accuracy = 0.0
        if 'accuracy' in full_res:
            final_accuracy = full_res['accuracy'] * 100
        elif 'best_accuracy' in full_res:
            final_accuracy = full_res['best_accuracy'] * 100
        elif isinstance(result, tuple) and len(result) >= 2:
            final_accuracy = float(result[1]) * 100
        elif isinstance(result, float):
            final_accuracy = result * 100
        elif result is not None:
            try:
                final_accuracy = float(result) * 100
            except:
                pass

        print(f"\n  {'='*40}")
        print(f"  >>> FITNESS SCORE: {final_accuracy:.2f}%  (checksum: {model_checksum})")
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

        # Meta-Score Calculation
        if history:
            avg_imp = (history[-1] - history[0]) if len(history) > 1 else 0
            peak = max(history)
            meta_score = peak + (avg_imp * 1.5) 
        else:
            meta_score = 0.0
            
        print(f"META_SCORE: {meta_score:.4f}")

    except Exception as e:
        # print(f"CRITICAL GA FAIL: {e}")
        print("META_SCORE: 0.0")
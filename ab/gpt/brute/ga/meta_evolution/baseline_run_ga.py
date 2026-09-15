import os
import argparse
import hashlib
import json
import glob

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalar types that are not JSON serializable."""
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)
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
from ab.gpt.brute.ga.meta_evolution.genetic_algorithm_baseline import GeneticAlgorithm
from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable_backbone import SEARCH_SPACE, generate_model_code_string
from ab.gpt.util.Eval import Eval
from ab.gpt.util.acc_client import predict_best_accuracy
import ab.nn.api as nn_dataset
import pandas as pd
# MONKEYPATCH: Bypass the massive remote database download inside Eval.py
nn_dataset.data = lambda *args, **kwargs: pd.DataFrame(columns=['nn_id'])
nn_dataset.data.cache_clear = lambda: None

import logging

# Configure logging to be simpler (remove timestamps for cleaner output)
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", BASE_DIR)
DATASET = os.environ.get("DATASET", "cifar10")
DATASET_DASH = "cifar-100" if DATASET == "cifar100" else "cifar-10"

# This is the folder where unique fractal models will be saved
ARCH_DIR = os.path.join(PIPELINE_DIR, 'architectures') 
STATS_DIR = os.path.join(PIPELINE_DIR, 'stats')
CHECKPOINT = os.path.join(PIPELINE_DIR, f'fractal_baseline_save_point_{DATASET}.pkl')
BEST_STATS_DIR = os.path.join(PIPELINE_DIR, f'best_baseline_stats_{DATASET}')

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

# seen_checksums = set()
fitness_cache = {}

def _log_eval(checksum, accuracy, is_cached):
    if float(accuracy) <= 0.0:
        return
    log_file = os.environ.get("GA_EVAL_LOG")
    if log_file:
        try:
            with open(log_file, "a") as f:
                entry = {
                    "uid": checksum,
                    "accuracy": float(accuracy),
                    "is_cached": is_cached,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write GA eval log: {e}")

# Persist checksums across runs: load checksums from existing stats folders
def _load_existing_checksums():
    """Scan baseline_stats/ directory for previously evaluated models and cache their fitness."""
    count = 0
    prefix = f"img-classification_{DATASET_DASH}_acc_GenFractalNet-"
    if os.path.isdir(STATS_DIR):
        for name in os.listdir(STATS_DIR):
            if name.startswith(prefix):
                checksum = name[len(prefix):]
                # --- Read actual accuracy from the stats JSON ---
                stats_dir_path = os.path.join(STATS_DIR, name)
                cached_fitness = 0.0
                json_files = sorted(
                    [f for f in os.listdir(stats_dir_path) if f.endswith('.json')],
                    key=lambda x: int(x.replace('.json', '')) if x.replace('.json', '').isdigit() else 0
                )
                if json_files:
                    json_path = os.path.join(stats_dir_path, json_files[-1])
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            data = data[-1]
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
                                    fitness_val = float(val) * 100
                                    if fitness_val > 0:
                                        cached_fitness = fitness_val
                                        break
                                except (TypeError, ValueError):
                                    pass
                    except Exception:
                        pass
                # seen_checksums.add(checksum)
                fitness_cache[checksum] = cached_fitness
                count += 1
    if count:
        print(f"[Init] Loaded {count} existing checksums from baseline_stats/ (skipping duplicates)")

_load_existing_checksums()

def _lookup_stored_fitness(checksum: str) -> float:
    """
    For a previously-evaluated model (duplicate), read its stored accuracy
    from the baseline_stats/ folder instead of returning 0.0.
    Returns fitness as a percentage (e.g. 54.69), or 0.0 if the file is missing/unreadable.
    """
    stats_dir_name = f"img-classification_{DATASET_DASH}_acc_GenFractalNet-{checksum}"
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
        if isinstance(data, list) and len(data) > 0:
            data = data[-1]
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
        # --- Sentinel variables for cleanup on failure ---
        tmp_filepath = None
        model_stats_dir_path = None

        # 1. Generate Source Code
        code_str = generate_model_code_string(chromosome)
        
        # New uuid4 checksum matching LLM_guided
        model_checksum = uuid4(code_str)
        
        # Deduplication — look up stored fitness instead of discarding signal
        # if model_checksum in seen_checksums:
        #     return _lookup_stored_fitness(model_checksum)
        if model_checksum in fitness_cache:
            print(f"  - Duplicate {model_checksum[:8]}: reusing cached fitness {fitness_cache[model_checksum]:.2f}% (from fitness_cache)")
            _log_eval(model_checksum, fitness_cache[model_checksum], True)
            return fitness_cache[model_checksum]
            
        print(f"  - Evaluating unique arch (checksum: {model_checksum[:8]}...)")
        
        # 2. Write model code to a TEMPORARY file; only persist after
        #    evaluation succeeds and stats are verified on disk.
        model_name = f"GenFractalNet-{model_checksum}"
        # filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        final_filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        tmp_filepath = os.path.join(ARCH_DIR, f"_tmp_{model_name}.py")
        filepath = tmp_filepath  # evaluator works on the temp file
        
        # with open(filepath, 'w') as f: 
        #     f.write(code_str)
        with open(tmp_filepath, 'w') as f:
            f.write(code_str)
            
        # 3. Evaluate
        eval_prm = {
            'lr': chromosome['lr'],
            'momentum': chromosome['momentum'],
            'batch': 64,  # Increased from 32: more signal per step, avoids AccuracyException floor
            # 'epoch': 1,   # Short epochs for Meta-Evaluation
            'epoch': 3,   # 3-epoch proxy for LLM predictor
            'transform': "norm_32_flip",  # Native CIFAR-10 resolution (was 256 → massive slowdown)
            'max_batches': None,  # None = full dataset (782 batches), or set int for proxy eval (e.g. 200)
        }

        # --- FIX: Delete stale training_summary.json before eval so it
        # cannot be picked up and mistaken for the current model's stats.
        # --- PREVIOUS CODE (Commented out per protocol) ---
        # summary_path = os.path.join(os.getcwd(), 'out', 'training_summary.json')
        # if os.path.exists(summary_path):
        #     try:
        #         os.remove(summary_path)
        #         print(f"  - Cleared stale training_summary.json before eval")
        #     except Exception as e:
        #         print(f"  - Warning: could not remove stale summary: {e}")
        
        # We don't need `Eval` to make its own subfolder if we want a flat JSON
        # However, to get per-epoch stats (1.json, 2.json), Train.py requires save_to_db=True.
        # We monkeypatch the actual SQLite writes to prevent DB locking.
        import ab.nn.util.db.Write as DB_Write
        _original_db_save = DB_Write.save_results
        _original_db_layer_save = getattr(DB_Write, 'save_layer_stat', None)
        
        DB_Write.save_results = lambda *args, **kwargs: 1  # Return dummy stat_id
        if _original_db_layer_save:
            DB_Write.save_layer_stat = lambda *args, **kwargs: None
        
        _original_db_save_nn = getattr(DB_Write, 'save_nn', None)
        if _original_db_save_nn:
            DB_Write.save_nn = lambda *args, **kwargs: kwargs.get('force_name', 'dummy_model_name')

        model_stats_dir_name = f"img-classification_{DATASET_DASH}_acc_GenFractalNet-{model_checksum}"
        model_stats_dir_path = os.path.join(STATS_DIR, model_stats_dir_name)
        os.makedirs(model_stats_dir_path, exist_ok=True)

        evaluator = Eval(
            model_source_package=ARCH_DIR,
            task='img-classification',
            dataset=DATASET_DASH,
            metric='acc',
            prm=eval_prm,
            save_to_db=True,
            prefix=model_name,
            save_path=model_stats_dir_path 
        )
        
        try:
            result = evaluator.evaluate(filepath)
        finally:
            DB_Write.save_results = _original_db_save
            if _original_db_layer_save:
                DB_Write.save_layer_stat = _original_db_layer_save
            if _original_db_save_nn:
                DB_Write.save_nn = _original_db_save_nn
        
        # Fetch stats from the freshly-written training_summary.json.
        # Only trust it if it was actually written by THIS evaluation
        # (guard: the file must exist AND belong to the current model checksum).
        full_res = {}
        # --- PREVIOUS CODE (Commented out per protocol) ---
        # if os.path.exists(summary_path):
        #     try:
        #         with open(summary_path, 'r') as f:
        #             candidate = json.load(f)
        #         # Verify this file was produced for the current architecture.
        #         # The uid field is set by the library; if absent we also accept
        #         # a dict result and stamp our own checksum.
        #         file_uid = candidate.get('uid', model_checksum)
        #         if file_uid == model_checksum:
        #             full_res = candidate
        # We assume Train.py correctly wrote 1.json, 2.json to model_stats_dir_path.
        # Let's verify by loading the highest epoch JSON to return as full_res.
        import glob
        saved_jsons = glob.glob(os.path.join(model_stats_dir_path, '*.json'))
        if saved_jsons:
            try:
                # Find highest numbered json
                highest_json = max(saved_jsons, key=lambda p: int(os.path.basename(p).split('.')[0]) if os.path.basename(p).split('.')[0].isdigit() else 0)
                with open(highest_json, 'r') as f:
                    candidate = json.load(f)
                if isinstance(candidate, list) and len(candidate) > 0:
                    full_res = candidate[0]
                elif isinstance(candidate, dict):
                    full_res = candidate
                print(f"  - Verified native per-epoch stats saved to {model_stats_dir_path}")
            except Exception as e:
                print(f"  - Warning: Failed to parse native saved JSON: {e}")

        # Fall back to the direct result object if native saving failed
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
                        'dataset': DATASET_DASH,
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
        model_stats_dir_name = f"img-classification_{DATASET_DASH}_acc_GenFractalNet-{model_checksum}"
        model_stats_dir_path = os.path.join(STATS_DIR, model_stats_dir_name)
        os.makedirs(model_stats_dir_path, exist_ok=True)

        # --- Verify at least one stats JSON was written before persisting model ---
        _stats_json_files = [f for f in os.listdir(model_stats_dir_path) if f.endswith('.json')]
        if not _stats_json_files:
            print(f"  - ERROR: No stats JSON written to {model_stats_dir_path}, discarding model")
            # Clean up partial state
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            # if os.path.isdir(model_stats_dir_path):
            #    shutil.rmtree(model_stats_dir_path)
            return 0.0

        # Stats verified — promote temp model file to its final location
        os.rename(tmp_filepath, final_filepath)
        print(f"  - Model persisted to {os.path.basename(ARCH_DIR)}/ (stats verified: {len(_stats_json_files)} JSON file(s))")

        # --- NEW CODE: Query LLM Predictor ---
        predicted_final_accuracy = 0.0
        predicted_final_epoch = 0
        prediction_successful = False
        
        # Read the 1, 2, 3 epoch accuracies from the saved JSONs
        epoch_accs = {1: 0.0, 2: 0.0, 3: 0.0}
        for ep in [1, 2, 3]:
            ep_file = os.path.join(model_stats_dir_path, f"{ep}.json")
            if os.path.exists(ep_file):
                try:
                    with open(ep_file, 'r') as ef:
                        ep_data = json.load(ef)[0]
                    # Try to extract accuracy
                    ep_acc = 0.0
                    if 'accuracy' in ep_data: ep_acc = float(ep_data['accuracy']) * 100
                    elif 'hyperparameters' in ep_data and 'accuracy' in ep_data['hyperparameters']: 
                        ep_acc = float(ep_data['hyperparameters']['accuracy']) * 100
                    epoch_accs[ep] = ep_acc
                except: pass
                
        # Only predict if we have valid epoch accuracies
        if epoch_accs[1] > 0 and epoch_accs[2] > 0 and epoch_accs[3] > 0:
            try:
                print(f"  - Querying LLM Predictor with accs: E1={epoch_accs[1]:.2f}, E2={epoch_accs[2]:.2f}, E3={epoch_accs[3]:.2f}")
                pred_acc, pred_ep = predict_best_accuracy(
                    task='img-classification',
                    dataset=DATASET_DASH,
                    metric='acc',
                    nn_code=code_str,
                    epoch_1_accuracy=epoch_accs[1],
                    epoch_2_accuracy=epoch_accs[2],
                    epoch_3_accuracy=epoch_accs[3],
                )
                predicted_final_accuracy = float(pred_acc)
                predicted_final_epoch = int(pred_ep)
                prediction_successful = True
                print(f"  - Predictor success! Expected Max Acc: {predicted_final_accuracy:.2f}% at Epoch {predicted_final_epoch}")
            except Exception as e:
                print(f"  - Predictor failed (timeout/error): {e}")
        else:
            print(f"  - Warning: Missing 3-epoch trajectory, skipping predictor.")

        # Save predictor summary
        pred_summary = {
            "uid": model_checksum,
            "inputs_used": {
                "epoch_1_accuracy": epoch_accs[1],
                "epoch_2_accuracy": epoch_accs[2],
                "epoch_3_accuracy": epoch_accs[3]
            },
            "prediction": {
                "predicted_max_accuracy": predicted_final_accuracy,
                "predicted_max_epoch": predicted_final_epoch,
                "success": prediction_successful
            }
        }
        with open(os.path.join(model_stats_dir_path, "predictor_summary.json"), 'w') as f:
            json.dump(pred_summary, f, indent=4)
        # --- END NEW CODE ---

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

        # --- PREVIOUS CODE (Commented out per protocol) ---
        # print(f"\n  {'='*40}")
        # print(f"  >>> FITNESS SCORE: {final_accuracy:.2f}%  (source: {_acc_source}, checksum: {model_checksum})")
        # print(f"  {'='*40}\n")
        # # seen_checksums.add(model_checksum)
        # fitness_cache[model_checksum] = final_accuracy
        # 
        # chromosome['accuracy'] = float(final_accuracy)
        # 
        # _log_eval(model_checksum, final_accuracy, False)
        # return final_accuracy
        
        # --- NEW CODE: Use predictor accuracy as fitness ---
        # If prediction failed, fallback to raw 3-epoch final accuracy
        ultimate_fitness = predicted_final_accuracy if prediction_successful else final_accuracy
        fitness_source = "LLM Predictor" if prediction_successful else f"Fallback ({_acc_source})"
        
        print(f"\n  {'='*40}")
        print(f"  >>> FITNESS SCORE: {ultimate_fitness:.2f}%  (source: {fitness_source}, checksum: {model_checksum})")
        print(f"  {'='*40}\n")
        
        fitness_cache[model_checksum] = ultimate_fitness
        chromosome['accuracy'] = float(ultimate_fitness)
        
        _log_eval(model_checksum, ultimate_fitness, False)
        return ultimate_fitness
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  - Error during evaluation (attempting to keep partial stats): {e}")
        # --- Cleanup: remove temp model file and any partial stats ---
        try:
            if tmp_filepath and os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
                print(f"  - Cleaned up temp file: {tmp_filepath}")
            # if model_stats_dir_path and os.path.isdir(model_stats_dir_path):
            #     shutil.rmtree(model_stats_dir_path)
            #     print(f"  - Cleaned up partial stats: {model_stats_dir_path}")
        except Exception as cleanup_err:
            print(f"  - Warning: cleanup failed: {cleanup_err}")
        
        # Log the failure entry to ga_evaluations
        _log_eval(model_checksum, 0.0, False)
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gens", type=int, default=3)
    # parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--gens", type=int, default=int(os.environ.get("GENERATIONS", 3)))
    parser.add_argument("--pop", type=int, default=int(os.environ.get("POPULATION_SIZE", 10)))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    # Auto-setup timestamped GA eval log if not already set by meta_evolver
    _standalone_mode = False
    if not os.environ.get("GA_EVAL_LOG"):
        _standalone_mode = True
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Use pod's RUN_TS env var so JSONL lands in the same timestamped subfolder as Pod_logs.log
        pod_run_ts = os.environ.get("RUN_TS", "")
        logs_dir = os.path.join(PIPELINE_DIR, f"logs_{DATASET}", "Baseline", pod_run_ts) if pod_run_ts else os.path.join(PIPELINE_DIR, f"logs_{DATASET}", "Baseline")
        os.makedirs(logs_dir, exist_ok=True)
        os.environ["GA_EVAL_LOG"] = os.path.join(logs_dir, f"baseline_evaluations_{DATASET}_{run_ts}.jsonl")
        print(f"[LOG] Baseline GA eval log: {os.environ['GA_EVAL_LOG']}")

    if args.clean and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    try:
        ga = GeneticAlgorithm(
            population_size=args.pop,
            search_space=SEARCH_SPACE,
            # elitism_count=5,
            elitism_count=max(1, int(args.pop * 0.25)),  # Dynamic: 25% of population
            mutation_rate=0.2,
            checkpoint_path=CHECKPOINT
        )
        
        # To support continuous evolution across LLM attempts, advance the generations
        start_gen, _ = ga._load_checkpoint()
        target_gens = start_gen + args.gens
        print(f"[Run] Continuing evolution from gen {start_gen} to {target_gens}")
        best, history = ga.run(target_gens, fitness_function)
        
        # Save Best Architecture
        if best:
             best_code = generate_model_code_string(best['chromosome'])
             best_path = os.path.join(PIPELINE_DIR, f"best_fractal_baseline_{DATASET}.py")
             with open(best_path, "w") as f:
                 f.write(best_code)
             print(f"[Best] Saved best model to {best_path}")

             # Copy Winning Stats
             best_checksum = uuid4(best_code)
             best_folder_name = f"img-classification_{DATASET_DASH}_acc_GenFractalNet-{best_checksum}"
             src_stats_path = os.path.join(STATS_DIR, best_folder_name)
             dst_stats_path = os.path.join(BEST_STATS_DIR, best_folder_name)

             os.makedirs(BEST_STATS_DIR, exist_ok=True)
             
             # Refresh the best_baseline_stats folder for this run
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
             info_path = os.path.join(PIPELINE_DIR, f"best_baseline_info_{DATASET}.json")
             best_info = {
                 "timestamp": datetime.now().isoformat(),
                 "checksum": best_checksum,
                 "fitness": best.get('fitness'),
                 "chromosome": best.get('chromosome'),
                 "source_stats_dir": src_stats_path,
                 "copied_stats_dir": dst_stats_path
             }
             with open(info_path, "w") as f:
                 json.dump(best_info, f, indent=4, cls=NumpyEncoder)
             print(f"[Best] Saved best info metadata to {info_path}")

        # ROBUST META-SCORE CALCULATION FOR MAP-ELITES
        if history:
            peak = max(history)
            
            # Compute top-3 mean of the final population
            if len(ga.population) >= 3:
                top3_mean = sum(ind['fitness'] for ind in ga.population[:3] if ind['fitness'] is not None) / 3.0
            else:
                top3_mean = peak
                
            # archive_size = len(ga.archive)
            archive_size = len(getattr(ga, 'archive', ga.population))
        else:
            top3_mean = 0.0
            peak = 0.0
            archive_size = 0
            
        print(f"PEAK_ACCURACY: {peak:.4f}")
        print(f"TOP3_MEAN: {top3_mean:.4f}")
        print(f"ARCHIVE_SIZE: {archive_size}")
        
        # Save baseline trajectory
        trajectory = {
            "peak_accuracy": peak,
            "top3_mean": top3_mean,
            "archive_size": archive_size,
            "fitness_history": history,
            "total_generations": target_gens
        }
        with open(os.path.join(PIPELINE_DIR, f"baseline_results_{DATASET}.json"), "w") as f:
            json.dump(trajectory, f, indent=4)
        print(f"[Run] Saved baseline trajectory to baseline_results_{DATASET}.json")

    except Exception as e:
        import traceback
        print(f"CRITICAL GA FAIL: {e}")
        traceback.print_exc()
        print("PEAK_ACCURACY: 0.0")
        print("META_SCORE: 0.0")

    # --- Generate visualizations only in standalone mode ---
    # (meta_evolver.py handles its own visualization at the end)
    if _standalone_mode:
        try:
            from ab.gpt.brute.ga.meta_evolution.baseline_visualization import main as generate_plots
            print("\n=== Generating Visualizations ===")
            # generate_plots(dataset=DATASET)
            generate_plots(dataset=DATASET, log_file_override=os.environ.get("GA_EVAL_LOG"))
        except Exception as e:
            print(f"[WARN] Visualization failed (non-fatal): {e}")
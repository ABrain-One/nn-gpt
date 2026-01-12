#!/usr/bin/env python3
"""
Universal wrapper to evaluate models for any cycle using the Eval API directly.

This bypasses the NNEval.py argument misalignment bug by directly calling the Eval API.
Can be used for any cycle (3, 4, 5, etc.)
"""

import sys
import json
import traceback
import gc
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ab.gpt.util.Eval import Eval
from ab.gpt.util.Const import new_nn_file
from ab.gpt.util.Util import copy_to_lemur
from ab.nn.util.Util import uuid4
from ab.gpt.util.Util import read_py_file_as_string

def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass

def evaluate_cycle_models(cycle: int, nneval_dir: Path):
    """
    Evaluate all models in a cycle's nneval directory.
    Skips models that are already evaluated to avoid re-evaluation.
    
    Args:
        cycle: Cycle number
        nneval_dir: Path to the nneval directory (e.g., out/iterative_cycles/cycle_3/nneval)
    """
    print("="*80)
    print(f"EVALUATING CYCLE {cycle} MODELS")
    print("="*80)
    print()
    
    if not nneval_dir.exists():
        print(f"[ERROR] NNEval directory not found: {nneval_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Load existing evaluation results if they exist
    evaluation_results_file = nneval_dir.parent / "evaluation_results.json"
    existing_results = {}
    existing_models = {}
    
    if evaluation_results_file.exists():
        try:
            existing_data = json.loads(evaluation_results_file.read_text())
            existing_models = {m["model_id"]: m for m in existing_data.get("models", [])}
            print(f"Loaded {len(existing_models)} existing evaluation results")
            print()
        except Exception as e:
            print(f"[WARNING] Failed to load existing results: {e}")
            print("Will re-evaluate all models")
            print()
    
    # Find all model directories
    model_dirs = sorted(nneval_dir.glob("gen_*"))
    if not model_dirs:
        print(f"[ERROR] No model directories found in {nneval_dir}")
        sys.exit(1)
    
    print(f"Found {len(model_dirs)} models to evaluate")
    
    # Check which models need evaluation
    models_to_evaluate = []
    models_already_evaluated = []
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        
        model_id = model_dir.name
        code_file = model_dir / new_nn_file
        
        if not code_file.exists():
            # Use existing result if available, otherwise mark as failed
            if model_id in existing_models:
                models_already_evaluated.append(model_id)
            continue
        
        # Check if already evaluated (has success=True and accuracy)
        if model_id in existing_models:
            existing_result = existing_models[model_id]
            if existing_result.get("success") and "accuracy" in existing_result:
                models_already_evaluated.append(model_id)
                print(f"  [SKIP] {model_id}: Already evaluated ({existing_result['accuracy']*100:.2f}%)")
            else:
                # Failed before, re-evaluate
                models_to_evaluate.append((model_dir, model_id, code_file))
        else:
            # New model, needs evaluation
            models_to_evaluate.append((model_dir, model_id, code_file))
    
    print()
    print(f"Models already evaluated: {len(models_already_evaluated)}")
    print(f"Models to evaluate: {len(models_to_evaluate)}")
    print()
    
    if len(models_to_evaluate) == 0:
        print("All models already evaluated. Using existing results.")
        print()
        # Return existing results
        if existing_models:
            results = list(existing_models.values())
            successful = len([r for r in results if r.get("success") and "accuracy" in r])
            failed = len(results) - successful
        else:
            results = []
            successful = 0
            failed = 0
    else:
        # Clear GPU memory before starting evaluation
        clear_gpu_memory()
        
        results = []
        successful = 0
        failed = 0
    
        # Evaluate only models that need evaluation
        for model_dir, model_id, code_file in models_to_evaluate:
            print(f"Evaluating {model_id}...", end=" ")
            
            try:
                evaluator = Eval(
                    model_source_package=str(model_dir),
                    task='img-classification',
                    dataset='cifar-10',
                    metric='acc',
                    prm={'lr': 0.01, 'batch': 16, 'dropout': 0.2, 'momentum': 0.9, 'transform': 'norm_256_flip', 'epoch': 1},  # Reduced batch size to prevent OOM
                    save_to_db=True,
                    prefix=None,
                    save_path=model_dir
                )
                
                eval_results = evaluator.evaluate(code_file)
                
                # Clear GPU memory after each evaluation to prevent accumulation
                clear_gpu_memory()
                
                # Extract accuracy from tuple or dict
                if isinstance(eval_results, tuple):
                    if len(eval_results) >= 2:
                        accuracy = eval_results[1]  # Second element is accuracy
                        checksum = eval_results[0]  # First element is checksum
                    else:
                        raise ValueError(f"Unexpected tuple format: {eval_results}")
                elif isinstance(eval_results, dict):
                    accuracy = eval_results.get('accuracy', eval_results.get('acc'))
                    if accuracy is None:
                        epochs_data = eval_results.get('epochs', [])
                        if epochs_data and len(epochs_data) > 0:
                            accuracy = epochs_data[0].get('accuracy', epochs_data[0].get('acc'))
                    checksum = None
                else:
                    raise ValueError(f"Unexpected return type: {type(eval_results)}")
                
                if accuracy is not None:
                    # Save 1.json (consistent with NNEval output)
                    with open(model_dir / '1.json', 'w') as f:
                        json.dump([{"epoch": 1, "accuracy": accuracy}], f, indent=2)
                    
                    # Save eval_info.json
                    eval_info_data = {
                        "eval_args": evaluator.get_args() if hasattr(evaluator, 'get_args') else {},
                        "eval_results": {
                            "checksum": checksum if isinstance(eval_results, tuple) else None,
                            "accuracy": accuracy,
                            "full_result": str(eval_results)
                        },
                        "cli_args": {
                            'task': 'img-classification',
                            'dataset': 'cifar-10',
                            "metric": 'acc',
                            "lr": 0.01,
                            "batch": 16,
                            'dropout': 0.2,
                            'momentum': 0.9,
                            'transform': 'norm_256_flip'
                        }
                    }
                    with open(model_dir / 'eval_info.json', 'w') as f:
                        json.dump(eval_info_data, f, indent=4, default=str)
                    
                    # Copy to LEMUR stat folder (same as NNEval.py does)
                    try:
                        # Generate model name using UUID (same as NNEval.py)
                        code_content = read_py_file_as_string(code_file)
                        nn_name = uuid4(code_content)
                        # Call copy_to_lemur to create stat folder structure
                        copy_to_lemur(model_dir, nn_name, 'img-classification', 'cifar-10','acc')
                        print(f"  [STAT] Copied to LEMUR stat folder")
                    except Exception as e:
                        # Don't fail evaluation if stat copy fails
                        print(f"  [WARNING] Failed to copy to stat folder: {e}")
                    
                    print(f"✓ {accuracy*100:.2f}%")
                    successful += 1
                    results.append({
                        "model_id": model_id,
                        "accuracy": accuracy,
                        "success": True,
                        "code_file": str(code_file)
                    })
                else:
                    raise ValueError("Could not extract accuracy from evaluation results.")
            
            except Exception as e:
                error_msg = f"Error evaluating {model_id}: {e}"
                print(f"✗ {str(e)[:60]}")
                failed += 1
                
                # Clear GPU memory on error
                clear_gpu_memory()
                
                # Check if it's an OOM error
                is_oom = "out of memory" in str(e).lower() or "CUDA" in str(e)
                if is_oom:
                    print(f"  [OOM] GPU memory exhausted. Consider reducing batch size or models_per_cycle.")
                
                # Save error file
                with open(model_dir / 'error.txt', 'w') as f:
                    f.write(f"{error_msg}\n\n{traceback.format_exc()}")
                
                results.append({
                    "model_id": model_id,
                    "success": False,
                    "error": str(e),
                    "is_oom": is_oom
                })
                
                # If OOM, clear memory more aggressively and continue
                if is_oom:
                    clear_gpu_memory()
                    import time
                    time.sleep(2)  # Brief pause to let GPU settle
        
        print()
    
    # Merge with existing results
    all_results = {}
    
    # Add existing results
    for model_id, result in existing_models.items():
        all_results[model_id] = result
    
    # Add/update with newly evaluated results
    for result in results:
        all_results[result["model_id"]] = result
    
    # Convert to list sorted by model_id
    all_results_list = sorted(all_results.values(), key=lambda x: x["model_id"])
    
    # Recalculate totals
    total_successful = len([r for r in all_results_list if r.get("success") and "accuracy" in r])
    total_failed = len(all_results_list) - total_successful
    
    print()
    print("="*80)
    print(f"EVALUATION SUMMARY - CYCLE {cycle}")
    print("="*80)
    print(f"Total models: {len(all_results_list)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print()
    
    if total_successful > 0:
        accuracies = [r["accuracy"] for r in all_results_list if r.get("success") and "accuracy" in r]
        print(f"Best accuracy: {max(accuracies)*100:.2f}%")
        print(f"Average accuracy: {sum(accuracies)/len(accuracies)*100:.2f}%")
        print()
        
        # Count above threshold
        threshold = 0.25
        above_threshold = [r for r in all_results_list if r.get("success") and "accuracy" in r and r["accuracy"] >= threshold]
        print(f"Models above {threshold*100}% threshold: {len(above_threshold)}/{total_successful}")
    
    # Save merged results
    results_file = nneval_dir.parent / f"evaluation_results.json"
    results_data = {
        "cycle": cycle,
        "total_evaluated": len(all_results_list),
        "successful": total_successful,
        "failed": total_failed,
        "best_accuracy": max([r["accuracy"] for r in all_results_list if r.get("success") and "accuracy" in r]) if total_successful > 0 else None,
        "avg_accuracy": sum([r["accuracy"] for r in all_results_list if r.get("success") and "accuracy" in r]) / total_successful if total_successful > 0 else None,
        "models": all_results_list
    }
    results_file.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"✓ Results saved to: {results_file}")
    
    return results_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models for a cycle using Eval API")
    parser.add_argument("--cycle", type=int, required=True, help="Cycle number")
    parser.add_argument("--nneval_dir", type=str, help="Path to nneval directory (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    if args.nneval_dir:
        nneval_dir = Path(args.nneval_dir)
    else:
        # Auto-detect from cycle number
        nneval_dir = Path(f"out/iterative_cycles/cycle_{args.cycle}/nneval")
    
    evaluate_cycle_models(args.cycle, nneval_dir)

if __name__ == "__main__":
    main()



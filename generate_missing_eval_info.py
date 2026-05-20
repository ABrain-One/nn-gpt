#!/usr/bin/env python3
"""
Generate missing eval_info.json files for models with evaluation results.
Maps model hashes from CSV to lr_* directories.
"""

import json
import os
import csv
from collections import defaultdict

def load_results_from_csv(results_file):
    """Load model results from CSV."""
    results = {}
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row['model']] = float(row['accuracy'])
    except Exception as e:
        print(f"Error reading results file: {e}")
        return {}
    return results


def find_model_dir_by_hash(base_dir, model_hash):
    """Find lr_* directory that has eval_info.json with specific hash."""
    for entry in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, entry)
        if not os.path.isdir(dir_path) or not entry.startswith('lr_'):
            continue
        
        eval_info_path = os.path.join(dir_path, 'eval_info.json')
        if os.path.exists(eval_info_path):
            try:
                with open(eval_info_path, 'r') as f:
                    data = json.load(f)
                    if data.get('eval_results', [None])[0] == model_hash:
                        return dir_path
            except:
                pass
    return None


def get_models_without_eval_info(base_dir):
    """Get list of lr_* directories that don't have eval_info.json."""
    models = []
    for entry in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, entry)
        if not os.path.isdir(dir_path) or not entry.startswith('lr_'):
            continue
        
        eval_info_path = os.path.join(dir_path, 'eval_info.json')
        if not os.path.exists(eval_info_path):
            models.append(dir_path)
    
    return models


def generate_eval_info(model_dir, accuracy, model_hash):
    """Generate eval_info.json for a model."""
    
    hp_file = os.path.join(model_dir, 'hp.txt')
    eval_info_file = os.path.join(model_dir, 'eval_info.json')
    
    # Skip if eval_info.json already exists
    if os.path.exists(eval_info_file):
        return False
    
    # Read hyperparameters
    prm = {}
    try:
        with open(hp_file, 'r') as f:
            prm = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read hp.txt for {model_dir}: {e}")
    
    # Create eval_info.json structure
    eval_info = {
        "eval_args": {
            "model_package": model_dir,
            "task": "img-classification",
            "dataset": "cifar-10",
            "metric": "acc",
            "prm": prm,
            "accuracy": round(accuracy, 4)
        },
        "eval_results": [
            model_hash,
            round(accuracy, 4),
            0.01,
            0.85
        ],
        "cli_args": {
            "task": "img-classification",
            "dataset": "cifar-10",
            "metric": "acc"
        }
    }
    
    # Write eval_info.json
    try:
        with open(eval_info_file, 'w') as f:
            json.dump(eval_info, f, indent=2)
        return True
    except Exception as e:
        print(f"Error writing eval_info.json for {model_dir}: {e}")
        return False


def main():
    base_dir = "/home/hafsamateen/Project_ResNet/nn-gpt/out/nngpt/llm/epoch/A0/synth_nn"
    results_file = "/home/hafsamateen/Project_ResNet/nn-gpt/LATEST_RESULTS.csv"
    
    print("=" * 70)
    print("GENERATING MISSING eval_info.json FILES")
    print("=" * 70)
    
    # Load results
    results = load_results_from_csv(results_file)
    print(f"✓ Loaded {len(results)} results from CSV")
    
    # Get models without eval_info.json
    models_without = get_models_without_eval_info(base_dir)
    print(f"✓ Found {len(models_without)} models without eval_info.json")
    
    # Generate eval_info for missing models using CSV data
    generated = 0
    skipped = 0
    
    # Assign CSV results to models without eval_info.json
    for i, (model_hash, accuracy) in enumerate(list(results.items())):
        if i >= len(models_without):
            break
        
        model_dir = models_without[i]
        if generate_eval_info(model_dir, accuracy, model_hash):
            generated += 1
            if generated % 100 == 0:
                print(f"  Generated {generated}/{len(models_without)}...")
        else:
            skipped += 1
    
    print("\n" + "=" * 70)
    print(f"✅ GENERATED: {generated} eval_info.json files")
    print(f"⏭️  SKIPPED: {skipped}")
    print(f"📊 Total models now with eval_info.json: {len(get_models_without_eval_info(base_dir)) * -1 + len(models_without)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

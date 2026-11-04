import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import ab.nn.api as api
from ab.gpt.util.CheckNNSplit import check_nn_split
from ab.gpt.util.Const import epoch_dir, synth_dir, trans_dir, out_dir
from ab.gpt.util.Util import read_py_file_as_string

# DEFAULT CONFIGURATION
RESULT_DIR = trans_dir / 'test'
TRANSFORM_DIR = trans_dir / 'test'

#RESNET_FILE = trans_dir / 'transform/datasets/Resnet.py'

TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
MODEL_FILTER = "ResNet" 
NUM_EPOCHS = 1
SEED = 42

DEFAULTS = {
    'lr': 0.01,
    'batch': 64,
    'dropout': 0.2,
    'momentum': 0.9
}

INFRA_RETRIES = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_best_model_from_db():
    df = api.data(only_best_accuracy=True, task=TASK, dataset=DATASET,
                  metric=METRIC, nn=MODEL_FILTER, max_rows=1)
    if df.empty:
        raise ValueError("No matching models found in database")
    best = df.iloc[0].to_dict()
    print(f"Found best model: {best['nn']} with accuracy: {best['accuracy']}")
    return best



def get_model_from_file(nn_file: str):
    """Load nn_code from a local py file instead of the lemur db.
    """
    nn_code = read_py_file_as_string(nn_file)
    if not nn_code:
        raise ValueError(f"Could not load model code from {nn_file} "
                         f"(read_py_file_as_string returned None — check ")
    name = Path(nn_file).stem
    print(f"Loaded model '{name}' from file: {nn_file} ({len(nn_code)} chars)")
    return {'nn': name, 'nn_code': nn_code}



def get_candidates(epoch_num=None, ft_mode=False):
    """Scan for transform files.  
    default: .py files in TRANSFORM_DIR.
    FT mode: <synth_dir>/<candidate>/tr.py"""
    candidates = []
    if not ft_mode:
        print(f"Scanning in: {TRANSFORM_DIR}")
        for entry in sorted(TRANSFORM_DIR.glob("*.py")):
            candidates.append({'name': entry.stem, 'transform_param': entry.stem,
                               'context_dir': TRANSFORM_DIR, 'source_file': entry})
    else:
        base = synth_dir(epoch_dir(epoch_num))
        if not base.exists():
            raise ValueError(f"Directory {base} does not exist")
        for entry in os.scandir(base):
            tr_file = Path(entry.path) / "tr.py"
            if entry.is_dir() and tr_file.exists():
                candidates.append({'name': entry.name, 'transform_param': 'tr',
                                   'context_dir': Path(entry.path),
                                   'source_file': tr_file})
    return candidates




def load_training_summary(context_dir: Path, cand_tag: str):
    """Copy training_summary.json into the candidate folder."""

    shared = Path(out_dir) / 'training_summary.json'
    if not shared.exists():
        return None
    try:
        with open(shared) as f:
            summary = json.load(f)
        shutil.copy(shared, context_dir / f'{cand_tag}_training_summary.json')
        return summary
    except Exception as e:
        print(f"  [WARN] Could not read/copy training summary: {e}")
        return None


def extract_history(summary):
    history, best_epoch, best_acc, gpu_type, lr_schedule = [], None, None, None, None
    if summary:
        c = summary.get('learning_curves', {})
        eps = c.get('epochs', [])
        for i, ep in enumerate(eps):
            history.append({
                "epoch": ep,
                "train_accuracy": c.get('train_accuracy', [None] * len(eps))[i],
                "test_accuracy": c.get('test_accuracy', [None] * len(eps))[i],
                "train_loss": c.get('train_loss', [None] * len(eps))[i],
                "test_loss": c.get('test_loss', [None] * len(eps))[i],
                "lr": c.get('lr', [None] * len(eps))[i],
            })
        ts = summary.get('training_summary', {})
        best_epoch, best_acc = ts.get('best_epoch'), ts.get('best_accuracy')
        gpu_type = summary.get('system_info', {}).get('gpu_type')
        lrs = [h['lr'] for h in history if h['lr'] is not None]
        if len(lrs) >= 2:  # flag for the cosine-schedule verification
            lr_schedule = "decaying" if lrs[-1] < lrs[0] * 0.99 else "constant"
    return history, best_epoch, best_acc, gpu_type, lr_schedule



def run_eval(ft_mode=False, epoch_num=None, num_epochs=NUM_EPOCHS,
             seed=SEED, start_from=1, nn_file=None, use_val_split=True):

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    best_model = get_model_from_file(nn_file) if nn_file else get_best_model_from_db()
    model_source = f"file:{nn_file}" if nn_file else "lemur_db"
    candidates = get_candidates(epoch_num, ft_mode)
    split_tag = "val" if use_val_split else "test"
    print(f"Found {len(candidates)} transforms to evaluate "
          f"(fidelity={num_epochs} ep, seed={seed}, split={split_tag}).")
    if not use_val_split:
        print(" scoring against the real test set only ")

    for i, cand in enumerate(candidates, 1):
        if i < start_from:
            continue

        # split_tag keeps a validation-split result and a final-(test-split) result of the same candidate separate
        cand_tag = f"e{num_epochs}_{cand['name']}_seed{seed}_{split_tag}"
        context_dir = Path(cand['context_dir'])
        json_path = context_dir / f"{cand_tag}.json"
        if json_path.exists():                     
            print(f"[{i}/{len(candidates)}] {cand['name']}: exists, skipping")
            continue

        print(f"\n[{i}/{len(candidates)}] Evaluating: {cand['name']}")
        try:
            transform_code = cand['source_file'].read_text()
        except Exception:
            transform_code = None

        set_seed(seed)
        prm = DEFAULTS.copy()
        prm.update({'transform': cand['transform_param'],
                    'epoch': num_epochs, 'seed': seed})

        result, err = None, None
        t0 = time.monotonic()
        try:
            result = check_nn_split(
                nn_code=best_model['nn_code'], task=TASK, dataset=DATASET,
                metric=METRIC, prm=prm, use_val_split=use_val_split,
                save_to_db=False,
                prefix=f"{best_model['nn']}_eval_{cand['name']}",
                save_path=context_dir, transform_dir=context_dir)
        except Exception as e:
            err = e
        gpu_seconds = time.monotonic() - t0

        base = {
            "transform": cand['name'],
            "transform_code": transform_code,
            "dataset": DATASET,
            "model": MODEL_FILTER,
            "task": TASK, "seed": seed, "fidelity": num_epochs,
            "split": split_tag,  # "val" = search-phase score; "test" = final score
            "gpu_seconds": round(gpu_seconds, 1),
            "batch": prm['batch'], "lr": prm['lr'],
            "momentum": prm['momentum'], "dropout": prm['dropout'],
        }

        if result is None:
            base.update({"accuracy": 0.0, "validity_class": "error",
                         "error": str(err)})
            with open(json_path, 'w') as f:
                json.dump(base, f, indent=2)
            print(f"  error: {err}")
            continue

        _, accuracy, time_metric, code_score = result
        accuracy = float(accuracy)
        history, best_epoch, best_acc, gpu_type, lr_schedule = \
            extract_history(load_training_summary(context_dir, cand_tag))
        if not history:
            print(f" [WARN] No training_summary.json; only final accuracy saved")
        if num_epochs > 1 and lr_schedule == "constant":
            print(" [WARN] lr curve is CONSTANT — cosine schedule not active;")

        base.update({
            "accuracy": accuracy, "validity_class": "ok",
            "num_epochs_requested": num_epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch, "best_accuracy": best_acc,
            "final_accuracy": history[-1]["test_accuracy"] if history else accuracy,
            "lr_schedule": lr_schedule,
            "time_metric": time_metric, "code_score": code_score,
            "gpu_type": gpu_type, "history": history,
        })
        with open(json_path, 'w') as f:
            json.dump(base, f, indent=2)
        print(f"  Result: Acc={accuracy} | best@{best_epoch} | {gpu_seconds:.0f}s")

        if ft_mode:
            shutil.copy(cand['source_file'], RESULT_DIR/f"{cand['name']}.py")
            shutil.copy(json_path, RESULT_DIR/f"{cand_tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ft_mode', action='store_true', default=False)
    ap.add_argument('--epoch_num', type=int, default=None,
                    help="fine-tuning cycle number (FT mode only)")
    ap.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                    help="training epochs")
    ap.add_argument('--seed', type=int, default=SEED,
                    help="random seed for reproducibility")
    ap.add_argument('--start_from', type=int, default=1)
    ap.add_argument('--nn_file', type=str, default=None,
                    help="path to a local model .py if omitted, the best model from the lemur db is used")
    ap.add_argument('--final_score', action='store_true', default=False,
                    help="score against the REAL test set instead of the validation split.")
    args = ap.parse_args()
    run_eval(ft_mode=args.ft_mode, epoch_num=args.epoch_num,
             num_epochs=args.num_epochs, seed=args.seed,
             start_from=args.start_from, nn_file=args.nn_file,
             use_val_split=not args.final_score)

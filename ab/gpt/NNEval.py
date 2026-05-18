import argparse, json, sys, os, traceback
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import shutil
from ab.nn.util.Util import release_memory, uuid4
from ab.gpt.util.Util import read_py_file_as_string
import ab.nn.api as nn_dataset

from ab.gpt.util.Const import epoch_dir, new_nn_file, nngpt_dir, synth_dir, hp_file, NN_TRAIN_EPOCHS
from ab.gpt.util.Eval import Eval
from ab.gpt.util.Util import verify_nn_code, copy_to_lemur
from ab.gpt.util.CycleResults import generate_cycle_results, collect_cycle_metrics, save_cycle_results

# --- Default Evaluation Parameters ---
# These will be used as defaults for argparse arguments
TASK = 'img-classification'
DATASET = 'cifar-10'
METRIC = 'acc'

# Default hyperparameters. 'epoch' will be overridden.
LR = 0.01
BATCH = 64
DROPOUT = 0.2
MOMENTUM = 0.9
TRANSFORM = 'norm_256_flip'  # A common default, used by NNEval if prm is None

STOCHASTIC_DEPTH_PROB = 0.0
NORM_EPS = 1e-5
NORM_STD = 0.5
TIE_WEIGHTS = 0.0
DROPOUT_AUX = 0.0
ATTENTION_DROPOUT = 0.0
NORM_MOMENTUM = 0.1
SCORE_THRESH = 0.01
NMS_THRESH = 0.45
IOU_THRESH = 0.5
DETECTIONS_PER_IMG = 0.5
TOPK_CANDIDATES = 0.5
NEG_TO_POS_RATIO = 0.5
PRETRAINED = 0.0
PATCH_SIZE = 0.125

PRM_JSON = None  # Optional JSON string to override hyperparameters, e.g. '{"lr": 0.017, "batch": 32}'

SAVE_TO_DB = True
NN_NAME_PREFIX = None
NN_ALTER_EPOCHS = None
ONLY_EPOCH = None
EPOCH_LIMIT_MINUTES = None
CUSTOM_SYNTH_DIR = None
CYCLE = None  # Cycle number (separate from epoch - cycle is the finetuning iteration)


def main(nn_name_prefix=NN_NAME_PREFIX, nn_train_epochs=NN_TRAIN_EPOCHS, only_epoch=ONLY_EPOCH, save_to_db=SAVE_TO_DB,
         nn_alter_epochs=NN_ALTER_EPOCHS, task=TASK, dataset=DATASET, metric=METRIC, lr=LR, batch=BATCH, dropout=DROPOUT, momentum=MOMENTUM,
         transform=TRANSFORM, epoch_limit_minutes=EPOCH_LIMIT_MINUTES, custom_synth_dir=CUSTOM_SYNTH_DIR, cycle=CYCLE,
         stochastic_depth_prob=STOCHASTIC_DEPTH_PROB, norm_eps=NORM_EPS, norm_std=NORM_STD, tie_weights=TIE_WEIGHTS,
         dropout_aux=DROPOUT_AUX, attention_dropout=ATTENTION_DROPOUT, norm_momentum=NORM_MOMENTUM,
         score_thresh=SCORE_THRESH, nms_thresh=NMS_THRESH, iou_thresh=IOU_THRESH, detections_per_img=DETECTIONS_PER_IMG,
         topk_candidates=TOPK_CANDIDATES, neg_to_pos_ratio=NEG_TO_POS_RATIO, pretrained=PRETRAINED, patch_size=PATCH_SIZE,
         prm_json=PRM_JSON,
         feature_cache_dir=None, feature_cache_mode='read', num_workers=0, pin_memory=False,
         freeze_gpt2=False, force_eval=False):
    import sys
    print(f"DEBUG: sys.argv={sys.argv}")
    
    # SE STANDARD: Use relative path resolution via Const.py for generalization
    base_nngpt_path = nngpt_dir
    epoch_base = epoch_dir()
    
    if not epoch_base.is_dir():
        print(f"[ERROR] Epoch directory {epoch_base} doesn't exist.")
        return

    # Scan all epoch directories matching * pattern (e.g., A0, A1, etc.)
    if custom_synth_dir:
        epoch_dirs = [Path("custom_epoch")]
        print("Using custom synth directory, bypassing epoch scan.")
    else:
        epoch_dirs = sorted(list(epoch_base.glob("*")))
        print(f"Found {len(epoch_dirs)} epoch directories to scan.")
    
    for current_alter_epoch_path in epoch_dirs:
        # Filter by only_epoch if specified
        valid_names = {f"A{only_epoch}", f"Epoch_{only_epoch}"}
        if only_epoch is not None and not custom_synth_dir and current_alter_epoch_path.name not in valid_names:
            continue
            
        models_base_dir = current_alter_epoch_path / "synth_nn"
        if custom_synth_dir:
            models_base_dir = Path(custom_synth_dir)
            
        if not models_base_dir.is_dir():
            print(f"Directory {models_base_dir} not found. Skipping.")
            continue
        
        print(f"\n--- Scanning NNAlter Epoch: {current_alter_epoch_path.name} ---")
        cycle_start_time = time.time()
        current_cycle = cycle if cycle is not None else current_alter_epoch_path.name
        
        for model_id in os.listdir(models_base_dir):
            model_dir_path = models_base_dir / model_id
            if not model_dir_path.is_dir(): continue

            code_file_path = model_dir_path / new_nn_file
            if not code_file_path.exists(): continue

            if not force_eval and (model_dir_path / 'eval_info.json').exists():
                print(f"  [SKIP] Model {model_id} already evaluated (eval_info.json exists).")
                continue

            print(f"\n--- Evaluating Model: {model_id} ---")

            if not verify_nn_code(model_dir_path, code_file_path):
                print(f"Code verification failed for {model_id}. Skipping.")
                continue

            # Construct Hyperparameters (prm)
            prm = {
                'lr': lr, 'batch': batch, 'dropout': dropout, 'momentum': momentum,
                'transform': transform, 'epoch': nn_train_epochs,
                'limit': prm_json.get('limit', 7000) if prm_json else 7000,
                'num_workers': num_workers, 'pin_memory': pin_memory,
                'feature_cache_dir': feature_cache_dir, 'feature_cache_mode': feature_cache_mode,
                'freeze_gpt2': freeze_gpt2
            }
            
            prefix_for_db = nn_name_prefix
            orig_pref = None
            
            # 1. Try SQL Priority (Professor Requirement)
            try:
                original_files = list(model_dir_path.glob("original_*.py"))
                if original_files:
                    with open(original_files[0], 'r') as f:
                        b_code = f.read()
                    b_checksum = uuid4(b_code)
                    db_df = nn_dataset.data(nn=b_checksum)
                    if not db_df.empty:
                        row = db_df.iloc[0]
                        task, dataset, metric = row.get('task', task), row.get('dataset', dataset), row.get('metric', metric)
                        if isinstance(row.get('prm'), dict): prm.update(row['prm'])
                        orig_pref = row['nn'].split('-')[0] if 'nn' in row else None
                        print(f"  [SQL] Found baseline metadata for {b_checksum}")
            except Exception: pass

            # 2. Fallback to dataframe.df if SQL failed or was incomplete
            df_file_path = model_dir_path / 'dataframe.df'
            if df_file_path.exists():
                try:
                    origdf = pd.read_pickle(df_file_path)
                    task = origdf.get('task', task)
                    dataset = origdf.get('dataset', dataset)
                    metric = origdf.get('metric', metric)
                    if isinstance(origdf.get('prm'), dict): prm.update(origdf['prm'])
                    orig_pref = orig_pref or (origdf['nn'].split('-')[0] if 'nn' in origdf else None)
                except Exception: pass

            if prm_json: prm.update(prm_json)
            
            # Ensure CLI overrides last only if explicitly provided or missing
            if lr != LR or 'lr' not in prm: prm['lr'] = lr
            if batch != BATCH or 'batch' not in prm: prm['batch'] = batch
            prm['epoch'] = nn_train_epochs
            
            print(f"DEBUG: CLI transform={transform}, Default TRANSFORM={TRANSFORM}, prm['transform'] before override={prm.get('transform')}")
            if transform != TRANSFORM or 'transform' not in prm: prm['transform'] = transform
            print(f"DEBUG: prm['transform'] after override={prm.get('transform')}")
            prefix_for_db = nn_name_prefix or orig_pref

            try:
                evaluator = Eval(
                    model_source_package=str(model_dir_path),
                    task=task, dataset=dataset, metric=metric, prm=prm,
                    save_to_db=save_to_db, prefix=prefix_for_db, save_path=model_dir_path
                )
                if epoch_limit_minutes: evaluator.epoch_limit_minutes = epoch_limit_minutes
                
                # --- Lightweight NAS Bridge Contract Check ---
                if task == 'img-captioning' and prm.get('transform') == 'cached_blip2':
                    print(f"  [NAS] Checking CrossModalBridge shape contract for {model_id}...")
                    try:
                        import torch
                        import importlib.util
                        import sys

                        module_name = f"nas_check_{model_id}_{uuid4(str(code_file_path))[:8]}"
                        spec = importlib.util.spec_from_file_location(module_name, str(code_file_path))
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = mod
                        spec.loader.exec_module(mod)

                        # Check only the bridge to save memory/time
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        bridge = mod.CrossModalBridge(prm).to(device)
                        bridge.eval()

                        x = torch.randn(2, 32, 768, device=device)
                        with torch.no_grad():
                            y = bridge(x, None)

                        if not torch.is_tensor(y) or tuple(y.shape) != (2, 32, 768):
                            raise RuntimeError(f"Bridge shape invalid: got {tuple(y.shape) if torch.is_tensor(y) else type(y)}")

                        del bridge, x, y
                        release_memory()
                        print("  [NAS] Bridge shape check passed.")
                    except Exception as dry_run_err:
                        print(f"  [SKIP] Bridge contract failed: {dry_run_err}")
                        with open(model_dir_path / 'error.txt', 'w+') as f:
                            f.write(f"Bridge Contract Failed: {dry_run_err}")
                        continue

                eval_results = evaluator.evaluate(code_file_path)
                
                eval_info_data = {
                    "eval_args": evaluator.get_args(),
                    "eval_results": eval_results,
                    "cli_args": {'task': task, 'dataset': dataset, "metric": metric}
                }
                with open(model_dir_path / 'eval_info.json', 'w+') as f:
                    json.dump(eval_info_data, f, indent=4, default=str)
                
                # Standard Naming & LEMUR Storage (Exact Location)
                # Save directly to the LEMUR staging area without subfolders
                from ab.gpt.util.Const import new_lemur_nn_dir
                Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
                
                # Final naming: The folder name IS the proper name (Blip2Fast-A0-B0)
                final_name = model_id
                
                print(f"  [Standard] Saving model to LEMUR (Flat): {new_lemur_nn_dir}/{final_name}.py")
                shutil.copyfile(model_dir_path / new_nn_file, new_lemur_nn_dir / f"{final_name}.py")
                
                # Also record in stats (this handles the stat folder nesting as required by Train.py)
                copy_to_lemur(model_dir_path, final_name, task, dataset, metric)

            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"  [ERROR] {e}")
                with open(model_dir_path / 'error.txt', 'w+') as f: f.write(error_msg)
            finally:
                release_memory()
                time.sleep(1)

        # Cycle Results
        cycle_time = (time.time() - cycle_start_time) / 60.0
        metrics = collect_cycle_metrics(models_base_dir, current_alter_epoch_path)
        c_res = generate_cycle_results(current_cycle, models_base_dir, *metrics, cycle_time, current_alter_epoch_path)
        save_cycle_results(c_res, base_nngpt_path / "cycle_results.json")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neural Networks generated by NNAlter.py.")
    parser.add_argument('-ae', '--nn_alter_epochs', type=int, default=NN_ALTER_EPOCHS,
                        help="Number of epochs NNAlter.py was run for (e.g., if NNAlter's -e was 8, use 8 here).")
    parser.add_argument('-oe', '--only_epoch', type=int, default=ONLY_EPOCH,
                        help="Run NNAlter.py for the specified epoch only.")
    parser.add_argument('-te', '--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f"Number of epochs to train each altered NN during evaluation (default: {NN_TRAIN_EPOCHS}).")
    # Configurable evaluation parameters
    parser.add_argument('--task', type=str, default=TASK,
                        help=f"Default task for NNEval if not in dataframe.df (default: {TASK}).")
    parser.add_argument('--dataset', type=str, default=DATASET,
                        help=f"Default dataset for NNEval if not in dataframe.df (default: {DATASET}).")
    parser.add_argument('--metric', type=str, default=METRIC,
                        help=f"Default metric for NNEval if not in dataframe.df (default: {METRIC}).")

    # Configurable hyperparameters (part of prm dictionary for NNEval)
    parser.add_argument('--lr', type=float, default=LR,
                        help=f"Learning rate for NNEval if not in dataframe.df's prm (default: {LR}).")
    parser.add_argument('--batch_size', type=int, default=BATCH,
                        help=f"Batch size for NNEval if not in dataframe.df's prm (default: {BATCH}). Stored as 'batch' in prm.")
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help=f"Dropout rate for NNEval if not in dataframe.df's prm (default: {DROPOUT}).")
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help=f"Momentum for NNEval if not in dataframe.df's prm (default: {MOMENTUM}).")
    parser.add_argument('--transform', type=str, default=TRANSFORM,
                        help=f"Default transform for NNEval if not in dataframe.df's prm (default: {TRANSFORM}). Stored as 'transform' in prm.")
    parser.add_argument('--stochastic_depth_prob', type=float, default=STOCHASTIC_DEPTH_PROB,
                        help=f"Stochastic depth probability (default: {STOCHASTIC_DEPTH_PROB}).")
    parser.add_argument('--norm_eps', type=float, default=NORM_EPS,
                        help=f"Epsilon for normalization layers (default: {NORM_EPS}).")
    parser.add_argument('--norm_std', type=float, default=NORM_STD,
                        help=f"Std for normalization (default: {NORM_STD}).")
    parser.add_argument('--tie_weights', type=float, default=TIE_WEIGHTS,
                        help=f"Tie weights flag as float, >0.5 means True (default: {TIE_WEIGHTS}).")
    parser.add_argument('--dropout_aux', type=float, default=DROPOUT_AUX,
                        help=f"Auxiliary dropout rate (default: {DROPOUT_AUX}).")
    parser.add_argument('--attention_dropout', type=float, default=ATTENTION_DROPOUT,
                        help=f"Attention dropout rate (default: {ATTENTION_DROPOUT}).")
    parser.add_argument('--norm_momentum', type=float, default=NORM_MOMENTUM,
                        help=f"Momentum for normalization layers (default: {NORM_MOMENTUM}).")
    parser.add_argument('--score_thresh', type=float, default=SCORE_THRESH,
                        help=f"Score threshold for detection (default: {SCORE_THRESH}).")
    parser.add_argument('--nms_thresh', type=float, default=NMS_THRESH,
                        help=f"NMS threshold for detection (default: {NMS_THRESH}).")
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESH,
                        help=f"IoU threshold for detection (default: {IOU_THRESH}).")
    parser.add_argument('--detections_per_img', type=float, default=DETECTIONS_PER_IMG,
                        help=f"Detections per image as float in [0,1] (default: {DETECTIONS_PER_IMG}).")
    parser.add_argument('--topk_candidates', type=float, default=TOPK_CANDIDATES,
                        help=f"Top-k candidates as float in [0,1] (default: {TOPK_CANDIDATES}).")
    parser.add_argument('--neg_to_pos_ratio', type=float, default=NEG_TO_POS_RATIO,
                        help=f"Neg-to-pos ratio as float in [0,1] (default: {NEG_TO_POS_RATIO}).")
    parser.add_argument('--pretrained', type=float, default=PRETRAINED,
                        help=f"Use pretrained weights as float, >0.5 means True (default: {PRETRAINED}).")
    parser.add_argument('--patch_size', type=float, default=PATCH_SIZE,
                        help=f"Patch size as fraction of image size, used by VisionTransformer (default: {PATCH_SIZE}).")
    parser.add_argument('--prm_json', type=str, default=PRM_JSON,
                        help='JSON string of hyperparameter overrides applied last, e.g. \'{"lr": 0.017, "batch": 32}\'. Overrides all other sources except epoch.')

    # Other NNEval options
    parser.add_argument('--save_to_db', action=argparse.BooleanOptionalAction, default=SAVE_TO_DB,
                        help="Whether to save evaluation results to the database (enables with --save-to-db, disables with --no-save-to-db; default: enabled).")
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f"Default neural network name prefix (default: {NN_NAME_PREFIX}).")
    # Custom custom_synth_dir
    parser.add_argument('--custom_synth_dir', dest='custom_synth_dir', type=str, default=CUSTOM_SYNTH_DIR,
                        help="Custom directory containing generated models")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=False, help="Whether to pin memory in DataLoader.")
    parser.add_argument('--feature_cache_dir', type=str, default=None, help="Directory for feature caching.")
    parser.add_argument('--feature_cache_mode', type=str, default='read', choices=['read', 'write', 'auto'], help="Feature cache mode.")
    parser.add_argument('--epoch_limit_minutes', type=int, default=None, help="Time limit per epoch.")
    parser.add_argument('--freeze_gpt2', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to freeze the GPT-2 decoder backbone (default: disabled).")
    parser.add_argument('--cycle', type=int, default=CYCLE,
                        help="Cycle number (finetuning iteration, separate from epoch). If not specified, defaults to epoch number.")
    parser.add_argument('--force_eval', action='store_true', help="Force re-evaluation of models even if eval_info.json exists.")

    args = parser.parse_args()
    """
    Evaluates neural networks generated by NNAlter.py.

    :param args: Parsed command-line arguments.
    """
    print(f"Starting evaluation of altered NNs...")
    print(f"NNAlter run epochs to scan: {args.nn_alter_epochs}")
    print(f"Each altered NN will be trained for: {args.nn_train_epochs} epochs for evaluation.")
    print(f"Base task: {args.task}, Base dataset: {args.dataset}, Base metric: {args.metric}")
    print(f"Base Hyperparameters for NNEval (before df override):")
    print(f"  LR: {args.lr}, Batch Size: {args.batch_size}, Dropout: {args.dropout}, Momentum: {args.momentum}, Transform: {args.transform}")
    print(f"Save to DB: {args.save_to_db}")
    print(f"Prefix for the names of generated neural network: {args.nn_name_prefix}")

    prm_json_dict = None
    if args.prm_json:
        try:
            prm_json_dict = json.loads(args.prm_json)
            print(f"--prm_json overrides: {prm_json_dict}")
        except json.JSONDecodeError as e:
            print(f"Error parsing --prm_json: {e}", file=sys.stderr)
            sys.exit(1)

    main(
        nn_alter_epochs=args.nn_alter_epochs,
        nn_train_epochs=args.nn_train_epochs,
        task=args.task,
        dataset=args.dataset,
        metric=args.metric,
        save_to_db=args.save_to_db,
        nn_name_prefix=args.nn_name_prefix,
        custom_synth_dir=args.custom_synth_dir,
        epoch_limit_minutes=args.epoch_limit_minutes,
        cycle=args.cycle,
        lr=args.lr,
        batch=args.batch_size,
        dropout=args.dropout,
        momentum=args.momentum,
        transform=args.transform,
        prm_json=prm_json_dict,
        feature_cache_dir=args.feature_cache_dir,
        feature_cache_mode=args.feature_cache_mode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        freeze_gpt2=args.freeze_gpt2,
        force_eval=args.force_eval
    )

import argparse, json, sys, os, traceback
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import shutil
from ab.nn.util.Util import release_memory, uuid4
from ab.gpt.util.Util import read_py_file_as_string

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
         prm_json=PRM_JSON):
    base_nngpt_path = nngpt_dir  # out/nngpt
    if nn_alter_epochs is None:
        if epoch_dir().is_dir():
            nn_alter_epochs = len(os.listdir(epoch_dir()))
        else:
            print()
            print(f"Directory {epoch_dir()} doesn't exist", file=sys.stderr)

    if nn_alter_epochs:
        for i in ([only_epoch] if only_epoch is not None else range(nn_alter_epochs)):
            # Track cycle number and epoch number separately
            # Cycle is the finetuning iteration, epoch is within a cycle
            current_cycle = cycle if cycle is not None else i  # Default to epoch if cycle not specified
            current_epoch = i
            cycle_start_time = time.time()
            
            # Path to the output of one NNAlter.py epoch (e.g., out/nngpt/llm/epoch/A0)
            current_alter_epoch_path = epoch_dir(i)

            # If a custom custom_synth_dir is given, use it; otherwise build from epoch_dir
            if custom_synth_dir:
                models_base_dir = Path(custom_synth_dir)
            else:
                models_base_dir = synth_dir(current_alter_epoch_path)

            if not models_base_dir.exists():
                print(f"Directory {models_base_dir} for NNAlter epoch {i} not found. Skipping.")
                continue

            print(f"\n--- Scanning NNAlter Epoch Directory: {current_alter_epoch_path} ---")
            print(f"--- Synthesized Models Directory: {models_base_dir} ---")

            # Initialize cycle metrics collection (will be populated from eval_info.json files after evaluation)
            for model_id in os.listdir(models_base_dir):
                model_dir_path = models_base_dir / model_id
                if not model_dir_path.is_dir():
                    continue

                code_file_path = model_dir_path / new_nn_file
                df_file_path = model_dir_path / 'dataframe.df'  # Original model's metadata

                if not code_file_path.exists():
                    print(f"Code file {new_nn_file} not found in {model_dir_path}. Skipping.")
                    continue

                print(f"\n--- Evaluating Model: {model_dir_path.relative_to(base_nngpt_path)} ---")

                if not verify_nn_code(model_dir_path, code_file_path):
                    print(f"Code verification failed for {code_file_path}. Skipping evaluation.")
                    with open(model_dir_path / 'eval_verification_failed.txt', 'w') as f:
                        f.write("Initial code verification failed.")
                    continue

                # Initialize task, dataset, metric, and prm from command-line arguments (or their defaults)
                task = task
                dataset = dataset
                metric = metric
                # This prm structure is consistent with LEMUR dataset's expectations for model training
                prm = None
                hp_path = model_dir_path / hp_file
                if hp_path.exists():
                    try:
                        with open(hp_path) as f:
                            prm = json.load(f)
                        print(f'Training model {model_id} with LLM recommended prm {prm}')
                    except Exception as e:
                        print(f"Error loading LLM recommended training params from {hp_path}: {e}.")
                if not prm:
                    prm = {
                        'lr': lr,
                        'batch': batch,
                        'dropout': dropout,
                        'momentum': momentum,
                        'transform': transform, # Default transform from CLI
                        # 'epoch' will be set explicitly later
                        'stochastic_depth_prob': stochastic_depth_prob,
                        'norm_eps': norm_eps,
                        'norm_std': norm_std,
                        'tie_weights': tie_weights,
                        'dropout_aux': dropout_aux,
                        'attention_dropout': attention_dropout,
                        'norm_momentum': norm_momentum,
                        'score_thresh': score_thresh,
                        'nms_thresh': nms_thresh,
                        'iou_thresh': iou_thresh,
                        'detections_per_img': detections_per_img,
                        'topk_candidates': topk_candidates,
                        'neg_to_pos_ratio': neg_to_pos_ratio,
                        'pretrained': pretrained,
                        'patch_size': patch_size,
                    }
                    print(f'Training model {model_id} with command-line/default training params {prm}')
                prefix_for_db = nn_name_prefix  # Default prefix
                origdf = None
                orig_pref = None
                if df_file_path.exists():
                    try:
                        origdf = pd.read_pickle(df_file_path)
                        # Override with values from dataframe.df if they exist
                        task = origdf.get('task', task)
                        dataset = origdf.get('dataset', dataset)
                        metric = origdf.get('metric', metric)
                        orig_pref = origdf['nn'].split('-')[0]

                        original_prm_from_df = origdf.get('prm')
                        if isinstance(original_prm_from_df, dict):
                            # Update prm with values from df, df values take precedence.
                            # This will update lr, batch, dropout, momentum, transform if they exist in original_prm_from_df,
                            # and also add any other model-specific hyperparameters from the original model.
                            prm.update(original_prm_from_df)

                        prefix_for_db = nn_name_prefix or (origdf.get('nn', 'unknown').split('-')[0] if 'nn' in origdf else prefix_for_db)
                        print(f"  Loaded metadata from dataframe.df: task={task}, dataset={dataset}, metric={metric}")
                    except Exception as e:
                        print(f"  Error loading dataframe.df from {df_file_path}: {e}. Using command-line/default parameters for task, dataset, metric, and prm structure.")
                else:
                    print(f"  No dataframe.df found. Using command-line/default evaluation parameters.")

                if prm_json:
                    prm.update(prm_json)
                    print(f"  Applied --prm_json overrides: {prm_json}")

                # Crucial: set training epochs for this evaluation from nn_train_epochs
                # This overrides any 'epoch' value that might have come from original_prm_from_df
                prm['epoch'] = nn_train_epochs
                
                # Ensure transform is never None (must be a valid string for ab.nn.transform module)
                if prm.get('transform') is None or not isinstance(prm.get('transform'), str):
                    prm['transform'] = transform if transform else TRANSFORM
                
                print(f"  Final parameters for Eval: {prm}")
                print(f"  Task: {task}, Dataset: {dataset}, Metric: {metric}, Prefix: {prefix_for_db}")

                try:
                    evaluator = Eval(
                        model_source_package=str(model_dir_path),
                        task=task,
                        dataset=dataset,
                        metric=metric,
                        prm=prm,  # Pass the constructed prm
                        save_to_db=save_to_db,
                        prefix=prefix_for_db,
                        save_path=model_dir_path
                    )
                    if epoch_limit_minutes:
                        evaluator.epoch_limit_minutes = epoch_limit_minutes
                    eval_results = evaluator.evaluate(code_file_path)
                    print(f"  Evaluation results for {model_id}: {eval_results}")
                    
                    eval_info_data = {
                        "eval_args": evaluator.get_args(),  # This will show the prm used by Eval
                        "eval_results": eval_results,
                        "cli_args": {'task': task, 'dataset': dataset, "metric": metric, "lr": lr, "batch": batch,
                                     'dropout': dropout, 'momentum': momentum, 'transform': transform}
                    }
                    with open(model_dir_path / 'eval_info.json', 'w+') as f:
                        json.dump(eval_info_data, f, indent=4, default=str)

                    nn_name = uuid4(read_py_file_as_string(code_file_path))

                    pref = nn_name_prefix or orig_pref
                    if pref:
                        nn_name = pref + '-' + nn_name
                    copy_to_lemur(model_dir_path, nn_name, task, dataset, metric)

                except Exception as e:
                    error_msg = f"Error evaluating model {model_id}: {e}"
                    print(f"  {error_msg}")
                    detailed_error = traceback.format_exc()
                    print(detailed_error)
                    with open(model_dir_path / 'error.txt', 'w+') as f:
                        f.write(f"{error_msg}\n\n{detailed_error}")
                finally:
                    release_memory()
            
            # Generate cycle results JSON after all evaluations complete
            # Read all eval_info.json files that were created during evaluation
            cycle_end_time = time.time()
            cycle_time_minutes = (cycle_end_time - cycle_start_time) / 60.0
            
            # Collect metrics from all eval_info.json files in the models directory
            eval_results_list, model_dirs_list, successful_models, failed_models = collect_cycle_metrics(
                models_base_dir, current_alter_epoch_path
            )
            
            # Generate cycle results JSON from collected metrics
            # Note: cycle is the finetuning iteration, epoch is within a cycle
            cycle_results = generate_cycle_results(
                cycle=current_cycle,
                models_base_dir=models_base_dir,
                eval_results_list=eval_results_list,
                model_dirs_list=model_dirs_list,
                successful_models=successful_models,
                failed_models=failed_models,
                cycle_time_minutes=cycle_time_minutes,
                current_alter_epoch_path=current_alter_epoch_path
            )
            
            # Save cycle results JSON
            cycle_results_path = base_nngpt_path / "cycle_results.json"
            #create backup
            if cycle_results_path.exists():
                backup_path = base_nngpt_path / f"cycle_results_{i-1}.json"
                shutil.copy2(cycle_results_path, backup_path)
                print(f"Backup saved → {backup_path}")
            save_cycle_results(cycle_results, cycle_results_path)
            print(f"\n--- Cycle {current_cycle} (Epoch {current_epoch}) results saved to: {cycle_results_path} ---")
            print(f"  Found {len(eval_results_list)} successful evaluations from eval_info.json files")
            print(f"  Found {len(failed_models)} failed models")




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
    parser.add_argument('--epoch_limit_minutes', type=int, default=EPOCH_LIMIT_MINUTES,
                        help="Max minutes allowed per epoch (default: specified in NN Dataset).")
    parser.add_argument('--cycle', type=int, default=CYCLE,
                        help="Cycle number (finetuning iteration, separate from epoch). If not specified, defaults to epoch number.")

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

    main(args.nn_name_prefix, args.nn_train_epochs, args.only_epoch, args.save_to_db, args.nn_alter_epochs,
         args.task, args.dataset, args.metric, args.lr, args.batch_size, args.dropout, args.momentum,
         args.transform, args.epoch_limit_minutes, args.custom_synth_dir, args.cycle,
         args.stochastic_depth_prob, args.norm_eps, args.norm_std, args.tie_weights,
         args.dropout_aux, args.attention_dropout, args.norm_momentum,
         args.score_thresh, args.nms_thresh, args.iou_thresh, args.detections_per_img,
         args.topk_candidates, args.neg_to_pos_ratio, args.pretrained, args.patch_size,
         prm_json_dict)

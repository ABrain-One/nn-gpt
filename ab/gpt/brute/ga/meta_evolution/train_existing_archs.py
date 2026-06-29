"""
train_existing_archs.py
-----------------------
Standalone script that iterates over all FractalNet architectures already
generated in  ga_fractal_arch/, trains each for 1 epoch on CIFAR-10,
and saves the resulting stats into  secondary_stats/  using the same
naming convention as the existing  stats/  directory.

No GA, no LLM, no meta-evolution — pure batch (re-)training.
"""

import os
import sys
import json
import re
import time
import glob
import logging

# --- PATH SETUP (mirrors run_fractal_evolution.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- DIRECTORY CONSTANTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(BASE_DIR, 'ga_fractal_arch')
SECONDARY_STATS_DIR = os.path.join(BASE_DIR, 'secondary_stats')
LOG_FILE = os.path.join(BASE_DIR, 'train_existing_archs.log')

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(SECONDARY_STATS_DIR, exist_ok=True)

# --- LOGGING SETUP ---
logger = logging.getLogger("TrainExistingArchs")
logger.setLevel(logging.INFO)
logger.propagate = False

# Formatter: '2025-03-15 10:23:45 | INFO | message'
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Stream Handler (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File Handler (persistent)
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from ab.gpt.util.Eval import Eval
from ab.nn.util.Const import out_dir

# Regex to extract the 32-char MD5 checksum from the arch filename
# e.g.  img-classification_cifar-10_FractalNet-5d129b9b39e6f9f64b64eb02c89645c3.py
CHECKSUM_RE = re.compile(r'FractalNet-([0-9a-f]{32})\.py$')


def extract_checksum(filename: str) -> str | None:
    """Return the MD5 checksum from a ga_fractal_arch filename, or None."""
    m = CHECKSUM_RE.search(filename)
    return m.group(1) if m else None


def stats_exist(checksum: str) -> bool:
    """Check whether secondary_stats already contains a 1.json for this model."""
    stat_file = os.path.join(
        SECONDARY_STATS_DIR,
        f"img-classification_cifar_FractalNet-{checksum}",
        "1.json"
    )
    return os.path.isfile(stat_file)


def clear_stale_summary() -> None:
    """Delete out/training_summary.json so it cannot leak between models."""
    summary_path = out_dir / "training_summary.json"
    old_summary_path = os.path.join(os.getcwd(), 'out', 'training_summary.json')
    for p in [summary_path, old_summary_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
                logger.info(f"Cleared stale {p}")
            except Exception as e:
                logger.warning(f"Could not remove stale summary {p}: {e}")


def train_and_save(filepath: str, model_name: str, checksum: str) -> float:
    """
    Train *one* model for 1 epoch and persist the stats JSON.
    Returns the accuracy (0-100 scale) or 0.0 on failure.
    """
    eval_prm = {
        'lr': 0.01,
        'momentum': 0.9,
        'batch': 64,
        'epoch': 1,
        'transform': 'norm_32_flip',
    }

    clear_stale_summary()

    # ---- Evaluate ----
    logger.info("Starting evaluation ...")
    sys.stdout.flush()

    evaluator = Eval(
        model_source_package=ARCH_DIR,
        task='img-classification',
        dataset='cifar-10',
        metric='acc',
        prm=eval_prm,
        save_to_db=False,
        prefix=model_name,
        save_path=SECONDARY_STATS_DIR,
    )

    result = evaluator.evaluate(filepath)

    # ---- Collect stats (same logic as run_fractal_evolution.py) ----
    summary_path = out_dir / "training_summary.json"
    if not summary_path.exists():
        summary_path = os.path.join(os.getcwd(), 'out', 'training_summary.json')
    full_res = {}

    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                candidate = json.load(f)
            file_uid = candidate.get('uid', checksum)
            if file_uid == checksum:
                full_res = candidate
                logger.info("Loaded fresh training_summary.json (uid match)")
            else:
                logger.warning(f"training_summary.json uid mismatch "
                               f"({file_uid[:8]} vs {checksum[:8]}), ignoring stale file")
        except Exception as e:
            logger.error(f"Failed to read training summary: {e}")

    # Fall back to the direct result object
    if not full_res:
        if isinstance(result, dict):
            full_res = result
        else:
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
                    'model': model_name,
                },
                'hyperparameters': eval_prm,
                'training_summary': {
                    'total_epochs': eval_prm.get('epoch', 1),
                    'best_accuracy': acc_val,
                    'final_accuracy': acc_val,
                },
            }

    full_res['uid'] = checksum

    # ---- Save stats to secondary_stats/ ----
    stats_folder_name = f"img-classification_cifar_FractalNet-{checksum}"
    stats_folder_path = os.path.join(SECONDARY_STATS_DIR, stats_folder_name)
    os.makedirs(stats_folder_path, exist_ok=True)

    stat_file_saved = None
    epoch_details = full_res.get('epoch_details', [])
    if epoch_details:
        for ep_data in epoch_details:
            ep_num = ep_data.get('epoch', len(epoch_details))
            ep_res = dict(full_res)
            ep_res['current_epoch'] = ep_num
            ep_res['uid'] = checksum
            stat_file = os.path.join(stats_folder_path, f"{ep_num}.json")
            with open(stat_file, 'w') as sf:
                json.dump(ep_res, sf, indent=4)
            stat_file_saved = stat_file
        logger.info(f"Saved {len(epoch_details)} epoch JSON(s) to {stats_folder_path}")
    else:
        max_epochs = eval_prm.get('epoch', 1)
        if 'epoch_max' in full_res:
            max_epochs = full_res['epoch_max']
        elif 'training_summary' in full_res and 'total_epochs' in full_res['training_summary']:
            max_epochs = full_res['training_summary']['total_epochs']
        stat_file = os.path.join(stats_folder_path, f"{max_epochs}.json")
        with open(stat_file, 'w') as sf:
            json.dump(full_res, sf, indent=4)
        stat_file_saved = stat_file
        logger.info(f"Saved stats sequentially to {stats_folder_path}")

    # ---- Extract accuracy for logging ----
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
        except Exception:
            pass

    return final_accuracy, stat_file_saved


def main():
    logger.info("="*60)
    logger.info("train_existing_archs.py started.")
    sys.stdout.flush()

    # Discover all model files
    pattern = os.path.join(ARCH_DIR, '*.py')
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        logger.warning(f"No .py files found in {ARCH_DIR}")
        return

    total = len(model_files)
    logger.info(f"Found {total} architecture(s) in ga_fractal_arch/")
    logger.info(f"Stats will be saved to secondary_stats/")
    logger.info(f"Log output will be persisted in train_existing_archs.log")
    logger.info("="*60)
    sys.stdout.flush()

    trained = 0
    skipped = 0
    failed = 0
    total_t0 = time.time()

    for idx, filepath in enumerate(model_files, start=1):
        filename = os.path.basename(filepath)
        checksum = extract_checksum(filename)

        if checksum is None:
            logger.info(f"[{idx}/{total}] SKIP (no checksum in filename): {filename}")
            skipped += 1
            sys.stdout.flush()
            continue

        # Model name used for Eval prefix (uses cifar-10, matching the arch files)
        model_name = f"img-classification_cifar-10_FractalNet-{checksum}"

        # Skip if already evaluated
        if stats_exist(checksum):
            logger.info(f"[{idx}/{total}] SKIP (stats already exist): FractalNet-{checksum}")
            skipped += 1
            sys.stdout.flush()
            continue

        logger.info(f"[{idx}/{total}] ▶ Starting: FractalNet-{checksum}")
        sys.stdout.flush()
        
        t0 = time.time()
        try:
            accuracy, dest_file = train_and_save(filepath, model_name, checksum)
            elapsed = time.time() - t0
            
            try:
                rel_dest = os.path.relpath(dest_file, BASE_DIR)
            except Exception:
                rel_dest = dest_file
                
            logger.info("─" * 42)
            logger.info(f"[{idx}/{total}] FractalNet-{checksum}")
            logger.info(f"        ✓ Accuracy      : {accuracy:.2f}%")
            logger.info(f"        ✓ Elapsed       : {elapsed:.1f}s")
            logger.info(f"        ✓ Saved to      : {rel_dest}")
            logger.info("─" * 42)
            
            trained += 1
            sys.stdout.flush()
        except Exception as e:
            import traceback
            logger.error(f"[{idx}/{total}] FAILED | Exception: {e}")
            logger.error(traceback.format_exc())
            failed += 1
            sys.stdout.flush()

    total_elapsed = time.time() - total_t0
    logger.info("="*60)
    logger.info(f"DONE — Trained: {trained} | Skipped: {skipped} | Failed: {failed} | Total Time: {total_elapsed:.1f}s")
    logger.info("="*60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

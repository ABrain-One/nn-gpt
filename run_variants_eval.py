"""
run_variants_eval.py

Evaluates all variants produced by generate_variants.py on multiple datasets.
Calls Eval directly — no NNEval.main(), no temp dirs.

Prefix is derived per-model from variant_meta.json:
    {base_nn}-loss_{loss}-opt_{optimizer}

Usage:
    python run_variants_eval.py
    python run_variants_eval.py --datasets cifar-10 mnist --nn_train_epochs 3
"""
import argparse, json, os, random, sqlite3, sys, time, traceback
from pathlib import Path

from ab.gpt.util.Const import new_nn_file, NN_TRAIN_EPOCHS
from ab.gpt.util.Eval import Eval
from ab.gpt.util.Util import verify_nn_code, copy_to_lemur, read_py_file_as_string
from ab.nn.util.Util import release_memory, uuid4
from ab.gpt.NNEval import (
    LR, BATCH, DROPOUT, MOMENTUM, TRANSFORM,
    STOCHASTIC_DEPTH_PROB, NORM_EPS, NORM_STD, TIE_WEIGHTS, DROPOUT_AUX,
    ATTENTION_DROPOUT, NORM_MOMENTUM, SCORE_THRESH, NMS_THRESH, IOU_THRESH,
    DETECTIONS_PER_IMG, TOPK_CANDIDATES, NEG_TO_POS_RATIO, PRETRAINED, PATCH_SIZE,
)

# ---------- Defaults ----------
DEFAULT_SYNTH_DIR = 'out/nngpt/llm/epoch/A0/synth_nn'
DEFAULT_DATASETS = [
    'celeba-gender', 'cifar-10', 'cifar-100',
    'imagenette', 'mnist', 'places365', 'svhn',
]
TASK   = 'img-classification'
METRIC = 'acc'

DEFAULT_PRM = {
    'lr':                    LR,
    'batch':                 BATCH,
    'dropout':               DROPOUT,
    'momentum':              MOMENTUM,
    'transform':             TRANSFORM,
    'stochastic_depth_prob': STOCHASTIC_DEPTH_PROB,
    'norm_eps':              NORM_EPS,
    'norm_std':              NORM_STD,
    'tie_weights':           TIE_WEIGHTS,
    'dropout_aux':           DROPOUT_AUX,
    'attention_dropout':     ATTENTION_DROPOUT,
    'norm_momentum':         NORM_MOMENTUM,
    'score_thresh':          SCORE_THRESH,
    'nms_thresh':            NMS_THRESH,
    'iou_thresh':            IOU_THRESH,
    'detections_per_img':    DETECTIONS_PER_IMG,
    'topk_candidates':       TOPK_CANDIDATES,
    'neg_to_pos_ratio':      NEG_TO_POS_RATIO,
    'pretrained':            PRETRAINED,
    'patch_size':            PATCH_SIZE,
}


def copy_to_lemur_with_retry(model_dir, nn_name, task, dataset, metric,
                              max_retries: int = 8, base_delay: float = 1.0):
    """Call copy_to_lemur, retrying on SQLite lock with exponential backoff + jitter.

    With 7 parallel pods all finishing a model at the same time, lock contention
    is predictable. Jitter spreads retries so pods don't keep colliding on the
    same backoff interval.

    Waits: ~1s, ~2s, ~4s, ~8s, ~16s, ~32s, ~64s, ~128s  (+ up to 1s jitter each)
    Total worst-case wait before giving up: ~4 minutes.
    """
    for attempt in range(max_retries):
        try:
            copy_to_lemur(model_dir, nn_name, task, dataset, metric)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"     DB locked — retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)


def resolve_prefix(model_dir: Path, override: str | None) -> str | None:
    if override:
        return override
    meta_path = model_dir / 'variant_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                m = json.load(f)
            return f"{m['base_nn']}-loss_{m['loss']}-opt_{m['optimizer']}"
        except Exception as e:
            print(f"  Warning: could not read variant_meta.json: {e}")
    return None


def run(synth_dir: Path, datasets: list, nn_train_epochs: int,
        save_to_db: bool, nn_name_prefix: str | None):

    model_ids = sorted(d for d in os.listdir(synth_dir) if (synth_dir / d).is_dir())
    print(f"Found {len(model_ids)} variant(s) in {synth_dir.resolve()}")

    for model_id in model_ids:
        model_dir = synth_dir / model_id
        code_file = model_dir / new_nn_file

        if not code_file.exists():
            print(f"\n[{model_id}] No {new_nn_file} — skipping.")
            continue

        prefix = resolve_prefix(model_dir, nn_name_prefix)
        print(f"\n{'='*60}")
        print(f"Variant : {model_id}  |  prefix: {prefix}")
        print(f"{'='*60}")

        if not verify_nn_code(model_dir, code_file):
            print(f"  Verification failed — skipping.")
            continue

        prm = {**DEFAULT_PRM, 'epoch': nn_train_epochs}

        for dataset in datasets:
            ds_safe = dataset.replace('/', '_').replace('-', '_')
            print(f"\n  [{model_id}] → {dataset}")
            try:
                evaluator = Eval(
                    model_source_package=str(model_dir),
                    task=TASK,
                    dataset=dataset,
                    metric=METRIC,
                    prm=prm,
                    save_to_db=save_to_db,
                    prefix=prefix,
                    save_path=model_dir,
                )
                eval_results = evaluator.evaluate(code_file)
                print(f"     Result: {eval_results}")

                eval_info = {
                    'eval_args':    evaluator.get_args(),
                    'eval_results': eval_results,
                    'dataset':      dataset,
                }
                with open(model_dir / f'eval_info_{ds_safe}.json', 'w') as f:
                    json.dump(eval_info, f, indent=4, default=str)
                with open(model_dir / 'eval_info.json', 'w') as f:
                    json.dump(eval_info, f, indent=4, default=str)

                nn_name = uuid4(read_py_file_as_string(code_file))
                if prefix:
                    nn_name = prefix + '-' + nn_name
                copy_to_lemur_with_retry(model_dir, nn_name, TASK, dataset, METRIC)

            except Exception as e:
                err = traceback.format_exc()
                print(f"     ERROR: {e}")
                with open(model_dir / f'error_{ds_safe}.txt', 'w') as f:
                    f.write(f"{e}\n\n{err}")
            finally:
                release_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generate_variants.py output across multiple datasets."
    )
    parser.add_argument('--synth_dir', default=DEFAULT_SYNTH_DIR,
                        help=f"Directory containing B* variant subdirs (default: {DEFAULT_SYNTH_DIR})")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help="Datasets to train on (default: all 7)")
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f"Training epochs per model/dataset (default: {NN_TRAIN_EPOCHS})")
    parser.add_argument('--save_to_db', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--nn_name_prefix', type=str, default=None,
                        help="Override prefix for ALL models (default: read from variant_meta.json)")

    args = parser.parse_args()
    synth_dir = Path(args.synth_dir)
    if not synth_dir.exists():
        print(f"Error: synth_dir not found: {synth_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"synth_dir  : {synth_dir.resolve()}")
    print(f"datasets   : {args.datasets}")
    print(f"epochs     : {args.nn_train_epochs}")
    print(f"save_to_db : {args.save_to_db}")
    print(f"prefix     : {args.nn_name_prefix or '(from variant_meta.json)'}")

    run(synth_dir, args.datasets, args.nn_train_epochs, args.save_to_db, args.nn_name_prefix)


if __name__ == '__main__':
    main()

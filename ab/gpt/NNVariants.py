"""Generate loss/optimizer variants of LEMUR neural networks.

Deterministic, rule-based counterpart to ``NNAlter``: pick one or more models
from the LEMUR dataset by name, substitute the loss function and optimizer, and
write each variant into the standard ``synth_nn/B*`` layout that ``NNEval``
consumes. ``NNEval`` reads the per-variant ``variant_meta.json`` and encodes the
loss/optimizer into the LEMUR row name, so no separate evaluator is needed.

Usage:
    # sweep the full loss × optimizer grid for one model
    python -m ab.gpt.NNVariants --nn ResNet

    # choose specific models and a sub-grid
    python -m ab.gpt.NNVariants --nn ResNet AlexNet --losses CrossEntropyLoss --optimizers Adam SGD

    # custom folder of nets not in the DB (filename = model name)
    python -m ab.gpt.NNVariants --src_dir ./my_models
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import ab.nn.api as nn_dataset
from ab.nn.util.Const import nn_dir

from ab.gpt.util.Const import epoch_dir, synth_dir, new_nn_file
from ab.gpt.util.VariantGen import LOSS_SPECS, OPTIM_SPECS, iter_variants

TASK = "img-classification"


def _read_nets_from_package() -> list[tuple[str, str]]:
    """Fallback: read NN source directly from the installed nn-dataset package
    files when the DB is empty (build failed / never committed)."""
    if not nn_dir.exists():
        return []
    models = []
    for p in sorted(nn_dir.iterdir()):
        if p.suffix == ".py" and p.name != "__init__.py":
            models.append((p.stem, p.read_text(encoding="utf-8")))
    return models


def fetch_nets(task: str) -> list[tuple[str, str]]:
    """Retrieve ``(name, code)`` pairs using the same sequence as NNAlter.

    Mirrors ab/gpt/util/AlterNN.py: pull the best-accuracy rows for the task, then
    take one row per distinct net via ``groupby('nn').sample(n=1)``, reading the
    source from the ``nn_code`` column.

    Falls back to reading the installed package files directly if the DB is empty.
    """
    df = nn_dataset.data(only_best_accuracy=True, task=task)
    if df is not None and len(df) > 0:
        df = df.groupby(by="nn").sample(n=1)
        return [(row["nn"], row["nn_code"]) for _, row in df.iterrows()]
    # DB empty — fall back to reading .py files from the installed nn-dataset package
    print("  DB returned no nets — falling back to installed package files.")
    return _read_nets_from_package()


def collect_sources(nn_names: list[str] | None, src_dir: str | None,
                    task: str) -> list[tuple[str, str]]:
    """Gather ``(model_name, source)`` pairs from LEMUR and/or a local folder.

    DB retrieval uses NNAlter's sequence (``fetch_nets``). ``--nn`` filters that
    pool to specific names; ``--src_dir`` reads every ``*.py`` file (filename stem
    = model name). The inputs may be combined in one run.
    """
    models: list[tuple[str, str]] = []

    if nn_names:
        by_name = {name: code for name, code in fetch_nets(task)}
        for name in nn_names:
            if name in by_name:
                models.append((name, by_name[name]))
            else:
                print(f"  '{name}' not found in LEMUR (task={task}) — skipping.")

    if src_dir:
        folder = Path(src_dir)
        py_files = sorted(p for p in folder.rglob("*.py") if p.name != "__init__.py")
        if not py_files:
            print(f"  No .py model files found in {folder.resolve()}")
        for p in py_files:
            models.append((p.stem, p.read_text(encoding="utf-8")))

    return models


def generate(models: list[tuple[str, str]], losses: list[str], optimizers: list[str],
             out_epoch: int, clean: bool) -> int:
    out_base = synth_dir(epoch_dir(out_epoch))
    if clean:
        shutil.rmtree(out_base, ignore_errors=True)
    out_base.mkdir(parents=True, exist_ok=True)

    # Add new variants on top of whatever is already there: resume numbering after
    # the highest existing B* (or start at B0 if the dir is empty). With --clean the
    # dir was just wiped, so this naturally restarts at B0.
    existing = [int(d.name[1:]) for d in out_base.glob("B*") if d.name[1:].isdigit()]
    variant_counter = max(existing) + 1 if existing else 0
    skipped = 0

    for nn_name, src in models:
        for loss_name, optim_name, new_src, err in iter_variants(src, losses, optimizers):
            if err:
                print(f"  Skipping {nn_name} {loss_name}/{optim_name}: {err}")
                skipped += 1
                continue

            variant_dir = out_base / f"B{variant_counter}"
            variant_dir.mkdir(parents=True, exist_ok=True)

            (variant_dir / f"original_{nn_name}.py").write_text(src, encoding="utf-8")
            (variant_dir / new_nn_file).write_text(new_src, encoding="utf-8")

            meta = {"base_nn": nn_name, "loss": loss_name, "optimizer": optim_name}
            (variant_dir / "variant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            variant_counter += 1

    print(f"\n{'=' * 60}")
    print(f"Generated {variant_counter} variant(s) in: {out_base.resolve()}")
    if skipped:
        print(f"Skipped {skipped} variant(s) due to errors")
    print(f"\nTo evaluate, run (one dataset per run):")
    print(f"  python -m ab.gpt.NNEval --dataset cifar-10 --nn_train_epochs 5")
    print(f"{'=' * 60}\n")
    return variant_counter


def main():
    parser = argparse.ArgumentParser(
        description="Generate loss/optimizer variants of LEMUR models into the synth_nn layout."
    )
    parser.add_argument("--nn", nargs="+", default=None,
                        help="One or more model names from LEMUR (e.g. --nn ResNet AlexNet).")
    parser.add_argument("--src_dir", default=None,
                        help="Folder of custom *.py models not in the DB "
                             "(filename stem is used as the model name). "
                             "May be combined with --nn.")
    parser.add_argument("--losses", nargs="+", default=None, choices=list(LOSS_SPECS),
                        help=f"Loss functions to substitute (default: all {list(LOSS_SPECS)}).")
    parser.add_argument("--optimizers", nargs="+", default=None, choices=list(OPTIM_SPECS),
                        help=f"Optimizers to substitute (default: all {list(OPTIM_SPECS)}).")
    parser.add_argument("--task", default=TASK,
                        help=f"LEMUR task to pull models from (default: {TASK}).")
    parser.add_argument("--out_epoch", type=int, default=0,
                        help="Epoch index for the output dir A{n} (default: 0).")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False,
                        help="Wipe the synth_nn dir before writing and restart at B0 "
                             "(default: False — new variants are appended after the last B*).")

    args = parser.parse_args()

    if not args.nn and not args.src_dir:
        parser.error("provide --nn (model names from the DB) and/or --src_dir "
                     "(a folder of .py models).")

    print(f"models     : {args.nn or '(none from DB)'}")
    print(f"src_dir    : {args.src_dir or '(none)'}")
    print(f"losses     : {args.losses or list(LOSS_SPECS)}")
    print(f"optimizers : {args.optimizers or list(OPTIM_SPECS)}")
    print(f"task       : {args.task}")
    print(f"out dir    : {synth_dir(epoch_dir(args.out_epoch))}")

    models = collect_sources(args.nn, args.src_dir, args.task)
    if not models:
        print("No input models resolved — nothing to do.", file=sys.stderr)
        sys.exit(1)

    count = generate(models, args.losses, args.optimizers, args.out_epoch, args.clean)
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

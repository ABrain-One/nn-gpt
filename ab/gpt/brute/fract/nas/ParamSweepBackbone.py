"""
ParamSweepBackbone.py
=====================
Grid-search over training hyper-parameters AND backbone counts for
NNAlterBackboneNet, then print a ranked comparison table.

Example usage (mirrors NNAlterBackboneNet CLI):

    python -m ab.gpt.brute.fract.backbone.ParamSweepBackbone \
        -e 1 -m 3 -nb 1 2 3 4 -te 1 \
        --datasets imagenette \
        --lr 0.001 0.01 0.1 \
        --pretrained 0.0 1.0 \
        --batch 32 64 \
        --max_steps 100

Only the axes you supply are swept; everything else stays fixed.
The script reuses all helpers from NNAlterBackboneNet so the evaluation
logic is identical.
"""

import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

from ab.gpt.brute.fract.backbone.NNAlterBN import alter
from ab.gpt.NNEval import main as eval_main
from ab.gpt.util.Const import nngpt_dir, synth_dir, epoch_dir

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (duplicated from NNAlterBackboneNet so this script is self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def _best_accuracy(run_summary):
    """Return (best_test_acc, best_train_acc) from the run summary."""
    best_test = 0.0
    best_train = 0.0
    for epoch in run_summary.get("epochs", []):
        for r in epoch.get("model_results", []):
            if r.get("success"):
                acc = r.get("accuracy") or 0.0
                if acc > best_test:
                    best_test = acc
                train_acc = r.get("train_accuracy") or 0.0
                if train_acc > best_train:
                    best_train = train_acc
    return best_test, best_train


def _collect_models(run_summary):
    return [
        {
            "model_id": r.get("model_id"),
            "accuracy": r.get("accuracy"),
            "train_accuracy": r.get("train_accuracy"),
            "success": r.get("success"),
        }
        for epoch in run_summary.get("epochs", [])
        for r in epoch.get("model_results", [])
    ]


def _read_eval_info_accs(epochs, dataset_name):
    """Scan archived eval_info_<dataset>.json files and return the best
    (test_accuracy, train_accuracy) pair found across all models."""
    safe = dataset_name.replace("/", "_").replace(" ", "_")
    best_test = 0.0
    best_train = 0.0
    for i in range(epochs):
        s = Path(synth_dir(epoch_dir(i)))
        if not s.exists():
            continue
        for model_dir in s.iterdir():
            if not model_dir.is_dir():
                continue
            info_path = model_dir / f"eval_info_{safe}.json"
            if not info_path.exists():
                # fall back to unarchived name (before renaming)
                info_path = model_dir / "eval_info.json"
            if not info_path.exists():
                continue
            try:
                data = json.loads(info_path.read_text())
                prm = data.get("eval_args", {}).get("prm", {})
                test_acc = float(prm.get("accuracy") or prm.get("metric_acc") or 0.0)
                train_acc = float(prm.get("train_accuracy") or 0.0)
                if test_acc > best_test:
                    best_test = test_acc
                if train_acc > best_train:
                    best_train = train_acc
            except Exception:
                pass
    return best_test, best_train


def _archive_eval_artifacts(epochs, dataset_name):
    safe = dataset_name.replace("/", "_").replace(" ", "_")
    for i in range(epochs):
        s = Path(synth_dir(epoch_dir(i)))
        if not s.exists():
            continue
        for model_dir in s.iterdir():
            if model_dir.is_dir():
                for artifact in ("eval_info.json", "eval_summary.json"):
                    p = model_dir / artifact
                    if p.exists():
                        stem, ext = artifact.rsplit(".", 1)
                        p.rename(model_dir / f"{stem}_{safe}.{ext}")


# ──────────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_prm(prm: dict) -> str:
    """Short one-liner for a param dict."""
    return " | ".join(f"{k}={v}" for k, v in prm.items())


def _print_table(rows: list[dict], datasets: list[str]):
    """
    rows: list of {
        'nb': int,
        'prm': dict,
        'ds_best': {ds: float},          # test accuracy per dataset
        'ds_train': {ds: float},         # train accuracy per dataset
        'avg': float,
        'avg_train': float,
    }
    Sorted descending by avg test accuracy.
    """
    rows = sorted(rows, key=lambda r: r["avg"], reverse=True)

    # Column widths
    prm_w  = max(len(_fmt_prm(r["prm"])) for r in rows) + 2
    prm_w  = max(prm_w, len("Params") + 2)
    nb_w   = 9   # "Backbones"
    ds_w   = 14  # wide enough for "test / train"
    avg_w  = 16  # wide enough for "test / train avg"

    sep = (
        "+" + "-" * (nb_w + 2)
        + "+" + "-" * (prm_w + 2)
        + ("+" + "-" * (ds_w + 2)) * len(datasets)
        + "+" + "-" * (avg_w + 2) + "+"
    )

    def row_str(nb, prm_s, ds_vals, avg, header=False):
        fmt = f"| {{:<{nb_w}}} | {{:<{prm_w}}} |"
        for v in ds_vals:
            if header:
                fmt += f" {{:^{ds_w}}} |"
            else:
                fmt += f" {{:>{ds_w}}} |"
        fmt += f" {{:>{avg_w}}} |"
        return fmt.format(nb, prm_s, *ds_vals, avg)

    # Header label: each dataset column shows "test / train"
    ds_headers = [f"{ds[:8]} t/tr" for ds in datasets]

    print()
    print("=" * (nb_w + prm_w + (ds_w + 3) * len(datasets) + avg_w + 7))
    print("  PARAMETER SWEEP RESULTS  (sorted by avg test accuracy, best first)")
    print("  Column format per dataset: test_acc / train_acc")
    print("=" * (nb_w + prm_w + (ds_w + 3) * len(datasets) + avg_w + 7))
    print(sep)
    print(row_str("Backbones", "Params", ds_headers, "test / train avg", header=True))
    print(sep)
    for i, r in enumerate(rows):
        marker = " ★" if i == 0 else "  "
        nb_s = f"{r['nb']}_backbones"
        prm_s = _fmt_prm(r["prm"])
        ds_vals = [
            f"{r['ds_best'].get(ds, 0.0):.4f}/{r['ds_train'].get(ds, 0.0):.4f}"
            for ds in datasets
        ]
        avg_s = f"{r['avg']:.4f}/{r['avg_train']:.4f}"
        print(marker + row_str(nb_s, prm_s, ds_vals, avg_s)[2:])
    print(sep)
    print()

    # Mini-summary: best per sweep axis
    all_prm_keys = list(rows[0]["prm"].keys()) if rows else []
    for key in all_prm_keys:
        seen = {}
        for r in rows:
            v = r["prm"].get(key)
            if v not in seen or r["avg"] > seen[v]:
                seen[v] = r["avg"]
        best_val = max(seen, key=seen.get)
        print(f"  Best {key:>14}: {best_val}  (test_avg={seen[best_val]:.4f})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid-search over hyper-params + backbone counts for NNAlterBackboneNet."
    )
    # ── Architecture sweep axes ──────────────────────────────────────────────
    parser.add_argument("-nb", "--num_backbones", type=int, nargs="+", default=[1],
                        help="Backbone count(s) to sweep (e.g. -nb 1 2 3 4).")
    parser.add_argument("-e",  "--epochs",        type=int, default=1,
                        help="Generation epochs per run.")
    parser.add_argument("-m",  "--max_variants",  type=int, default=3,
                        help="Models generated per epoch.")
    parser.add_argument("-te", "--train_epochs",  type=int, default=1,
                        help="Training epochs per model.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["imagenette"],
                        help="Datasets to evaluate on.")

    # ── Hyper-param sweep axes (each accepts multiple values) ────────────────
    parser.add_argument("--lr",         type=float, nargs="+", default=[0.0005, 0.01],
                        help="Learning rate value(s) to sweep.")
    parser.add_argument("--pretrained", type=float, nargs="+", default=[0.0,1.0],
                        help="Pretrained weight(s) to sweep (0.0 = random, 1.0 = pretrained).")
    parser.add_argument("--batch",      type=int,   nargs="+", default=[32, 64],
                        help="Batch size(s) to sweep.")
    parser.add_argument("--max_steps",  type=int,   nargs="+", default=[100],
                        help="Max steps value(s) to sweep.")
    parser.add_argument("--dropout",     type=float, nargs="+", default=[0.1, 0.2],
                        help="Dropout value(s) to sweep.")
    parser.add_argument("--transform",   type=str,   nargs="+", default=["norm_256_flip", "complex_256_flip", "echo_256_flip"],
                        help="Transform value(s) to sweep.")
    parser.add_argument("--tie_weights", type=float, nargs="+", default=[0.0, 1.0],
                        help="Tie weights value(s) to sweep.")

    # ── Fixed overrides (passed straight through to eval) ───────────────────
    parser.add_argument("--extra_prm_json", type=str, default=None,
                        help="Extra fixed eval params as JSON (merged under sweep values).")

    args = parser.parse_args()

    extra_fixed = json.loads(args.extra_prm_json) if args.extra_prm_json else {}

    # Build the Cartesian product of all sweep axes
    param_grid = list(itertools.product(
        args.lr,
        args.pretrained,
        args.batch,
        args.max_steps,
        args.dropout,
        args.transform,
        args.tie_weights,
    ))

    total_runs = len(args.num_backbones) * len(param_grid)
    print(f"\n{'='*70}")
    print(f"  Parameter Sweep: {len(args.num_backbones)} backbone config(s) × "
          f"{len(param_grid)} param combo(s) = {total_runs} run(s)")
    print(f"  Backbone counts : {args.num_backbones}")
    print(f"  LR values       : {args.lr}")
    print(f"  Pretrained      : {args.pretrained}")
    print(f"  Batch sizes     : {args.batch}")
    print(f"  Max steps       : {args.max_steps}")
    print(f"  Dropout         : {args.dropout}")
    print(f"  Transform       : {args.transform}")
    print(f"  Tie Weights     : {args.tie_weights}")
    print(f"  Datasets        : {args.datasets}")
    print(f"{'='*70}\n")

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    run_tag = f"{date_str}_{time_str}"
    os.environ["NNGPT_EPOCH_DIR_OVERRIDE"] = str(Path(nngpt_dir) / "llm" / date_str / time_str)

    all_rows = []
    nb_run_idx = 0  # counts backbone groups (for clean flag)

    for nb in args.num_backbones:
        nb_run_idx += 1
        model_prefix = f"nb{nb}_"

        # ── Step 1: Generate architectures ONCE for this backbone count ───────
        print(f"\n{'='*70}")
        print(f"  [GENERATE] nb={nb} backbone(s)  ({nb_run_idx}/{len(args.num_backbones)})")
        print(f"{'='*70}")
        alter(
            args.epochs,
            "NN_alter.json",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            max_variants=args.max_variants,
            num_backbones=nb,
            clean=(nb_run_idx == 1),   # clean only before the very first backbone
            model_prefix=model_prefix,
        )

        # ── Sweep all param combos on the generated models ────────────
        total_param_runs = len(param_grid)
        for prm_idx, (lr, pretrained, batch, max_steps, dropout, transform, tie_weights) in enumerate(param_grid, 1):
            prm = {
                "lr": lr,
                "pretrained": pretrained,
                "batch": batch,
                "max_steps": max_steps,
                "dropout": dropout,
                "transform": transform,
                "tie_weights": tie_weights,
            }
            prm_merged = {**extra_fixed, **prm}  # sweep overrides extra_fixed

            print(f"\n{'─'*70}")
            print(f"  [EVAL] nb={nb} | combo {prm_idx}/{total_param_runs} | {_fmt_prm(prm)}")
            print(f"{'─'*70}")

            ds_best  = {}   # test accuracy per dataset
            ds_train = {}   # train accuracy per dataset
            for ds in args.datasets:
                print(f"\n  -- nb={nb} | {_fmt_prm(prm)} | dataset={ds} --")
                run_summary = eval_main(
                    nn_train_epochs=args.train_epochs,
                    dataset=ds,
                    save_to_db=False,
                    prm_json=prm_merged,
                )
                _archive_eval_artifacts(args.epochs, ds)

                # Custom saving to database
                from ab.nn.util.Util import uuid4
                from ab.nn.util.Const import db_file
                from ab.gpt.util.Const import new_lemur_nn_dir
                import sqlite3

                for epoch_info in run_summary.get("epochs", []):
                    for r in epoch_info.get("model_results", []):
                        if r.get("success") and r.get("code_file"):
                            code_file_path = Path(r["code_file"])
                            if code_file_path.exists():
                                code = code_file_path.read_text(encoding="utf-8")
                                checksum = uuid4(code)
                                
                                # Find the actual name that was copied to lemur
                                nn_name = checksum
                                if new_lemur_nn_dir.exists():
                                    matches = list(new_lemur_nn_dir.glob(f"*{checksum}.py"))
                                    if matches:
                                        nn_name = matches[0].stem
                                        
                                desc_file = code_file_path.parent / "model_description.json"
                                desc_json = desc_file.read_text(encoding="utf-8") if desc_file.exists() else None
                                
                                print(f"  [DB] Custom saving model {nn_name} to database...")
                                conn = sqlite3.connect(db_file)
                                try:
                                    cursor = conn.cursor()
                                    cursor.execute("SELECT name FROM nn WHERE name = ?", (nn_name,))
                                    if cursor.fetchone():
                                        cursor.execute(
                                            "UPDATE nn SET code = ?, id = ?, model_description_json = ? WHERE name = ?",
                                            (code, checksum, desc_json, nn_name)
                                        )
                                    else:
                                        cursor.execute(
                                            "INSERT INTO nn (name, code, id, model_description_json) VALUES (?, ?, ?, ?)",
                                            (nn_name, code, checksum, desc_json)
                                        )
                                    conn.commit()
                                    print(f"  [DB] Successfully saved {nn_name}")
                                except Exception as db_err:
                                    print(f"  [DB ERROR] Failed to save {nn_name}: {db_err}")
                                finally:
                                    conn.close()

                # Primary source: run_summary model_results
                best_test, best_train = _best_accuracy(run_summary)

                # Fallback / supplement: read archived eval_info files from disk
                # (they contain train_accuracy recorded by the trainer)
                if best_train == 0.0:
                    disk_test, disk_train = _read_eval_info_accs(args.epochs, ds)
                    if disk_test > best_test:
                        best_test = disk_test
                    best_train = disk_train

                ds_best[ds]  = best_test
                ds_train[ds] = best_train
                print(
                    f"  >> nb={nb} | {_fmt_prm(prm)} | {ds}: "
                    f"test={best_test:.4f}  train={best_train:.4f}"
                )

            avg       = sum(ds_best.values())  / len(ds_best)  if ds_best  else 0.0
            avg_train = sum(ds_train.values()) / len(ds_train) if ds_train else 0.0
            all_rows.append({
                "nb": nb, "prm": prm,
                "ds_best": ds_best, "ds_train": ds_train,
                "avg": avg, "avg_train": avg_train,
            })

    # Save full results JSON
    out_path = Path(nngpt_dir) / "llm" / date_str / time_str / f"param_sweep_{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n  Full results saved -> {out_path}")

    # Print ranked table
    _print_table(all_rows, args.datasets)


if __name__ == "__main__":
    main()

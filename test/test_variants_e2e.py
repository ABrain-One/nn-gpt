"""End-to-end: generate every variant for 10 random LEMUR nets, then evaluate.

Pipeline under test:
    1. pick N (default 10) random models from the NN dataset
    2. NNVariants generates every loss/optimizer variant for each (full grid)
    3. EvalVariants trains/evaluates each generated variant on a dataset
    4. every error — a generation skip or an evaluation failure — is collected and
       written to a JSON report so nothing fails silently

This is a heavy integration test (it trains networks and needs a populated LEMUR
DB plus the eval environment). It is **skipped automatically** when the DB is
empty. Run it directly:

    python test/test_variants_e2e.py
    python test/test_variants_e2e.py --n 10 --dataset cifar-10 --epochs 1

or under pytest (opt in with RUN_E2E=1, otherwise skipped):

    RUN_E2E=1 pytest test/test_variants_e2e.py
"""
import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ab.gpt.brute.loss_opt.NNVariants as NNV
import ab.gpt.brute.loss_opt.EvalVariants as EV
from ab.gpt.util.Const import epoch_dir, synth_dir
from ab.gpt.brute.loss_opt.VariantGen import iter_variants

N_NETS = 10
OUT_EPOCH = 0
DATASET = "cifar-10"
TRAIN_EPOCHS = 1
TASK = "img-classification"


def select_random_nns(n: int, task: str, seed: int | None = None) -> list[tuple[str, str]]:
    """Return up to ``n`` randomly chosen ``(name, code)`` pairs.

    Uses the same retrieval as NNVariants (DB with package-file fallback).
    """
    all_nets = NNV.fetch_nets(task)
    if not all_nets:
        return []
    rng = random.Random(seed)
    rng.shuffle(all_nets)
    return all_nets[:n]


def run_e2e(n=N_NETS, out_epoch=OUT_EPOCH, dataset=DATASET,
            train_epochs=TRAIN_EPOCHS, task=TASK, seed=None) -> dict | None:
    report = {
        "selected_nns": [],
        "generation_skips": [],
        "evaluation_errors": [],
        "generated": 0,
        "evaluated_ok": 0,
    }

    models = select_random_nns(n, task, seed)
    names = [name for name, _ in models]
    report["selected_nns"] = names
    if not models:
        print("No nets found — cannot run e2e.")
        return None
    print(f"Selected {len(models)} nets: {names}")

    # --- 1+2. record which variants couldn't be generated ---
    for nn_name, src in models:
        for loss_name, optim_name, _new_src, err in iter_variants(src):
            if err:
                report["generation_skips"].append(
                    {"nn": nn_name, "loss": loss_name, "optimizer": optim_name, "reason": err}
                )

    # actually write the variants to disk (full grid, fresh dir)
    report["generated"] = NNV.generate(models, None, None, out_epoch, clean=True)

    # --- 3. evaluate every generated variant with the dedicated EvalVariants ---
    synth = synth_dir(epoch_dir(out_epoch))
    try:
        EV.run(synth, [dataset], train_epochs, save_to_db=False, nn_name_prefix=None)
    except Exception:
        # Infrastructure-level failure (e.g. no GPU) — document it too.
        report["evaluation_errors"].append(
            {"model_id": "<EvalVariants.run>", "error": traceback.format_exc()}
        )

    # --- 4. tally results from the per-variant artifacts EvalVariants writes ---
    for b in sorted(synth.glob("B*")):
        meta = {}
        mp = b / "variant_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        err_files = sorted(b.glob("error_*.txt"))
        if err_files:
            for ef in err_files:
                report["evaluation_errors"].append({
                    "model_id": b.name,
                    "variant": meta,
                    "error": ef.read_text(encoding="utf-8")[:2000],
                })
        elif (b / "eval_info.json").exists():
            report["evaluated_ok"] += 1

    report_path = epoch_dir(out_epoch) / "variants_e2e_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"E2E report written to: {report_path}")
    print(f"  selected nets    : {len(report['selected_nns'])}")
    print(f"  generation skips : {len(report['generation_skips'])}")
    print(f"  variants written : {report['generated']}")
    print(f"  evaluated OK     : {report['evaluated_ok']}")
    print(f"  evaluation errors: {len(report['evaluation_errors'])}")
    print(f"{'=' * 60}\n")
    return report


# ---------- pytest entry (opt-in; skipped by default) ----------
def test_variants_e2e():
    import pytest  # type: ignore

    if not os.environ.get("RUN_E2E"):
        pytest.skip("heavy integration test; set RUN_E2E=1 to run")
    if not select_random_nns(N_NETS, TASK):
        pytest.skip("No nets found — check nn-dataset installation")

    report = run_e2e()
    assert report is not None
    assert report["generated"] > 0, "no variants were generated"


def main():
    p = argparse.ArgumentParser(description="E2E: generate variants for N random nets, then evaluate.")
    p.add_argument("--n", type=int, default=N_NETS, help="number of random nets to pick")
    p.add_argument("--out_epoch", type=int, default=OUT_EPOCH)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="nn_train_epochs per variant")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible net selection")
    args = p.parse_args()
    report = run_e2e(args.n, args.out_epoch, args.dataset, args.epochs, TASK, args.seed)
    sys.exit(0 if report is not None else 1)


if __name__ == "__main__":
    main()

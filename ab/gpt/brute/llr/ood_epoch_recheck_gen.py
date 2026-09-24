"""
OOD epoch/seed re-check (closes the gap the OOD scale-up left open).

Why this exists
----------------
`ood_scale_up.py` proved the fine-tuned LLM generator emits valid, correctly-
scoped LLR edits on unseen model families and that ~50% of evaluated
candidates improve over baseline — but ONLY at 1 epoch, and ONLY against
plain vanilla baseline (no `llr_uniform` control). The epoch/seed ablation
(`epoch_ablation_gen.py` / RESULT in out/nngpt/epoch_ablation_results.csv)
showed that exact 1-epoch/no-control combination is precisely what made the
brute-force sweep's "wins" look real when they weren't: every architecture
that won at 1 epoch decayed to noise or reversed to a loss by epoch 10 once
compared against a fair `llr_uniform` control. This script asks the same
question of the fine-tuned generator's own outputs: do ITS apparent 1-epoch
OOD wins survive to epoch 10 against the same fair control?

Design
----------------------------------------------------------------------------
6 held-out cifar-100 targets from the completed OOD run (out/nngpt/
ood_results_mem_c5.csv): the 4 largest 1-epoch gains (all pretrained-backbone
family, where the OOD signal concentrated) + 2 near-zero-delta targets as a
negative control (checks the "no signal" cases correctly stay null).
2 conditions per target:
  - 'generated': the LLM's own delta-applied code, already on disk at
    out/nngpt/llm/epoch/A950/synth_nn/B{idx}/new_nn.py (mem_c5 adapter,
    the run's designated best checkpoint) — reused verbatim, no re-generation.
  - 'uniform':   the SAME target's original code with `llr_uniform` injected
    directly via inject_layerwise_lr() (n_groups=1, 1.0x — behaviourally
    vanilla but carries the same _llr_groups scaffold, the established fair
    control from the sweep/ablation methodology). No LLM call needed.
3 seeds x 10 epochs x per-target transform (matches each target's own OOD
eval transform, NOT a fixed one — these are heterogeneous held-out models).

Route C (seed tagging) reused verbatim from epoch_ablation_gen.py — proven
working on cifar-10/catalog architectures in the full ablation run; this is
the first time it's exercised on cifar-100 + arbitrary DB-sourced (non-
catalog) architectures, which is exactly why a small test subset should be
run first (see the paired ood_epoch_recheck_analyze.py + test YAML).

Usage
-----
    # full 36-dir generation (6 targets x 2 conditions x 3 seeds) into A960
    python -m ab.gpt.brute.llr.ood_epoch_recheck_gen

    # test subset (2 targets, 1 seed) into A961
    python -m ab.gpt.brute.llr.ood_epoch_recheck_gen --epoch-idx 961 --seeds 1 --targets 5,8

Then evaluate (NNEval reads A{epoch-idx}/synth_nn; no --transform flag — each
model's transform comes from its own dataframe.df prm, exactly as
ood_scale_up.py's candidates were evaluated):
    python -m ab.gpt.act.eval.Eval --only_epoch <idx> --nn_train_epochs 10 \\
        --dataset cifar-100 --task img-classification --metric acc \\
        --nn_name_prefix llroodep --save_to_db

Nothing here trains or touches the DB — it only reads the `nn` table (for the
uniform-control's original source) and writes model directories.
"""

import argparse
import csv
import json
import shutil
import sqlite3
from pathlib import Path

from ab.gpt.brute.llr.layerwise_lr import LAYERWISE_STRATEGIES as S1, inject_layerwise_lr
from ab.gpt.brute.llr.layerwise_lr2 import LAYERWISE_STRATEGIES as S2
from ab.gpt.brute.llr.layerwise_lr3 import LAYERWISE_STRATEGIES as S3

DB_PATH_DEFAULT = "/a/mm/db/ab.nn.db"
UNIFORM_STRATEGY = next(s for s in S1 if s['name'] == 'llr_uniform')

# (B_idx in the A950/mem_c5 OOD run, target nn name, family, generator strategy
# used, transform, 1-epoch baseline_acc, role) — from out/nngpt/ood_results_mem_c5.csv.
# The 4 largest-gain targets (pretrained-backbone families, where OOD signal
# concentrated) + 2 near-zero-delta targets as a negative control.
OOD_RECHECK_TARGETS = [
    (8,  'rl-bb-init-3cb905a2eaf46f79b977b26d7cc3e632',   'rl-bb-init',   'llr3_2grp_inv_05x',
     'bf-v1-RandomAdjustSharpnes_RandomAutocontrast', 0.4278, 'top_gain'),        # +0.2117 @1ep
    (5,  'rl-back-init-b2c20573d5f5c4a9760d2bd2c6d85966', 'rl-back-init', 'llr2_4grp_cos',
     'bf-v1-Pad_RandomCrop_5', 0.5687, 'top_gain'),                                # +0.1606 @1ep
    (6,  'rl-bb-init-37d3db9c7b79ddf35e3a21817ca53039',   'rl-bb-init',   'llr3_4grp_geo_splits',
     'bf-v1-CenterCrop_RandomGrayscale_6', 0.5063, 'top_gain'),                    # +0.0973 (A2) / -0.0432 (A5) @1ep — mixed, worth rechecking
    (4,  'rl-back-init-99633ffec4c8c19fb7e92c1491146868', 'rl-back-init', 'llr3_6grp_cos_inv',
     'bf-v1-Pad_RandomPosterize_CenterCrop', 0.6842, 'top_gain'),                  # +0.0215 @1ep
    (17, 'ga-mut-5',                                       'ga-mut',       'llr2_4grp_cos',
     'echo_flip', 0.2873, 'null_control'),                                         # +0.052 @1ep, near-zero family overall
    (22, 'alt-nn2-1a9eac176debdd5ca03b333b4fb6a1e9',       'alt-nn2',      'llr_4grp_lin',
     'bf-v1-Pad_RandomAutocontrast', 0.2654, 'null_control'),                      # +0.0315 @1ep, near-zero family overall
]

SOURCE_EPOCH_DIR = 950  # A950/synth_nn — mem_c5 OOD run, where 'generated' new_nn.py lives
DEFAULT_SEEDS = 3
DEFAULT_EPOCH_IDX = 960
DEFAULT_PREFIX = 'llroodep'
DATASET_NAME = 'cifar-100'


def _seed_tag(seed: int) -> str:
    return (
        f"\n\n# OOD epoch-recheck seed replicate — behaviourally inert, distinct checksum only\n"
        f"_replicate_seed = {seed}\n"
    )


def generate(epoch_idx: int, n_seeds: int, wanted_b: list, prefix: str, clean: bool,
             db_path: str, project_root: Path):
    out_dir = project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{epoch_idx}' / 'synth_nn'
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{SOURCE_EPOCH_DIR}' / 'synth_nn'

    if clean:
        removed = 0
        for d in out_dir.iterdir():
            if d.is_dir() and d.name.startswith(f'{prefix}_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing {prefix}_ dirs in {out_dir}")

    plan = [t for t in OOD_RECHECK_TARGETS if (not wanted_b or t[0] in wanted_b)]
    if not plan:
        raise SystemExit(f"No targets matched --targets {wanted_b!r}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    manifest = []
    model_idx = 1
    print(f"Output dir : {out_dir}")
    print(f"Targets    : {[t[1] for t in plan]}")
    print(f"Seeds      : {n_seeds}")
    print(f"Dataset    : {DATASET_NAME} (per-target transform)")
    print()

    for b_idx, nn_name, family, strategy, transform, baseline_acc, role in plan:
        # 'generated' condition: the LLM's own delta-applied code, already on disk.
        gen_path = src_dir / f'B{b_idx}' / 'new_nn.py'
        if not gen_path.exists():
            print(f"[SKIP] {nn_name}: no generated code at {gen_path}")
            continue
        generated_code = gen_path.read_text(encoding='utf-8')

        # 'uniform' condition: same target's ORIGINAL code, llr_uniform injected fresh.
        row = cur.execute("SELECT code FROM nn WHERE name=?", (nn_name,)).fetchone()
        if not row:
            print(f"[SKIP] {nn_name}: not in nn table")
            continue
        original_code = row[0]
        uniform_code = inject_layerwise_lr(original_code, UNIFORM_STRATEGY)
        if uniform_code is None:
            print(f"[SKIP] {nn_name}: llr_uniform injection failed (no matching train_setup/optimizer pattern)")
            continue

        conditions = [('generated', generated_code), ('uniform', uniform_code)]
        for condition, base_code in conditions:
            for seed in range(n_seeds):
                code = base_code + _seed_tag(seed)
                name = f"{prefix}_{model_idx:04d}"
                mdir = out_dir / name
                mdir.mkdir(parents=True, exist_ok=True)
                (mdir / 'new_nn.py').write_text(code, encoding='utf-8')

                import pandas as pd
                pd.Series({
                    'nn': nn_name, 'accuracy': baseline_acc, 'epoch': 1,
                    'dataset': DATASET_NAME, 'task': 'img-classification', 'metric': 'acc',
                    'prm': {'transform': transform},
                }).to_pickle(str(mdir / 'dataframe.df'))

                (mdir / 'model_meta.txt').write_text(
                    '\n'.join([
                        f"target_nn: {nn_name}", f"family: {family}", f"role: {role}",
                        f"condition: {condition}", f"strategy: {strategy if condition == 'generated' else 'llr_uniform'}",
                        f"seed: {seed}", f"dataset: {DATASET_NAME}", f"transform: {transform}",
                        f"baseline_acc_1ep: {baseline_acc}",
                    ]) + '\n', encoding='utf-8',
                )

                manifest.append({
                    'model_dir': name, 'target_nn': nn_name, 'family': family, 'role': role,
                    'condition': condition, 'strategy': strategy if condition == 'generated' else 'llr_uniform',
                    'seed': seed, 'dataset': DATASET_NAME, 'transform': transform,
                    'baseline_acc_1ep': baseline_acc,
                })
                model_idx += 1
        print(f"[OK] {nn_name} ({role}): generated={strategy}, uniform=llr_uniform")

    con.close()
    if not manifest:
        raise SystemExit("No models generated — nothing to write.")

    manifest_path = project_root / 'out' / 'nngpt' / f'ood_epoch_recheck_manifest_A{epoch_idx}.csv'
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(manifest)} model dirs into {out_dir}")
    print(f"Manifest : {manifest_path}")
    print(f"Evaluate : python -m ab.gpt.act.eval.Eval --only_epoch {epoch_idx} "
          f"--nn_train_epochs 10 --dataset {DATASET_NAME} --task img-classification "
          f"--metric acc --nn_name_prefix {prefix} --save_to_db")
    print(f"{'=' * 60}")
    return len(manifest)


def main():
    p = argparse.ArgumentParser(description="Generate the OOD epoch/seed re-check model dirs.")
    p.add_argument('--epoch-idx', type=int, default=DEFAULT_EPOCH_IDX)
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--targets', type=str, default='',
                    help="Comma-separated B-indices subset (default: all 6), e.g. --targets 5,8")
    p.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
    p.add_argument('--db', type=str, default=DB_PATH_DEFAULT)
    p.add_argument('--no-clean', dest='clean', action='store_false')
    args = p.parse_args()
    wanted_b = [int(x) for x in args.targets.split(',') if x.strip()] if args.targets else []
    project_root = Path(__file__).resolve().parents[4]
    generate(args.epoch_idx, args.seeds, wanted_b, args.prefix, args.clean, args.db, project_root)


if __name__ == '__main__':
    main()

"""
Epoch/seed ablation generator (Route C — self-contained, no shared-pipeline edits).

Purpose
-------
The brute-force LLR sweep was run at 1 epoch, 1 seed. Layerwise-LR is a
multi-epoch training-dynamics mechanism, so a 1-epoch score structurally
penalises the low-backbone-LR / freeze strategies and cannot tell a real
improvement from run-to-run noise (measured ~±1.5 acc pts at 1 epoch). This
script emits a small, curated set of models so we can re-evaluate them at
several epoch checkpoints and several seeds, then check whether the 1-epoch
winners survive.

Design
---------------------------------------------------------------------
6 architectures x {winning strategy, llr_uniform control} x N seeds. Trained to
10 epochs; the {1,3,10} checkpoints are read from the single run's per-epoch
`stat` rows. cifar-10 / norm_64_flip (matching the original sweep exactly, so
the epoch-1 numbers reproduce).

Route C — how seeds work without touching NNEval
------------------------------------------------
NNEval dedups on uuid4(code): a 2nd run of an identical model is rejected as
"NN already exists" when save_to_db=True (ab/gpt/util/Eval.py:106-107). So each
seed copy carries one INERT trailing line `_replicate_seed = k` — it changes the
checksum (distinct DB row) but has zero effect on training. The k copies of a
model therefore differ only by SGD nondeterminism = genuine seed replicates,
stored as k separate `llrep-<checksum>` rows. No shared-pipeline change needed.

Usage
-----
    # full 60-dir generation (6 archs x 2 conditions x 5 seeds) into A900/synth_nn
    python -m ab.gpt.brute.llr.epoch_ablation_gen

    # test subset (DPN68 only, 2 seeds) into A901/synth_nn
    python -m ab.gpt.brute.llr.epoch_ablation_gen --epoch-idx 901 --seeds 2 --archs DPN68

Then evaluate (NNEval reads A{epoch-idx}/synth_nn):
    python -m ab.gpt.act.eval.Eval --only_epoch 900 --nn_train_epochs 10 \
        --dataset cifar-10 --transform norm_64_flip --task img-classification \
        --metric acc --nn_name_prefix llrep --save_to_db

Nothing here trains or touches the DB — it only writes model directories.
"""

import argparse
import csv
import shutil
from pathlib import Path

# Reuse the exact helpers the brute-force sweep used, so generated code /
# hyperparameters are identical to the original models being re-tested.
from ab.gpt.brute.llr.layerwise_lr import (
    LAYERWISE_STRATEGIES as STRAT_LLR,
    inject_layerwise_lr as inject_llr,
    find_nn_source_dir,
    get_supported_hp_from_source,
    build_hp_dict,
    read_architecture_source,
)
from ab.gpt.brute.llr.layerwise_lr2 import (
    LAYERWISE_STRATEGIES as STRAT_LLR2,
    inject_layerwise_lr as inject_llr2,
)
from ab.gpt.brute.llr.layerwise_lr3 import (
    LAYERWISE_STRATEGIES as STRAT_LLR3,
    inject_layerwise_lr as inject_llr3,
)

# ---------------------------------------------------------------------------
# Configuration — the locked design
# ---------------------------------------------------------------------------
# (architecture, winning cifar-10 strategy, role). Winning strategies from
# out/nngpt/best_strategy_per_arch_dataset.csv. Roles are for reporting only.
ABLATION_ARCHS = [
    ('GoogLeNet', 'llr2_cyclic_001x',   'responder'),        # +0.0999 (~7 sigma)
    ('DPN68',     'llr_2grp_01x',       'responder'),        # +0.1206 (~8 sigma)
    ('ConvNeXt',  'llr2_2grp_inv_001x', 'responder'),        # +0.0321 (~2 sigma)
    ('ICNet',     'llr3_2grp_inv_05x',  'lucky_hit'),        # +0.0063 (<1 sigma)
    ('UNet2D',    'llr3_2grp_05x',      'lucky_hit'),        # +0.0215 (~1.5 sigma)
    ('DenseNet',  'llr3_2grp_inv_05x',  'negative_control'), # best is -0.0084, 0/38 wins
]
CONTROL_STRATEGY = 'llr_uniform'   # single group @ 1.0x == vanilla (from layerwise_lr)

DEFAULT_SEEDS = 5
DEFAULT_EPOCH_IDX = 900            # A900/synth_nn — dedicated, avoids clobbering A0..A20
DEFAULT_PREFIX = 'llrep'
DATASET_CFG = {
    'name': 'cifar-10',
    'task': 'img-classification',
    'metric': 'acc',
    'transform': 'norm_64_flip',   # MUST match the original sweep (777/777 cifar-10 rows)
}


# ---------------------------------------------------------------------------
# Strategy lookup across the three generator modules
# ---------------------------------------------------------------------------
def _build_strategy_index():
    """name -> (strategy_dict, inject_fn), spanning llr_/llr2_/llr3_ modules."""
    idx = {}
    for strat_list, inject_fn in (
        (STRAT_LLR, inject_llr),
        (STRAT_LLR2, inject_llr2),
        (STRAT_LLR3, inject_llr3),
    ):
        for strat in strat_list:
            idx[strat['name']] = (strat, inject_fn)
    return idx


def _seed_tag(seed: int) -> str:
    """
    Inert trailing line that makes each seed copy a distinct checksum without
    changing behaviour. A module-level assignment (not a comment) so it survives
    any AST round-trip a downstream tool might apply before hashing.
    """
    return (
        f"\n\n# epoch-ablation seed replicate — behaviourally inert, distinct checksum only\n"
        f"_replicate_seed = {seed}\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(epoch_idx: int, n_seeds: int, archs: list, prefix: str, clean: bool):
    project_root = Path(__file__).resolve().parents[4]  # nn-gpt root
    out_dir = project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{epoch_idx}' / 'synth_nn'
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        removed = 0
        for d in out_dir.iterdir():
            if d.is_dir() and d.name.startswith(f'{prefix}_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing {prefix}_ dirs in {out_dir}")

    strat_index = _build_strategy_index()
    src_dir = find_nn_source_dir()

    # Filter arch list if a subset was requested
    wanted = {a.strip() for a in archs} if archs else None
    plan = [row for row in ABLATION_ARCHS if (wanted is None or row[0] in wanted)]
    if not plan:
        raise SystemExit(f"No architectures matched --archs {archs!r}")

    manifest = []
    model_idx = 1

    print(f"Output dir : {out_dir}")
    print(f"Archs      : {[r[0] for r in plan]}")
    print(f"Seeds      : {n_seeds}")
    print(f"Dataset    : {DATASET_CFG['name']} / {DATASET_CFG['transform']}")
    print()

    for arch, win_strategy, role in plan:
        source_code = read_architecture_source(src_dir, arch)
        if source_code is None:
            print(f"[SKIP] {arch}: source not found")
            continue
        if get_supported_hp_from_source(source_code) is None:
            print(f"[SKIP] {arch}: supported_hyperparameters() not found")
            continue

        conditions = [('winning', win_strategy), ('control', CONTROL_STRATEGY)]
        for condition, strat_name in conditions:
            if strat_name not in strat_index:
                print(f"[SKIP] {arch}/{strat_name}: strategy not found in any module")
                continue
            strat, inject_fn = strat_index[strat_name]
            base_code = inject_fn(source_code, strat)
            if base_code is None:
                print(f"[SKIP] {arch}/{strat_name}: injection failed")
                continue

            for seed in range(n_seeds):
                code = base_code + _seed_tag(seed)
                name = f"{prefix}_{model_idx:04d}"
                mdir = out_dir / name
                mdir.mkdir(parents=True, exist_ok=True)
                (mdir / 'new_nn.py').write_text(code, encoding='utf-8')

                hp = build_hp_dict(arch, DATASET_CFG)
                import json
                (mdir / 'hp.txt').write_text(json.dumps(hp, indent=2), encoding='utf-8')

                # dataframe.df — NNEval fallback (we also pass --dataset/--transform)
                import pandas as pd
                pd.Series({
                    'nn': f"{arch}-{strat_name}",
                    'task': DATASET_CFG['task'],
                    'dataset': DATASET_CFG['name'],
                    'metric': DATASET_CFG['metric'],
                    'prm': hp,
                }).to_pickle(str(mdir / 'dataframe.df'))

                (mdir / 'model_meta.txt').write_text(
                    '\n'.join([
                        f"architecture: {arch}",
                        f"strategy: {strat_name}",
                        f"condition: {condition}",
                        f"role: {role}",
                        f"seed: {seed}",
                        f"n_groups: {strat['n_groups']}",
                        f"multipliers: {strat['multipliers']}",
                        f"dataset: {DATASET_CFG['name']}",
                        f"transform: {DATASET_CFG['transform']}",
                        f"description: {strat['description']}",
                    ]) + '\n',
                    encoding='utf-8',
                )

                manifest.append({
                    'model_dir': name,
                    'architecture': arch,
                    'strategy': strat_name,
                    'condition': condition,
                    'role': role,
                    'seed': seed,
                    'dataset': DATASET_CFG['name'],
                    'transform': DATASET_CFG['transform'],
                })
                model_idx += 1

        print(f"[OK] {arch} ({role}): winning={win_strategy}, control={CONTROL_STRATEGY}")

    # Manifest — analysis maps uuid4(new_nn.py) -> these (arch, strategy, seed)
    manifest_path = project_root / 'out' / 'nngpt' / f'epoch_ablation_manifest_A{epoch_idx}.csv'
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(manifest)} model dirs into {out_dir}")
    print(f"Manifest : {manifest_path}")
    print(f"Evaluate : python -m ab.gpt.act.eval.Eval --only_epoch {epoch_idx} "
          f"--nn_train_epochs 10 --dataset cifar-10 --transform norm_64_flip "
          f"--task img-classification --metric acc --nn_name_prefix {prefix} --save_to_db")
    print(f"{'=' * 60}")
    return len(manifest)


def main():
    p = argparse.ArgumentParser(description="Generate the LLR epoch/seed ablation model dirs.")
    p.add_argument('--epoch-idx', type=int, default=DEFAULT_EPOCH_IDX,
                   help=f"A{{N}}/synth_nn dir to write into (default {DEFAULT_EPOCH_IDX}).")
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS,
                   help=f"Seed replicates per (arch, condition) (default {DEFAULT_SEEDS}).")
    p.add_argument('--archs', type=str, default='',
                   help="Comma-separated arch subset (default: all 6).")
    p.add_argument('--prefix', type=str, default=DEFAULT_PREFIX,
                   help=f"Model dir / DB name prefix (default {DEFAULT_PREFIX}).")
    p.add_argument('--no-clean', dest='clean', action='store_false',
                   help="Do not remove existing <prefix>_ dirs first.")
    args = p.parse_args()
    archs = [a for a in args.archs.split(',') if a.strip()] if args.archs else []
    generate(args.epoch_idx, args.seeds, archs, args.prefix, args.clean)


if __name__ == '__main__':
    main()

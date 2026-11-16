"""
Collect every per-seed TNNLS run into one long-format CSV (one row per run x epoch).

Sources are the per-job collector outputs in out/nngpt (tnnls_per_epoch_*.csv,
epoch_ablation_per_seed.csv, ood_recheck2_per_run_*.csv) plus the first
out-of-distribution re-check, read straight from its model dirs (A960/A961).

Usage:
    python -m ab.gpt.brute.llr.build_per_seed_csv --out per_seed_runs.csv
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
D = PROJECT_ROOT / 'out' / 'nngpt'

# collector file -> experiment label (see the paper's Sec. III)
FILES = {
    'epoch_ablation_per_seed.csv': 'ablation_10ep',
    'tnnls_per_epoch_dpn68_topup.csv': 'ablation_10ep',
    'tnnls_per_epoch_smokeB.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_googlenet.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_convnext.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_densenet.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_densenet_b.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_densenet_c.csv': 'repl_100ep',
    'tnnls_per_epoch_e4_googlenet_matched.csv': 'ratematched_100ep',
    'tnnls_per_epoch_rerun_googlenet_matched.csv': 'ratematched_100ep',
    'tnnls_per_epoch_e4_convnext_matched_a.csv': 'ratematched_100ep',
    'tnnls_per_epoch_e4_convnext_matched_b.csv': 'ratematched_100ep',
    'tnnls_per_epoch_e2a.csv': 'structure_10ep',
    'tnnls_per_epoch_e2b.csv': 'baserate_10ep',
    'tnnls_per_epoch_e2b_top_a.csv': 'baserate_10ep',
    'tnnls_per_epoch_e2b_top_b.csv': 'baserate_10ep',
    'tnnls_per_epoch_regmean_googlenet.csv': 'regmean_10ep',
    'tnnls_per_epoch_regmean_convnext.csv': 'regmean_10ep',
    'tnnls_per_epoch_regmean_dpn68.csv': 'regmean_10ep',
    'tnnls_per_epoch_regmean_icnet.csv': 'regmean_10ep',
    'tnnls_per_epoch_regmean_unet2d.csv': 'regmean_10ep',
    'tnnls_per_epoch_regmean_densenet.csv': 'regmean_10ep',
    'tnnls_per_epoch_rerun_misc.csv': 'ablation_10ep',     # DPN68 seed 5
    'tnnls_per_epoch_heldout_maxvit.csv': 'heldout_10ep',
    'tnnls_per_epoch_heldout_squeezenet.csv': 'heldout_10ep',
    'tnnls_per_epoch_heldout_squeezenet_matched.csv': 'heldout_10ep',
}
# The generator labels the regression-to-the-mean arms 'strat'; these are the picks
# recorded in the job YAMLs (worst sweep strategy, and one drawn with Random('regmean-<Arch>')).
REGMEAN = {
    'GoogLeNet': {'llr2_3grp_80_10_10': 'worst', 'llr3_2grp_95_5': 'random'},
    'ConvNeXt': {'llr_2grp_75pct': 'worst', 'llr3_5grp_plateau': 'random'},
    'DPN68': {'llr2_6grp_exp': 'worst', 'llr_5grp_exp': 'random'},
    'ICNet': {'llr2_2grp_90_10': 'worst', 'llr_2grp_01x': 'random'},
    'UNet2D': {'llr2_6grp_exp': 'worst', 'llr3_6grp_cos_inv': 'random'},
    'DenseNet': {'llr2_3grp_80_10_10': 'worst', 'llr_2grp_01x': 'random'},
}
COLS = ['experiment', 'source', 'architecture', 'condition', 'strategy', 'base_lr', 'seed',
        'epoch', 'accuracy', 'train_loss', 'test_loss', 'train_accuracy']


def per_seed():
    parts = []
    for f, exp in FILES.items():
        d = pd.read_csv(D / f)
        if 'base_lr' not in d.columns:
            d['base_lr'] = 0.01
        d['experiment'] = exp
        d['source'] = f + ':' + d['model_dir'].astype(str)
        if exp == 'regmean_10ep':
            d['condition'] = [REGMEAN[x][y] for x, y in zip(d.architecture, d.strategy)]
        parts.append(d.reindex(columns=COLS))
    return pd.concat(parts, ignore_index=True)


def ood():
    rows = []
    for slot in (960, 961):
        man = {r['model_dir']: r for r in csv.DictReader(open(D / f'ood_epoch_recheck_manifest_A{slot}.csv'))}
        synth = D / 'llm' / 'epoch' / f'A{slot}' / 'synth_nn'
        for name, m in man.items():
            for p in (synth / name).glob('*.json'):
                if p.stem.isdigit():
                    t = json.loads(p.read_text())[0]
                    rows.append(dict(target=m['target_nn'], role=m['role'], condition=m['condition'],
                                     strategy=m['strategy'], seed=int(m['seed']), epoch=int(p.stem),
                                     accuracy=t['accuracy'], stored_baseline=m['baseline_acc_1ep'],
                                     source=f'A{slot}:{name}', batch='recheck1'))
    for tag in ('ood_recheck2_a', 'ood_recheck2_b'):
        d = pd.read_csv(D / f'ood_recheck2_per_run_{tag}.csv')
        for r in d.itertuples():
            rows.append(dict(target=r.target_nn, role=r.role, condition=r.condition, strategy=r.strategy,
                             seed=r.seed, epoch=r.epoch, accuracy=r.accuracy,
                             stored_baseline=r.baseline_acc_1ep, source=f'{tag}:{r.model_dir}', batch='recheck2'))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='per_seed_runs.csv')
    ap.add_argument('--ood-out', default='ood_fresh_control_runs.csv')
    a = ap.parse_args()
    p = per_seed()
    p.to_csv(a.out, index=False)
    o = ood()
    o.to_csv(a.ood_out, index=False)
    print(f"{a.out}: {len(p)} rows, {p.groupby(['source']).ngroups} runs")
    print(f"{a.ood_out}: {len(o)} rows, {o.source.nunique()} runs")


if __name__ == '__main__':
    main()

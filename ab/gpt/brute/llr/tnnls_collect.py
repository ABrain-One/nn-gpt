"""
Collect per-epoch results for a TNNLS-revision slot straight from the model dirs.

NNEval (save_to_db) leaves one full nn-dataset trial JSON per epoch at
<model_dir>/<dataset>/<N>.json (N = epoch), so results do not depend on reading the
1.3 GB DB and cannot be affected by another job writing to it. Reads only files.

Usage:
    python -m ab.gpt.brute.llr.tnnls_collect --epoch-idx 970 --tag smokeA
Writes out/nngpt/tnnls_per_epoch_<tag>.csv (one row per model x epoch) and prints, per
(architecture, condition, base_lr), the accuracy at the checkpoint epochs.
"""

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIELDS = ['architecture', 'condition', 'strategy', 'recipe', 'base_lr', 'seed', 'model_dir', 'epoch',
          'accuracy', 'lr_in_prm', 'lr_now', 'train_loss', 'test_loss', 'train_accuracy',
          'gradient_norm', 'duration_seconds']


def collect(epoch_idx: int):
    manifest = PROJECT_ROOT / 'out' / 'nngpt' / f'tnnls_manifest_A{epoch_idx}.csv'
    synth = PROJECT_ROOT / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{epoch_idx}' / 'synth_nn'
    rows, missing = [], []
    for m in csv.DictReader(open(manifest)):
        # Older nn-dataset wrote <model_dir>/<dataset>/N.json, newer writes <model_dir>/N.json
        # (N = epoch). Only numeric stems: eval_info.json / eval_summary.json also live there.
        mdir = synth / m['model_dir']
        ddir = mdir / m['dataset'] if (mdir / m['dataset']).is_dir() else mdir
        files = sorted((p for p in ddir.glob('*.json') if p.stem.isdigit()), key=lambda p: int(p.stem))
        if not files:
            missing.append(m['model_dir'])
            continue
        for f in files:
            payload = json.loads(f.read_text(encoding='utf-8'))
            rec = payload[0] if isinstance(payload, list) else payload
            ts = rec.get('train_stat', {})
            rows.append({
                'architecture': m['architecture'], 'condition': m['condition'],
                'strategy': m['strategy'], 'recipe': m.get('recipe', 'const'),
                'base_lr': m['base_lr'], 'seed': m['seed'],
                'model_dir': m['model_dir'], 'epoch': int(f.stem), 'accuracy': rec.get('accuracy'),
                'lr_in_prm': rec.get('lr'), 'lr_now': ts.get('lr'), 'train_loss': ts.get('train_loss'),
                'test_loss': ts.get('test_loss'), 'train_accuracy': ts.get('train_accuracy'),
                'gradient_norm': ts.get('gradient_norm'), 'duration_seconds': rec.get('duration_seconds'),
            })
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch-idx', type=int, required=True)
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--checkpoints', type=str, default='1,3,10,30,100')
    a = ap.parse_args()
    rows, missing = collect(a.epoch_idx)
    out = PROJECT_ROOT / 'out' / 'nngpt' / f'tnnls_per_epoch_{a.tag}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} model-epoch rows -> {out}")
    if missing:
        print(f"[WARN] {len(missing)} models have no per-epoch JSON: {missing[:6]}")

    cps = [int(x) for x in a.checkpoints.split(',')]
    cell = defaultdict(list)
    for r in rows:
        cell[(r['architecture'], r['strategy'], r['base_lr'], r['seed'], int(r['epoch']))].append(
            (float(r['accuracy']), r['lr_in_prm'], r['train_loss']))
    keys = sorted({k[:4] for k in cell})
    print(f"\n{'arch':10s} {'strategy':28s} {'lr':>6s} {'seed':>5s} {'lr(prm)':>8s} " +
          ' '.join(f"acc@{c:<4d}" for c in cps) + '  last_loss  max_ep')
    for k in keys:
        eps = sorted(e for (*kk, e) in cell if tuple(kk) == k)
        accs = [f"{cell[k + (c,)][0][0]:8.4f}" if k + (c,) in cell else f"{'-':>8s}" for c in cps]
        last = eps[-1]
        print(f"{k[0]:10s} {k[1]:28s} {k[2]:>6s} {k[3]:>5s} {str(cell[k + (last,)][0][1]):>8s} "
              + ' '.join(accs) + f"  {cell[k + (last,)][0][2]}  {last}")


if __name__ == '__main__':
    main()

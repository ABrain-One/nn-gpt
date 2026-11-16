"""
Collect results for ood_recheck2_gen.py straight from the model dirs (no DB dependency,
same approach as tnnls_collect.py): reads <model_dir>/<dataset>/N.json or
<model_dir>/N.json per generated/uniform pair and reports the matched-control delta.

Usage:
    python -m ab.gpt.brute.llr.ood_recheck2_collect --epoch-idx 990 --tag ood2a
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIELDS = ['target_nn', 'family', 'role', 'condition', 'strategy', 'seed', 'model_dir',
          'epoch', 'accuracy', 'baseline_acc_1ep']


def collect(epoch_idx: int):
    manifest = PROJECT_ROOT / 'out' / 'nngpt' / f'ood_epoch_recheck_manifest_A{epoch_idx}.csv'
    synth = PROJECT_ROOT / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{epoch_idx}' / 'synth_nn'
    rows, missing = [], []
    for m in csv.DictReader(open(manifest)):
        mdir = synth / m['model_dir']
        ddir = mdir / m['dataset'] if (mdir / m['dataset']).is_dir() else mdir
        files = sorted((p for p in ddir.glob('*.json') if p.stem.isdigit()), key=lambda p: int(p.stem))
        if not files:
            missing.append(m['model_dir'])
            continue
        for f in files:
            payload = json.loads(f.read_text(encoding='utf-8'))
            rec = payload[0] if isinstance(payload, list) else payload
            rows.append({
                'target_nn': m['target_nn'], 'family': m['family'], 'role': m['role'],
                'condition': m['condition'], 'strategy': m['strategy'], 'seed': m['seed'],
                'model_dir': m['model_dir'], 'epoch': int(f.stem), 'accuracy': rec.get('accuracy'),
                'baseline_acc_1ep': m['baseline_acc_1ep'],
            })
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch-idx', type=int, required=True)
    ap.add_argument('--tag', type=str, required=True)
    a = ap.parse_args()
    rows, missing = collect(a.epoch_idx)
    out = PROJECT_ROOT / 'out' / 'nngpt' / f'ood_recheck2_per_run_{a.tag}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    if missing:
        print(f"[WARN] {len(missing)} models have no per-epoch JSON: {missing[:6]}")

    by_target = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_target[r['target_nn'], r['baseline_acc_1ep']][r['condition']].append(float(r['accuracy']))
    print(f"\n{'target':46s} {'stored':>8s} {'uniform':>8s} {'generated':>10s} {'scale-up_d':>11s} {'recheck_d':>10s}")
    for (target, stored), conds in sorted(by_target.items()):
        u = st.mean(conds['uniform']) if conds.get('uniform') else float('nan')
        g = st.mean(conds['generated']) if conds.get('generated') else float('nan')
        stored = float(stored)
        print(f"{target[:44]:46s} {stored:8.4f} {u:8.4f} {g:10.4f} {g-stored:+11.4f} {g-u:+10.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Export a baseline-accuracy lookup CSV for the architectures used in LLR
fine-tuning: one row per (architecture, dataset, epoch, transform) combo,
with the vanilla accuracy the pipeline itself would use as the comparison
baseline.

Mirrors ab.gpt.brute.llr.llr_baselines.load_vanilla_baselines() exactly:
  - source 1 (preferred): the llr_uniform CONTROL run (single group, 1.0x LR),
    stored under an opaque llr-<hash> name and mapped back to its architecture
    via the metadata CSV;
  - source 2 (fallback):  the plain DB vanilla run stored under the bare
    architecture name (e.g. 'ResNet');
  - accuracy per key is MAX(accuracy) over all matching stat rows.

Use it to compare a generated candidate's eval result (e.g.
out/nngpt/llm/epoch/A3/synth_nn/B0/imagenette/1.json -> accuracy) against the
row with the same architecture (from the original_<Arch>.py filename), dataset
(the eval subdir name), epoch and transform (prm['transform'] in
eval_info.json / the <hp> block).

Stdlib-only; runs on the workstation or in the pod.

    python3 export_vanilla_baselines.py \
        [--db PATH] [--meta PATH] [--out PATH]
"""
import argparse
import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', default=str(REPO_ROOT / 'db' / 'ab.nn.db'))
    p.add_argument('--meta', default=str(REPO_ROOT / 'out' / 'nngpt' / 'llr_evaluated_with_meta.csv'))
    p.add_argument('--out', default=str(REPO_ROOT / 'out' / 'nngpt' / 'vanilla_baselines.csv'))
    p.add_argument('--epoch', type=int, default=1,
                   help='Keep only this training epoch (default 1, matching '
                        'nn_train_epochs=1 used for every generated LLR candidate). '
                        'Pass 0 to keep all epochs.')
    return p.parse_args()


def main():
    args = parse_args()

    # Architectures used in fine-tuning + hash->arch map for uniform controls
    archs, uniform_map = set(), {}
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            archs.add(row['architecture'])
            if row['strategy'] == 'llr_uniform':
                uniform_map[row['nn']] = row['architecture']
    archs = sorted(a for a in archs if a)
    print(f'{len(archs)} architectures, {len(uniform_map)} uniform-control models')

    con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    cur = con.cursor()

    epoch_clause = ''
    epoch_params = []
    if args.epoch:
        epoch_clause = ' AND epoch = ?'
        epoch_params = [args.epoch]

    def best_acc_lut(names, name_to_arch=None):
        """{(arch, dataset, epoch, transform): (max_acc, n_runs)}"""
        if not names:
            return {}
        qmarks = ','.join('?' * len(names))
        lut = {}
        for nn, ds, ep, tf, acc, n in cur.execute(
                f'SELECT nn, dataset, epoch, transform, MAX(accuracy), COUNT(*) '
                f'FROM stat WHERE nn IN ({qmarks}){epoch_clause} '
                f'GROUP BY nn, dataset, epoch, transform', list(names) + epoch_params):
            arch = name_to_arch.get(nn) if name_to_arch else nn
            if not arch:
                continue
            try:
                key = (arch, ds, int(ep), tf)
            except (TypeError, ValueError):
                key = (arch, ds, ep, tf)
            if key not in lut or acc > lut[key][0]:
                lut[key] = (acc, n)
        return lut

    vanilla = best_acc_lut(archs)
    uniform = best_acc_lut(list(uniform_map), uniform_map)
    con.close()

    rows = []
    for key in sorted(set(vanilla) | set(uniform)):
        arch, ds, ep, tf = key
        v = vanilla.get(key)
        u = uniform.get(key)
        # same priority as load_vanilla_baselines consumers: uniform control first
        best, source = (u[0], 'uniform_control') if u else (v[0], 'db_vanilla')
        rows.append({
            'architecture': arch,
            'dataset': ds,
            'epoch': ep,
            'transform': tf,
            'baseline_accuracy': best,
            'baseline_source': source,
            'db_vanilla_accuracy': v[0] if v else '',
            'db_vanilla_runs': v[1] if v else 0,
            'uniform_control_accuracy': u[0] if u else '',
            'uniform_control_runs': u[1] if u else 0,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()

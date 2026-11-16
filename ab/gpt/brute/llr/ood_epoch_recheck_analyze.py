"""
Analyse the OOD epoch/seed re-check (see ood_epoch_recheck_gen.py for design).

Maps each generated model dir -> its DB checksum (uuid4 of new_nn.py, exactly
as LEMUR/NNEval names it), joins the per-epoch `stat` rows, and reports, per
(target model, condition), accuracy mean +/- CI across seeds at each epoch
checkpoint {1,3,10}. Computes the generated-vs-uniform delta per epoch and
flags whether the OOD run's apparent 1-epoch win survives to epoch 10 —
the same question epoch_ablation_analyze.py asks of the brute-force sweep,
applied here to the fine-tuned generator's own OOD outputs.

Usage:
    python -m ab.gpt.brute.llr.ood_epoch_recheck_analyze --epoch-idx 960 --prefix llroodep
"""

import argparse
import csv
import math
import sqlite3
import statistics as st
from collections import defaultdict
from pathlib import Path

from ab.nn.util.Util import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS = (1, 3, 10)


def _ci95_halfwidth(vals):
    n = len(vals)
    if n < 2:
        return float('nan')
    sd = st.stdev(vals)
    se = sd / math.sqrt(n)
    t_table = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262}
    t = t_table.get(n, 1.96)
    return t * se


def _welch(a, b):
    diff = st.mean(a) - st.mean(b)
    if len(a) < 2 or len(b) < 2:
        return diff, float('nan'), float('nan')
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return diff, float('inf'), 0.0
    t = diff / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return diff, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch-idx', type=int, default=960)
    ap.add_argument('--prefix', type=str, default='llroodep')
    ap.add_argument('--db', type=str, default=str(PROJECT_ROOT / 'db' / 'ab.nn.db'))
    args = ap.parse_args()

    manifest_path = PROJECT_ROOT / 'out' / 'nngpt' / f'ood_epoch_recheck_manifest_A{args.epoch_idx}.csv'
    synth = PROJECT_ROOT / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{args.epoch_idx}' / 'synth_nn'
    rows = list(csv.DictReader(open(manifest_path)))
    print(f"Manifest: {manifest_path} ({len(rows)} models)")

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    acc = defaultdict(list)  # (target_nn, condition, epoch) -> [accuracy, ...]
    missing = []
    for r in rows:
        code = (synth / r['model_dir'] / 'new_nn.py').read_text(encoding='utf-8')
        nn_name = f"{args.prefix}-{uuid4(code)}"
        for ep in CHECKPOINTS:
            q = cur.execute(
                "SELECT accuracy FROM stat WHERE nn=? AND dataset=? AND metric='acc' AND epoch=?",
                (nn_name, r['dataset'], ep),
            ).fetchall()
            if q:
                acc[(r['target_nn'], r['condition'], ep)].append(q[-1][0])
            elif ep == 1:
                missing.append((r['target_nn'], r['condition'], r['seed'], nn_name))

    if missing:
        print(f"\n[WARN] {len(missing)} (target,condition,seed) have NO epoch-1 row yet "
              f"(not evaluated / still running). First few: {missing[:3]}")

    out = []
    targets = sorted({r['target_nn'] for r in rows})
    role_by_target = {r['target_nn']: r['role'] for r in rows}
    print(f"\n{'target':44s} {'ep':>3s} {'gen_mean':>9s} {'gen_ci':>7s} "
          f"{'uni_mean':>9s} {'uni_ci':>7s} {'delta':>8s} {'p~':>7s} {'verdict':>10s}")
    print('-' * 110)
    for target in targets:
        for ep in CHECKPOINTS:
            g = acc.get((target, 'generated', ep), [])
            u = acc.get((target, 'uniform', ep), [])
            if not g or not u:
                continue
            diff, t, p = _welch(g, u)
            if p != p:  # nan: fewer than 2 seeds on one side, no significance test possible
                verdict = f'n/a (n={len(g)}/{len(u)})'
            else:
                verdict = 'win' if diff > 0 and p < 0.05 else ('noise' if p >= 0.05 else 'LOSE')
            gm, um = st.mean(g), st.mean(u)
            gci, uci = _ci95_halfwidth(g), _ci95_halfwidth(u)
            print(f"{target[:44]:44s} {ep:3d} {gm:9.4f} {gci:7.4f} {um:9.4f} {uci:7.4f} "
                  f"{diff:+8.4f} {p:7.3f} {verdict:>10s}")
            out.append({
                'target_nn': target, 'role': role_by_target.get(target, ''), 'epoch': ep,
                'n_seeds_gen': len(g), 'n_seeds_uni': len(u),
                'gen_mean': round(gm, 4), 'gen_ci95': round(gci, 4),
                'uni_mean': round(um, 4), 'uni_ci95': round(uci, 4),
                'delta': round(diff, 4), 'welch_t': round(t, 3), 'p_approx': round(p, 4),
                'verdict': verdict,
            })

    print(f"\n{'=' * 60}\nEpoch-1 vs epoch-10 stability (the key question):")
    by_target = defaultdict(dict)
    for o in out:
        by_target[o['target_nn']][o['epoch']] = o
    for target in targets:
        e1, e10 = by_target[target].get(1), by_target[target].get(10)
        if e1 and e10:
            flip = 'RANKING FLIPPED' if (e1['delta'] > 0) != (e10['delta'] > 0) else 'stable sign'
            print(f"  {target[:44]:44s} ep1 {e1['delta']:+.4f} ({e1['verdict']:>6s})  ->  "
                  f"ep10 {e10['delta']:+.4f} ({e10['verdict']:>6s})   [{flip}]")

    res_path = PROJECT_ROOT / 'out' / 'nngpt' / 'ood_epoch_recheck_results.csv'
    if out:
        with open(res_path, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            wtr.writeheader()
            wtr.writerows(out)
        print(f"\nWrote {res_path}")
    con.close()


if __name__ == '__main__':
    main()

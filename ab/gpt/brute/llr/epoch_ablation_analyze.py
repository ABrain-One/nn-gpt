"""
Analyse the LLR epoch/seed ablation.

Maps each generated model dir -> its DB checksum (uuid4 of new_nn.py, exactly as
LEMUR/NNEval names it), joins the per-epoch `stat` rows, and reports, per
(architecture, condition), the accuracy mean +/- CI across seeds at each epoch
checkpoint {1,3,10}. Then computes the winning-vs-control delta per epoch and
flags whether the epoch-1 winner is still a winner at epoch 10.

Requires `ab.nn` (for uuid4) — run where the DB and package are available
(i.e. in the pod, or any env with nn-dataset installed). Reads nothing but the
manifest + DB; writes out/nngpt/epoch_ablation_results.csv.

Usage:
    python -m ab.gpt.brute.llr.epoch_ablation_analyze --epoch-idx 900 --prefix llrep
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
    """Rough 95% CI half-width for a small sample (t approx; normal fallback)."""
    n = len(vals)
    if n < 2:
        return float('nan')
    sd = st.stdev(vals)
    se = sd / math.sqrt(n)
    # t_0.975 for small df; fall back to 1.96 for larger n
    t_table = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262}
    t = t_table.get(n, 1.96)
    return t * se


def _welch(a, b):
    """Welch t-stat + approximate two-sided p (normal approx). Returns (diff, t, p)."""
    diff = st.mean(a) - st.mean(b)
    if len(a) < 2 or len(b) < 2:
        return diff, float('nan'), float('nan')
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return diff, float('inf'), 0.0
    t = diff / se
    # two-sided p via normal approx (conservative note: small-n, treat as indicative)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return diff, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch-idx', type=int, default=900)
    ap.add_argument('--prefix', type=str, default='llrep')
    ap.add_argument('--db', type=str, default=str(PROJECT_ROOT / 'db' / 'ab.nn.db'))
    args = ap.parse_args()

    manifest_path = PROJECT_ROOT / 'out' / 'nngpt' / f'epoch_ablation_manifest_A{args.epoch_idx}.csv'
    synth = PROJECT_ROOT / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{args.epoch_idx}' / 'synth_nn'
    rows = list(csv.DictReader(open(manifest_path)))
    print(f"Manifest: {manifest_path} ({len(rows)} models)")

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    # checksum -> DB accuracy per epoch
    # acc[(arch, condition, epoch)] -> list of accuracies across seeds
    acc = defaultdict(list)
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
                # one row expected per distinct-checksum seed; take the last if re-run
                acc[(r['architecture'], r['condition'], ep)].append(q[-1][0])
            elif ep == 1:
                missing.append((r['architecture'], r['condition'], r['seed'], nn_name))

    if missing:
        print(f"\n[WARN] {len(missing)} (arch,condition,seed) have NO epoch-1 row yet "
              f"(not evaluated / still running). First few: {missing[:3]}")

    # Per-cell summary + winning-vs-control delta per epoch
    out = []
    archs = sorted({r['architecture'] for r in rows})
    print(f"\n{'arch':12s} {'ep':>3s} {'win_mean':>9s} {'win_ci':>7s} "
          f"{'ctl_mean':>9s} {'ctl_ci':>7s} {'delta':>8s} {'p~':>7s} {'verdict':>10s}")
    print('-' * 84)
    for arch in archs:
        for ep in CHECKPOINTS:
            w = acc.get((arch, 'winning', ep), [])
            c = acc.get((arch, 'control', ep), [])
            if not w or not c:
                continue
            diff, t, p = _welch(w, c)
            verdict = 'win' if diff > 0 and p < 0.05 else ('noise' if p >= 0.05 else 'LOSE')
            wm, cm = st.mean(w), st.mean(c)
            wci, cci = _ci95_halfwidth(w), _ci95_halfwidth(c)
            print(f"{arch:12s} {ep:3d} {wm:9.4f} {wci:7.4f} {cm:9.4f} {cci:7.4f} "
                  f"{diff:+8.4f} {p:7.3f} {verdict:>10s}")
            out.append({
                'architecture': arch, 'epoch': ep,
                'n_seeds_win': len(w), 'n_seeds_ctl': len(c),
                'win_mean': round(wm, 4), 'win_ci95': round(wci, 4),
                'ctl_mean': round(cm, 4), 'ctl_ci95': round(cci, 4),
                'delta': round(diff, 4), 'welch_t': round(t, 3), 'p_approx': round(p, 4),
                'verdict': verdict,
            })

    # Headline: does the epoch-1 sign/verdict survive at epoch 10?
    print(f"\n{'=' * 60}\nEpoch-1 vs epoch-10 stability (the key question):")
    by_arch = defaultdict(dict)
    for o in out:
        by_arch[o['architecture']][o['epoch']] = o
    for arch in archs:
        e1, e10 = by_arch[arch].get(1), by_arch[arch].get(10)
        if e1 and e10:
            flip = 'RANKING FLIPPED' if (e1['delta'] > 0) != (e10['delta'] > 0) else 'stable sign'
            print(f"  {arch:12s} ep1 {e1['delta']:+.4f} ({e1['verdict']:>6s})  ->  "
                  f"ep10 {e10['delta']:+.4f} ({e10['verdict']:>6s})   [{flip}]")

    res_path = PROJECT_ROOT / 'out' / 'nngpt' / 'epoch_ablation_results.csv'
    if out:
        with open(res_path, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            wtr.writeheader()
            wtr.writerows(out)
        print(f"\nWrote {res_path}")
    con.close()


if __name__ == '__main__':
    main()

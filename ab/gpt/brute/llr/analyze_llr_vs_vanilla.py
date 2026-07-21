"""
Analyze brute-force layerwise-LR (llr/llr2/llr3) models against their vanilla
architecture baselines.

For every evaluated llr model it recovers the architecture from the metadata CSV
(llr DB names are opaque md5 hashes), finds the vanilla parent at the SAME
(dataset, epoch) — a fair, confounder-free comparison — and computes the accuracy
delta. From that it answers:

  * Did each strategy improve on the vanilla architecture?
  * Which strategy worked best for which (architecture, dataset)?
  * Which strategies win most often / by the largest margin overall?

Outputs (to out/nngpt/):
  llr_vs_vanilla.csv                  one row per llr model + its delta
  best_strategy_per_arch_dataset.csv  winning strategy per (arch, dataset)
  strategy_leaderboard.csv            per-strategy win-rate / mean delta

Usage:
    python -m ab.gpt.brute.llr.analyze_llr_vs_vanilla
"""

import csv
import sqlite3
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DB_PATH = PROJECT_ROOT / 'db' / 'ab.nn.db'
META_CSV = PROJECT_ROOT / 'out' / 'nngpt' / 'llr_evaluated_with_meta.csv'
OUT_DIR = PROJECT_ROOT / 'out' / 'nngpt'


def _db():
    for p in (DB_PATH, Path('/a/mm/db/ab.nn.db')):
        if p.exists():
            return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    raise FileNotFoundError(f"DB not found at {DB_PATH} or /a/mm/db/ab.nn.db")


def load_metadata() -> dict:
    with open(META_CSV, newline='') as f:
        return {row['nn']: row for row in csv.DictReader(f)}


def load_vanilla(cur, arch_names) -> dict:
    """
    Return {(arch, dataset, epoch, transform): best_vanilla_accuracy}.
    Matching on transform removes it as a confounder — the vanilla baseline must
    use the SAME preprocessing as the llr model it is compared against.
    """
    arch_names = [a for a in arch_names if a]
    qm = ",".join("?" * len(arch_names))
    acc = {}
    for nn, ds, ep, tf, a in cur.execute(
        f"SELECT nn, dataset, epoch, transform, MAX(accuracy) FROM stat "
        f"WHERE nn IN ({qm}) GROUP BY nn, dataset, epoch, transform", arch_names):
        acc[(nn, ds, int(ep), tf)] = a
    return acc


def load_llr_best(cur) -> dict:
    """Each llr model at its best epoch: {nn: (dataset, epoch, transform, accuracy)}."""
    best = {}
    for nn, ds, ep, tf, a in cur.execute(
            "SELECT nn, dataset, epoch, transform, accuracy FROM stat WHERE nn LIKE 'llr%'"):
        if nn not in best or a > best[nn][3]:
            best[nn] = (ds, int(ep), tf, a)
    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    arch_names = {m.get('architecture') for m in meta.values()}

    con = _db()
    cur = con.cursor()
    vanilla = load_vanilla(cur, arch_names)
    llr_best = load_llr_best(cur)
    con.close()

    # ---- uniform-control baseline ------------------------------------------
    # Primary baseline = our own llr_uniform control (single group, 1.0x LR),
    # evaluated under identical conditions to the strategies. Keyed on
    # (arch, dataset, epoch, transform). Falls back to the pre-existing DB vanilla
    # only where a uniform control was not evaluated.
    uniform = {}
    for nn, (ds, ep, tf, acc) in llr_best.items():
        m = meta.get(nn)
        if m and 'uniform' in (m.get('strategy', '').lower()):
            arch = m.get('architecture') or ''
            key = (arch, ds, ep, tf)
            if key not in uniform or acc > uniform[key]:
                uniform[key] = acc

    def baseline_for(arch, ds, ep, tf):
        """Uniform control first, then pre-existing DB vanilla. Returns (acc, source)."""
        if (arch, ds, ep, tf) in uniform:
            return uniform[(arch, ds, ep, tf)], 'uniform_control'
        if (arch, ds, ep, tf) in vanilla:
            return vanilla[(arch, ds, ep, tf)], 'db_vanilla'
        return None, ''

    # ---- per-model rows (exclude the uniform controls themselves) ----------
    rows = []
    for nn, (ds, ep, tf, t_acc) in llr_best.items():
        m = meta.get(nn)
        if not m:
            continue
        if 'uniform' in (m.get('strategy', '').lower()):
            continue  # control group is the baseline, not a strategy row
        arch = m.get('architecture') or ''
        v_acc, b_src = baseline_for(arch, ds, ep, tf)
        delta = (t_acc - v_acc) if v_acc is not None else None
        rows.append({
            'nn': nn,
            'architecture': arch,
            'dataset': ds,
            'epoch': ep,
            'transform': tf,
            'strategy': m.get('strategy', ''),
            'strategy_type': m.get('strategy_type', ''),
            'n_groups': m.get('n_groups', ''),
            'multipliers': m.get('multipliers', ''),
            'split_ratios': m.get('split_ratios', ''),
            'description': m.get('description', ''),
            'llr_accuracy': round(t_acc, 6),
            'vanilla_accuracy': round(v_acc, 6) if v_acc is not None else '',
            'baseline_source': b_src,
            'delta': round(delta, 6) if delta is not None else '',
            'pct_improvement': round(100 * delta / v_acc, 3) if v_acc else '',
            'improved': 1 if (delta is not None and delta > 0) else 0,
            'baseline_found': 1 if v_acc is not None else 0,
        })

    fields = ['nn', 'architecture', 'dataset', 'epoch', 'transform', 'strategy', 'strategy_type',
              'n_groups', 'multipliers', 'split_ratios', 'description',
              'llr_accuracy', 'vanilla_accuracy', 'baseline_source', 'delta', 'pct_improvement',
              'improved', 'baseline_found']
    with open(OUT_DIR / 'llr_vs_vanilla.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['architecture'], r['dataset'],
                                                -(r['delta'] if r['delta'] != '' else -1))))

    matched = [r for r in rows if r['baseline_found']]

    # ---- best strategy per (architecture, dataset) -------------------------
    groups = {}
    for r in matched:
        groups.setdefault((r['architecture'], r['dataset']), []).append(r)
    best_rows = []
    for (arch, ds), grp in sorted(groups.items()):
        grp_sorted = sorted(grp, key=lambda r: r['delta'], reverse=True)
        top = grp_sorted[0]
        best_rows.append({
            'architecture': arch,
            'dataset': ds,
            'best_strategy': top['strategy'],
            'strategy_type': top['strategy_type'],
            'best_delta': top['delta'],
            'best_pct_improvement': top['pct_improvement'],
            'llr_accuracy': top['llr_accuracy'],
            'vanilla_accuracy': top['vanilla_accuracy'],
            'strategies_tried': len(grp),
            'strategies_improved': sum(1 for r in grp if r['improved']),
        })
    with open(OUT_DIR / 'best_strategy_per_arch_dataset.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader()
        w.writerows(best_rows)

    # ---- strategy leaderboard ---------------------------------------------
    by_strat = {}
    for r in matched:
        by_strat.setdefault(r['strategy'], []).append(r)
    lb = []
    for strat, grp in by_strat.items():
        deltas = [r['delta'] for r in grp]
        wins = sum(1 for r in grp if r['improved'])
        lb.append({
            'strategy': strat,
            'strategy_type': grp[0]['strategy_type'],
            'n_groups': grp[0]['n_groups'],
            'times_tried': len(grp),
            'times_improved': wins,
            'win_rate_pct': round(100 * wins / len(grp), 1),
            'mean_delta': round(sum(deltas) / len(deltas), 4),
            'median_delta': round(st.median(deltas), 4),
            'best_delta': round(max(deltas), 4),
        })
    lb.sort(key=lambda r: (r['win_rate_pct'], r['mean_delta']), reverse=True)
    with open(OUT_DIR / 'strategy_leaderboard.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lb[0].keys()))
        w.writeheader()
        w.writerows(lb)

    # ---- console summary ---------------------------------------------------
    from collections import Counter as _Counter
    n_imp = sum(r['improved'] for r in matched)
    src = _Counter(r['baseline_source'] for r in matched)
    print(f"strategy models (excl. uniform): {len(rows)}")
    print(f"  matched to a baseline        : {len(matched)}")
    print(f"  improved over baseline       : {n_imp} ({100*n_imp/max(1,len(matched)):.1f}%)")
    print(f"  baseline source              : {dict(src)}")
    print(f"\nWrote:")
    print(f"  {OUT_DIR / 'llr_vs_vanilla.csv'}")
    print(f"  {OUT_DIR / 'best_strategy_per_arch_dataset.csv'}")
    print(f"  {OUT_DIR / 'strategy_leaderboard.csv'}")
    print(f"\nTop 10 strategies by win-rate (min plausible support):")
    for r in [x for x in lb if x['times_tried'] >= 3][:10]:
        print(f"  {r['strategy']:22s} tried={r['times_tried']:3d} "
              f"win%={r['win_rate_pct']:5.1f} meanΔ={r['mean_delta']:+.4f} "
              f"bestΔ={r['best_delta']:+.4f}")


if __name__ == '__main__':
    main()

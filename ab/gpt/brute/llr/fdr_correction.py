"""
Multiple-comparisons correction for the brute-force layerwise-LR (llr/llr2/llr3)
sweep in llr_vs_vanilla.csv.

The sweep ran ~39 strategies x ~26 architectures x 3 datasets (~2,800
evaluations), each a single trial (no seed replication). Testing "does
architecture X respond to layerwise-LR" or "does strategy Y win" that many
times without correction means some fraction of apparent wins are expected by
chance alone, even under a null of no true effect. This script quantifies that:

  1. Per architecture: is its win rate (fraction of tried strategies that beat
     the vanilla/uniform-control baseline) higher than the REST of the
     population's win rate? One-sided Fisher's-exact / hypergeometric
     enrichment test (H0: this architecture's trials are an unbiased random
     draw from the pooled pool of trials, so its win count is no higher than
     drawing that many trials at random would give; H1: enriched for wins),
     then Benjamini-Hochberg FDR correction across all architectures tested.
     Deliberately hypergeometric rather than "binomial against the pooled
     rate" — the latter would test each architecture against a null that
     includes its own trials, biasing the test toward the null (conservative
     in a way that's hard to reason about) for exactly the architectures
     that contribute the most trials.
  2. Same test/correction, per strategy across all architectures it was tried on.

No scipy/statsmodels dependency (neither is installed on this host) — the
exact hypergeometric tail and BH step-up procedure are both simple enough to
implement directly with math.comb.

Outputs (to out/nngpt/):
  architecture_fdr.csv   one row per architecture: n, wins, win_rate, p, q, significant_q05
  strategy_fdr.csv       same, per strategy

Usage:
    python -m ab.gpt.brute.llr.fdr_correction
"""

import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
IN_CSV = PROJECT_ROOT / 'out' / 'nngpt' / 'llr_vs_vanilla.csv'
OUT_DIR = PROJECT_ROOT / 'out' / 'nngpt'

MIN_TRIALS = 5  # below this, a binomial test has essentially no power; report but flag as underpowered
ALPHA = 0.05


def load_rows() -> list[dict]:
    with open(IN_CSV, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('baseline_found') == '1']
    for r in rows:
        r['improved'] = int(r['improved'])
    return rows


def hypergeom_sf_inclusive(k: int, N: int, K: int, n: int) -> float:
    """
    P(X >= k) for X ~ Hypergeometric(population N, population successes K, draws n) —
    i.e. one-sided enrichment p-value for observing >= k wins among n trials drawn
    without replacement from a pool of N trials containing K total wins.
    Computed exactly via math.comb (all values here are small enough, <~3000).
    """
    lo = max(0, n - (N - K))
    hi = min(n, K)
    if k > hi:
        return 0.0
    if k <= lo:
        return 1.0
    denom = math.comb(N, n)
    return sum(
        math.comb(K, i) * math.comb(N - K, n - i)
        for i in range(k, hi + 1)
    ) / denom


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up q-values. Returns q in the same order as input."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q_sorted = [0.0] * m
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end  # 1-indexed rank in ascending order
        q = min(prev, p_values[idx] * m / rank)
        q_sorted[idx] = q
        prev = q
    return q_sorted


def group_and_test(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)

    N = len(rows)
    K = sum(r['improved'] for r in rows)

    out = []
    for name, grp in groups.items():
        n = len(grp)
        wins = sum(r['improved'] for r in grp)
        win_rate = wins / n
        p_val = hypergeom_sf_inclusive(wins, N, K, n)
        out.append({key: name, 'n': n, 'wins': wins, 'win_rate_pct': round(100 * win_rate, 1),
                    'p_value': p_val, 'underpowered': n < MIN_TRIALS})

    q_vals = bh_fdr([r['p_value'] for r in out])
    for r, q in zip(out, q_vals):
        r['q_value'] = q
        r['significant_uncorrected_p05'] = r['p_value'] < ALPHA
        r['significant_fdr_q05'] = q < ALPHA
        r['p_value'] = round(r['p_value'], 6)
        r['q_value'] = round(r['q_value'], 6)
    out.sort(key=lambda r: r['p_value'])
    return out


def write_csv(path: Path, rows: list[dict], key: str):
    fields = [key, 'n', 'wins', 'win_rate_pct', 'p_value', 'q_value',
              'significant_uncorrected_p05', 'significant_fdr_q05', 'underpowered']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(label: str, rows: list[dict]):
    n_tested = len(rows)
    n_uncorrected = sum(r['significant_uncorrected_p05'] for r in rows)
    n_fdr = sum(r['significant_fdr_q05'] for r in rows)
    n_fdr_powered = sum(r['significant_fdr_q05'] and not r['underpowered'] for r in rows)
    print(f"\n{label}: {n_tested} tested")
    print(f"  significant at uncorrected p<0.05 : {n_uncorrected}")
    print(f"  significant at FDR q<0.05         : {n_fdr}  ({n_fdr_powered} with n>={MIN_TRIALS})")
    print(f"  survivors (q<0.05, n>={MIN_TRIALS}):")
    for r in rows:
        if r['significant_fdr_q05'] and not r['underpowered']:
            print(f"    {r[list(r.keys())[0]]:22s} n={r['n']:3d} wins={r['wins']:3d} "
                  f"win%={r['win_rate_pct']:5.1f} p={r['p_value']:.4g} q={r['q_value']:.4g}")


def main():
    rows = load_rows()
    pooled_wins = sum(r['improved'] for r in rows)
    pooled_rate = pooled_wins / len(rows)
    print(f"Pooled: {len(rows)} matched trials, {pooled_wins} wins, "
          f"overall win rate {100 * pooled_rate:.1f}%. Each architecture/strategy "
          f"is tested against the REST of the pool (hypergeometric enrichment), "
          f"not against this overall rate directly.")

    arch_rows = group_and_test(rows, 'architecture')
    write_csv(OUT_DIR / 'architecture_fdr.csv', arch_rows, 'architecture')
    summarize('Per-architecture', arch_rows)

    strat_rows = group_and_test(rows, 'strategy')
    write_csv(OUT_DIR / 'strategy_fdr.csv', strat_rows, 'strategy')
    summarize('Per-strategy', strat_rows)

    print(f"\nWrote:\n  {OUT_DIR / 'architecture_fdr.csv'}\n  {OUT_DIR / 'strategy_fdr.csv'}")


if __name__ == '__main__':
    main()

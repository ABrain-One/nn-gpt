"""Diversity of a generated cohort, and how much of it is copied from the fine-tune data.

WHY THIS EXISTS. The deploy fine-tune optimises a joint PSNR+latency objective, and the cheapest
way for a model to satisfy that objective is to stop exploring and reproduce architectures it was
trained on -- the `DnSeed2*` arms already sit at 100% NPU residency, so imitating them scores well
while contributing nothing. Residency alone cannot detect that: a collapsed cohort and a genuinely
improved one look identical on latency. Diversity has to be measured alongside it or the experiment
cannot distinguish success from mode collapse.

THREE MEASURES, and the third is the one that matters.

  1. SELF-DIVERSITY -- greedy retention at a Jaccard ceiling, the same filter used to build the
     SFT set. "If I deduplicated this cohort, how much would survive?" Low retention means the
     cohort repeats itself.

  2. MEAN PAIRWISE JACCARD -- how similar a random pair of cohort members is. Reported because
     retention is threshold-sensitive and this is not.

  3. NOVELTY AGAINST THE TRAINING SET -- for each generated model, the maximum Jaccard to any SFT
     target. This is the mode-collapse detector. A fine-tune that teaches the model to emit its
     training data drives this UP, and neither of the first two measures would notice: a cohort of
     faithful copies of 445 diverse targets is itself diverse. Reported as the fraction at or above
     0.90, i.e. effectively reproductions.

MinHash over token 5-shingles, 128 permutations; comments and blank lines stripped so reformatting
does not read as novelty. Validated against exact Jaccard at mean absolute error 0.017. This is the
same instrument arXiv:2601.02997 uses for its novelty filter, so the numbers are comparable.

  python cohort_diversity.py --prefix LLM_Pg            # a corpus producer
  python cohort_diversity.py --src DIR --vs-sft DIR     # a fresh cohort of .py files
"""
import argparse, glob, json, os, random, sqlite3, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run3_sft_data import _shingles, _minhash, _jac


def sigs_from_codes(codes):
    return {n: _minhash(_shingles(c)) for n, c in codes.items() if c and c.strip()}


def greedy_retain(sig_items, tau):
    kept = []
    for _, s in sig_items:
        if all(_jac(s, k) <= tau for k in kept):
            kept.append(s)
    return len(kept)


def mean_pairwise(sigs, n=4000, seed=11):
    rng = random.Random(seed)
    names = list(sigs)
    if len(names) < 2: return float('nan')
    vals = []
    for _ in range(n):
        a, b = rng.choice(names), rng.choice(names)
        if a != b: vals.append(_jac(sigs[a], sigs[b]))
    return st.fmean(vals) if vals else float('nan')


def report(label, sigs, ref_sigs=None, taus=(0.95, 0.90, 0.85)):
    items = sorted(sigs.items())
    out = {'label': label, 'n': len(items), 'mean_pairwise_jaccard': mean_pairwise(sigs)}
    print('\n=== %s  (n=%d) ===' % (label, len(items)))
    print('  mean pairwise Jaccard        : %.3f' % out['mean_pairwise_jaccard'])
    for t in taus:
        k = greedy_retain(items, t)
        out['retain_%.2f' % t] = 100.0 * k / len(items)
        print('  distinct at Jaccard <= %.2f   : %4d / %d  (%.1f%%)' % (t, k, len(items), 100.0*k/len(items)))
    if ref_sigs:
        rs = list(ref_sigs.values())
        nov = [max(_jac(s, r) for r in rs) for _, s in items]
        nov.sort()
        out['novelty_vs_ref'] = {'median': st.median(nov), 'p10': nov[int(.1*len(nov))],
                                 'p90': nov[int(.9*len(nov))],
                                 'pct_ge_090': 100.0*sum(1 for x in nov if x >= 0.90)/len(nov)}
        print('  --- novelty against reference set (n=%d) ---' % len(rs))
        print('  max-Jaccard to nearest ref   : p10 %.3f  median %.3f  p90 %.3f'
              % (out['novelty_vs_ref']['p10'], out['novelty_vs_ref']['median'], out['novelty_vs_ref']['p90']))
        print('  effectively reproductions    : %.1f%% at >= 0.90'
              % out['novelty_vs_ref']['pct_ge_090'])
    return out


def codes_from_db(db, prefix, limit=0):
    c = sqlite3.connect('file:%s?mode=ro' % db, uri=True)
    q = "select name, code from nn where name like ?"
    rows = c.execute(q, (prefix + '%',)).fetchall()
    if limit: rows = rows[:limit]
    return {n: k for n, k in rows if k}


def codes_from_dir(d):
    return {os.path.basename(p)[:-3]: open(p).read() for p in sorted(glob.glob(os.path.join(d, '*.py')))}


def codes_from_sft(d):
    """The assistant turn of each SFT example is the target architecture."""
    out = {}
    for i, line in enumerate(open(os.path.join(d, 'train.jsonl'))):
        if not line.strip(): continue
        msgs = json.loads(line).get('messages', [])
        a = next((m['content'] for m in msgs if m['role'] == 'assistant'), None)
        if a: out['sft%04d' % i] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db')
    ap.add_argument('--prefix', action='append', default=[])
    ap.add_argument('--src', action='append', default=[])
    ap.add_argument('--vs-sft', default=None, help='SFT dir whose targets are the novelty reference')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='/shared/ssd/home/b-b-ahmed/thesis/sweep_out/sft_deploy/diversity.json')
    a = ap.parse_args()

    ref = sigs_from_codes(codes_from_sft(a.vs_sft)) if a.vs_sft else None
    if ref: print('novelty reference: %d SFT targets from %s' % (len(ref), a.vs_sft))

    res = []
    for p in a.prefix:
        res.append(report(p, sigs_from_codes(codes_from_db(a.db, p, a.limit)), ref))
    for d in a.src:
        res.append(report(os.path.basename(d.rstrip('/')), sigs_from_codes(codes_from_dir(d)), ref))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('\nwrote ' + a.out)


if __name__ == '__main__':
    main()

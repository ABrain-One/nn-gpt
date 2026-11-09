"""Calibrate a diversity threshold FOR LLM OUTPUT, instead of inheriting the GA's.

THE PROBLEM. Two thresholds have been used to call models "distinct":

  exact WL hash    certifies the cold arm at 98.6% unique. It is a literal-clone test, so
                   400 minor rewrites of one U-Net all pass. Useless as a diversity claim
                   for a generator that paraphrases.
  cosine >= 0.95   calibrated on GA MUTANTS: trivial variation sat >= 0.985, distinct
                   architectures <= 0.921, so 0.95 split them with margin.

The second is the same category error as the first, one level up: a threshold calibrated on
one generator's similarity distribution, applied to a different generator. Mutants of one
parent occupy a tight, high-similarity band by construction; LLM samples from independent
prompts do not. Neither number is defensible for LLM output until it is re-derived on LLM
output.

THE CALIBRATION. Same two-sided design that produced 0.95, with LLM-appropriate anchors:

  POSITIVE (must be called duplicate)  pairs inside an exact-WL-hash collision group. These
      are graph-isomorphic with identical op labels -- genuinely the same architecture,
      established independently of the embedding. This is the honest upper anchor.
  NEGATIVE (must be called distinct)   pairs of DIFFERENT canonical hand-written families.
      Those 18 were designed as distinct architectures and are the corpus the whole project
      treats as its diversity reference.

A threshold is defensible iff it separates these two sets. If they overlap, no threshold on
this embedding is defensible for LLM output and that is the honest finding -- report the
curve, not a number.

ORDER-DEPENDENCE. DiversityGate is GREEDY: it accepts whatever arrives first and rejects
later neighbours, so "N distinct" depends on candidate ordering and is a lower bound, not a
property of the set. This also reports single-linkage CLUSTER COUNT at each threshold, which
is order-independent and is what should actually be quoted.

  python calibrate_diversity.py --run-dirs .../cold/run .../seeded/run --nn-dir .../ab/nn/nn
"""
import argparse
import glob
import itertools
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.921, 0.93, 0.95, 0.97, 0.985, 0.99]


def trace_all(items, log=print):
    """[(name, hash, embed)] for [(name, source)]."""
    import arch_embed
    out = []
    for i, (name, src) in enumerate(items, 1):
        try:
            h, e = arch_embed.both_source(src)
        except Exception:
            h, e = None, None
        out.append((name, h, e))
        if i % 50 == 0 or i == len(items):
            log(f"  traced {i}/{len(items)} (ok={sum(1 for _,hh,_ in out if hh is not None)})")
    return out


def pct(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return "n/a"
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return ("min %.4f  p5 %.4f  p25 %.4f  med %.4f  p75 %.4f  p95 %.4f  max %.4f" % tuple(q))


def single_linkage_clusters(names, E, thr):
    """Order-independent: union-find over all pairs with cosine >= thr."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    S = E @ E.T
    idx = {n: i for i, n in enumerate(names)}
    for a, b in itertools.combinations(names, 2):
        if S[idx[a], idx[b]] >= thr:
            union(a, b)
    return len({find(n) for n in names})


def greedy_count(names, E, thr, order):
    idx = {n: i for i, n in enumerate(names)}
    kept = []
    for n in order:
        v = E[idx[n]]
        if all(float(np.dot(v, E[idx[k]])) < thr for k in kept):
            kept.append(n)
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--nn-dir", default="/a/mm/nn-dataset/ab/nn/nn")
    ap.add_argument("--families", nargs="*", default=None,
                    help="canonical family .py stems used as the DISTINCT anchor")
    ap.add_argument("--out", default="/a/mm/sweep_out/llm_div")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- LLM candidates ----
    items = []
    for r in a.run_dirs:
        for p in sorted(glob.glob(os.path.join(r, "cand", "*", "candidate.py"))):
            items.append((os.path.basename(os.path.dirname(p)), open(p).read()))
    items.sort()
    print(f"=== tracing {len(items)} LLM candidates ===", flush=True)
    llm = trace_all(items)

    # ---- canonical families: the DISTINCT anchor ----
    fam_names = a.families or [
        "DenoiseSubPixel", "DenoiseDilatedFlat", "DenoiseLPPool", "DenoiseELUCtx",
        "DenoiseMBConv", "DenoisePoolDeconv", "DenoiseLinearMix", "DenoiseSelfAttn",
        "DenoiseThunder", "DenoiseMFDNet", "DenoiseSplitterNet", "DenoiseChanShuffle",
        "DenoiseLapPyr", "DenoiseParallelMR", "DenoiseReflectPad", "DenoiseGroupNorm",
    ]
    fitems = []
    for f in fam_names:
        p = os.path.join(a.nn_dir, f + ".py")
        if os.path.exists(p):
            fitems.append((f, open(p).read()))
    print(f"\n=== tracing {len(fitems)} canonical families (distinct anchor) ===", flush=True)
    fam = trace_all(fitems)

    # ---- ground-truth pair sets ----
    ll = [(n, h, e) for n, h, e in llm if h is not None and e is not None]
    by_hash = defaultdict(list)
    for n, h, e in ll:
        by_hash[h].append((n, e))
    pos = []
    for h, group in by_hash.items():
        if len(group) > 1:
            for (n1, e1), (n2, e2) in itertools.combinations(group, 2):
                pos.append(float(np.dot(e1, e2)))
    ff = [(n, h, e) for n, h, e in fam if e is not None]
    neg = [float(np.dot(e1, e2))
           for (n1, _, e1), (n2, _, e2) in itertools.combinations(ff, 2)]

    print("\n" + "=" * 78)
    print("GROUND TRUTH COSINE DISTRIBUTIONS")
    print("=" * 78)
    print(f"POSITIVE  same exact WL graph (must be DUPLICATE)   n={len(pos)}")
    print(f"          {pct(pos)}")
    print(f"NEGATIVE  different canonical family (must be DISTINCT) n={len(neg)}")
    print(f"          {pct(neg)}")
    if pos and neg:
        lo_pos, hi_neg = min(pos), max(neg)
        print(f"\n  lowest duplicate  {lo_pos:.4f}")
        print(f"  highest distinct  {hi_neg:.4f}")
        if lo_pos > hi_neg:
            print(f"  SEPARABLE: any threshold in ({hi_neg:.4f}, {lo_pos:.4f}] works; "
                  f"midpoint {(lo_pos+hi_neg)/2:.4f}")
        else:
            print(f"  OVERLAP of {lo_pos - hi_neg:+.4f} -- NO threshold on this embedding "
                  f"separates them. Report the curve, not a number.")

    # ---- what each threshold does to the LLM corpus ----
    names = [n for n, _, _ in ll]
    E = np.stack([e for _, _, e in ll])
    print("\n" + "=" * 78)
    print("LLM CORPUS vs THRESHOLD   (n=%d traceable)" % len(names))
    print("=" * 78)
    print(f"{'thr':>6} {'clusters':>9} {'greedy med':>11} {'greedy min-max':>16} "
          f"{'FP':>5} {'FN':>5}")
    rng = random.Random(0)
    rows = []
    for t in THRESHOLDS:
        cl = single_linkage_clusters(names, E, t)
        gs = []
        for _ in range(20):
            o = names[:]
            rng.shuffle(o)
            gs.append(greedy_count(names, E, t, o))
        fp = sum(1 for v in neg if v >= t)      # distinct pair wrongly merged
        fn = sum(1 for v in pos if v < t)       # duplicate pair wrongly kept
        rows.append({"thr": t, "clusters": cl, "greedy_med": int(np.median(gs)),
                     "greedy_min": min(gs), "greedy_max": max(gs),
                     "false_merge": fp, "false_split": fn})
        print(f"{t:>6.3f} {cl:>9d} {int(np.median(gs)):>11d} "
              f"{str(min(gs))+'-'+str(max(gs)):>16} {fp:>5d} {fn:>5d}")
    print("\n  clusters   = single-linkage components, ORDER-INDEPENDENT -> quote this")
    print("  greedy     = DiversityGate's own count over 20 random orders; the spread is")
    print("               how much 'N distinct' moves purely with arrival order")
    print("  FP         = canonical family pairs wrongly called duplicate at this threshold")
    print("  FN         = exact-graph-clone pairs wrongly called distinct")

    json.dump({"positive": pos, "negative": neg, "rows": rows,
               "n_traceable": len(names)},
              open(os.path.join(a.out, "calibration.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, "embeds.npz"),
                        names=np.array(names), E=E)
    print(f"\nwritten {a.out}/calibration.json and embeds.npz")


if __name__ == "__main__":
    raise SystemExit(main())

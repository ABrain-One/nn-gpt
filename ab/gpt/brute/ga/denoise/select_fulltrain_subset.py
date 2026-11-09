"""Pick the 50-EPOCH BENCHMARK SUBSET from the 5-epoch corpus.

WHY A SUBSET AND NOT THE POOL. 2,824 of 2,958 models hold only epoch<=5 rows, and the
5-epoch ranking recovers 3 of the true top-10 at 50 epochs (Spearman 0.68, n=56). Every
downstream claim -- cross-dataset transfer, skip ablation, any leaderboard -- inherits that
unreliability. 50 epochs on the full pool is ~2,000 GPU-hours; on a stratified 300 it is
~225, which is affordable and enough to (a) measure the 5ep->50ep rank correlation on a
large sample and (b) publish a converged leaderboard.

STRATIFICATION IS NOT UNIFORM, ON PURPOSE. The corpus is pathologically concentrated:
70% of models fall inside [33,34) dB, one decibel wide, against a measured 0.14 dB
run-to-run noise floor. A uniform sample would spend most of its GPU inside a band the
metric cannot resolve, and would report a low rank correlation that is an artifact of
sampling rather than a property of the proxy. So:

  TOP   (~40%) -- where promotion decisions are actually made; a proxy that misranks here
                  is useless even if it is globally correlated.
  MID   (~40%) -- spread across the bulk, to estimate correlation over the realized range.
  LOW   (~20%) -- below the bulk, anchors the low end. Kept deliberately small: separating
                  broken models from working ones is trivial and inflates rho for free.

DEDUPED BY arch_uid. Isomorphic duplicates (136 corpus-wide) would consume GPU to
re-measure the same architecture; the best-scoring representative of each uid is kept.

NO REPLICATES IN THIS PHASE. Re-running a model that already holds an epoch_max=50 row
would be SKIPPED by remaining_trials() -- the config key is (task,dataset,metric,nn,
epoch_max) and the trial already exists. That is the trap the earlier 9F control pod fell
into (see sweep_canonical.sh). Measuring 50-epoch run-to-run noise therefore needs its own
experiment with the prior rows removed from the pod-local DB; it is NOT free here.

  python select_fulltrain_subset.py --n 300 --out fulltrain_subset.json
"""
import argparse
import collections
import json
import random
import sqlite3

SCALE = 48.0            # stat.accuracy * 48 == PSNR dB (verified against run-3 logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    best = {}
    for nn, acc in c.execute("select nn,accuracy from stat where epoch=5"):
        best[nn] = max(best.get(nn, 0.0), acc)
    uid = dict(c.execute("select name,arch_uid from nn"))
    done50 = {r[0] for r in c.execute("select distinct nn from stat where epoch=50")}

    # one representative per architecture, best 5-ep score wins
    by_uid = {}
    for n, s in best.items():
        if n in done50:
            continue                      # already converged; it is an anchor, not a task
        u = uid.get(n) or n
        if u not in by_uid or s > by_uid[u][1]:
            by_uid[u] = (n, s)
    pool = sorted(by_uid.values(), key=lambda t: -t[1])
    print(f"pool after uid-dedupe and excluding {len(done50)} already-converged: {len(pool)}")

    db = [(n, s * SCALE) for n, s in pool]
    top = [t for t in db if t[1] >= 34.24]                    # >= p90 of the corpus
    mid = [t for t in db if 32.50 <= t[1] < 34.24]
    low = [t for t in db if t[1] < 32.50]
    print(f"strata available: top {len(top)}  mid {len(mid)}  low {len(low)}")

    want_top = int(round(0.40 * a.n))
    want_mid = int(round(0.40 * a.n))
    want_low = a.n - want_top - want_mid

    def spread(items, k):
        """Take k items spread evenly through a score-sorted list (not the head)."""
        if k >= len(items):
            return list(items)
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]

    pick = []
    pick += spread(top, min(want_top, len(top)))
    pick += spread(mid, min(want_mid, len(mid)))
    pick += spread(low, min(want_low, len(low)))
    # backfill from the top band if a stratum was short
    if len(pick) < a.n:
        chosen = {n for n, _ in pick}
        for n, s in db:
            if n not in chosen:
                pick.append((n, s))
                chosen.add(n)
            if len(pick) >= a.n:
                break

    rng.shuffle(pick)                     # decorrelate score from shard index
    names = [n for n, _ in pick]
    scores = [s for _, s in pick]
    fam = collections.Counter(n.split("-")[0].rstrip("0123456789") for n in names)
    print(f"SELECTED {len(names)} models")
    print(f"  5-ep dB: min {min(scores):.2f}  median {sorted(scores)[len(scores)//2]:.2f}  "
          f"max {max(scores):.2f}")
    print(f"  families: {fam.most_common(8)}")
    print(f"  distinct arch_uid: {len({uid.get(n) or n for n in names})}")
    json.dump({"models": names, "ep5_dB": {n: s for n, s in pick}},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

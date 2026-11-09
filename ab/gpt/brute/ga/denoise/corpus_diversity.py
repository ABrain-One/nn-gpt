"""How many DISTINCT architectures does the corpus actually hold? Two definitions, both.

  EXACT (WL graph hash)   -- two models are the same iff their computation graphs are
                             isomorphic under Weisfeiler-Lehman with behavioural labels.
                             Already stored per model in nn.arch_uid, so it costs a SELECT.
  GRADED (continuous-WL cosine) -- two models are "the same" iff cosine >= --cos-max. This
                             is the operating point the diversity gate rejects on, and it
                             is strictly coarser: every exact duplicate is also a graded
                             one, never the reverse.

WHY BOTH NUMBERS MATTER AND WHY THEY DISAGREE. Exact-hash uniqueness certifies as "distinct"
any pair that differs by one channel width or one activation swap -- a real difference in the
graph, not a real difference in architecture. On this corpus that gap has been large enough
to change decisions: exact-hash once certified 98.6% of the cold arm as distinct while the
graded gate found 39% of traceable candidates were near-duplicates. Quoting the exact count
as "distinct architectures" overstates the corpus; quoting only the graded count understates
it, because 0.95 also merges genuinely different small models. Report the pair.

GRADED COUNT IS CLUSTER COUNT, NOT A PAIR COUNT, and the two are not interchangeable.
Near-duplication is not transitive -- A~B and B~C does not give A~C -- so "how many pairs
exceed 0.95" and "how many groups are there" answer different questions. Greedy
single-representative clustering in descending-quality order is used here: each model joins
the first representative it is near, otherwise founds its own. That is exactly what
DiversityGate.check does when it admits or rejects a candidate against its store, so the
number reported is the number the gate would actually produce -- not a prettier one from a
better clustering algorithm the gate does not run.

Untraceable models are counted and reported separately, never folded into either statistic.

  python corpus_diversity.py --db /a/mm/nn-dataset/db/ab.nn.db --cos-max 0.95
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "img-denoising"


def arm_of(name):
    if name.startswith("LLM_Pg") or name.startswith("g") and "_seeded_" in name:
        return "LLM"
    if name.startswith("LLM_"):
        return "LLM"
    if name.startswith(("DenoiseGA-", "DnSeedGA-", "DenoiseSkip-", "DnSkip-",
                        "DnNewFam-", "DnSearch-")):
        return "GA"
    return "hand-written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/corpus_diversity.json")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, n.arch_uid FROM nn n WHERE n.name IN "
        "(SELECT DISTINCT nn FROM stat WHERE task=?)", (a.task,)).fetchall()
    # Best measured dB per model -- clustering order, so a cluster is represented by its
    # best member rather than by whichever row happened to be visited first.
    best = {}
    for nn, acc in con.execute(
            "SELECT nn, MAX(accuracy) FROM stat WHERE task=? AND accuracy IS NOT NULL "
            "GROUP BY nn", (a.task,)):
        best[nn] = (acc or 0) * 48.0
    con.close()

    print(f"=== corpus diversity | {len(rows)} models with a {a.task} result ===\n")

    # ---- EXACT, straight from the stored column
    uids = [u for _n, _c, u in rows if u]
    n_null = len(rows) - len(uids)
    exact_distinct = len(set(uids))
    print(f"EXACT (WL graph hash)")
    print(f"  models                     {len(rows)}")
    print(f"  traceable (arch_uid set)   {len(uids)}")
    print(f"  untraceable                {n_null}")
    print(f"  DISTINCT architectures     {exact_distinct}")
    print(f"  isomorphic duplicates      {len(uids) - exact_distinct}")

    # ---- GRADED
    import numpy as np
    from arch_embed import embed_source
    print(f"\nGRADED (continuous-WL cosine >= {a.cos_max}) -- embedding {len(rows)} models")
    vecs, names, failed, t0 = [], [], [], time.time()
    for i, (name, code, _u) in enumerate(rows, 1):
        try:
            v = np.asarray(embed_source(code), dtype=np.float64).ravel()
            n = np.linalg.norm(v)
            if n == 0 or not np.isfinite(n):
                raise ValueError("degenerate embedding")
            vecs.append(v / n)
            names.append(name)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"[:120]))
        if i % 100 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] embedded={len(vecs)} failed={len(failed)} "
                  f"{time.time()-t0:.0f}s", flush=True)

    if not vecs:
        print("no embeddings; cannot report a graded count")
        return 1

    # Equal-length check: _combine can emit different widths if tracing degrades. Silently
    # truncating would compute cosines between incomparable vectors.
    widths = collections.Counter(v.shape[0] for v in vecs)
    if len(widths) > 1:
        print(f"  WARNING: embedding widths differ {dict(widths)}; keeping the majority")
        w = widths.most_common(1)[0][0]
        keep = [i for i, v in enumerate(vecs) if v.shape[0] == w]
        vecs = [vecs[i] for i in keep]
        names = [names[i] for i in keep]

    order = sorted(range(len(names)), key=lambda i: -best.get(names[i], 0.0))
    M = np.stack([vecs[i] for i in order])
    ordered = [names[i] for i in order]

    def cluster(thresh):
        reps, members = [], []
        for i in range(M.shape[0]):
            if reps:
                sims = M[reps] @ M[i]
                j = int(np.argmax(sims))
                if sims[j] >= thresh:
                    members[j].append(ordered[i])
                    continue
            reps.append(i)
            members.append([ordered[i]])
        return reps, members

    # THE COUNT IS A CURVE, NOT A NUMBER, and reporting one point hides how much the choice
    # of threshold decides the answer. diversity_gate.py's calibration fixes two anchors --
    # "trivial variation (width + same-family mutants) sits >=0.985, distinct archs <=0.921"
    # -- and 0.95 was chosen as a midpoint between them. Everything in the 0.921-0.985 band
    # is UNRESOLVED by that calibration, and clustering at 0.95 forcibly merges all of it.
    # Since the cluster distribution is heavily skewed (largest cluster 172 models), where
    # the cut falls moves the total by hundreds. Sweeping costs nothing: the embedding pass
    # is the expensive part and it is already done.
    sweep = [0.90, 0.921, 0.93, 0.95, 0.97, 0.985, 0.99, 0.995]
    if a.cos_max not in sweep:
        sweep = sorted(set(sweep + [a.cos_max]))
    print(f"\n=== distinct-architecture count vs threshold ===")
    print(f"  {'cosine':>8}  {'clusters':>9}  {'largest':>8}  note")
    sweep_out = {}
    for t in sweep:
        r, m = cluster(t)
        sweep_out[t] = len(r)
        note = ""
        if abs(t - 0.921) < 1e-9:
            note = "calibration: distinct archs sit at or BELOW this"
        elif abs(t - 0.985) < 1e-9:
            note = "calibration: trivial variants sit at or ABOVE this"
        elif t == a.cos_max:
            note = "<- gate's operating point"
        print(f"  {t:8.3f}  {len(r):9d}  {max(len(x) for x in m):8d}  {note}")
    lo, hi = sweep_out[0.921], sweep_out[0.985]
    print(f"\n  => defensible range: {lo} to {hi} distinct architectures")
    print(f"     (below 0.921 the calibration says you are merging genuinely distinct "
          f"models;\n      above 0.985 you are counting trivial width variants as new "
          f"architectures)")

    reps, members = cluster(a.cos_max)
    graded_distinct = len(reps)
    print(f"\nGRADED RESULT")
    print(f"  embedded successfully      {M.shape[0]}")
    print(f"  embedding failed           {len(failed)}")
    print(f"  DISTINCT clusters          {graded_distinct}")
    print(f"  near-duplicates absorbed   {M.shape[0] - graded_distinct}")

    print(f"\n=== the two counts side by side ===")
    print(f"  rows in nn (denoise)                     {len(rows)}")
    print(f"  distinct by EXACT WL hash                {exact_distinct}")
    print(f"  distinct by COSINE >= {a.cos_max:<4}               {graded_distinct}")
    print(f"  => exact overstates by                   "
          f"{exact_distinct - graded_distinct} architectures")

    big = sorted(range(len(members)), key=lambda k: -len(members[k]))[:5]
    if members and len(members[big[0]]) > 1:
        print(f"\n=== 5 largest near-duplicate clusters ===")
        for k in big:
            if len(members[k]) < 2:
                continue
            print(f"  {len(members[k])} models around {members[k][0][:46]}")

    by_arm = collections.Counter(arm_of(n) for n in ordered)
    rep_arm = collections.Counter(arm_of(ordered[i]) for i in reps)
    print(f"\n=== per arm (models -> distinct clusters) ===")
    for arm in sorted(by_arm):
        print(f"  {arm:<14} {by_arm[arm]:5d} -> {rep_arm.get(arm,0):5d}")

    if failed:
        print(f"\n=== embedding failures ({len(failed)}) ===")
        for k, v in collections.Counter(e.split(":")[0] for _n, e in failed).most_common(6):
            print(f"  {v:4d}  {k}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"models": len(rows), "exact_distinct": exact_distinct,
                   "untraceable": n_null, "cos_max": a.cos_max,
                   "graded_distinct": graded_distinct,
                   "clusters": {members[k][0]: members[k] for k in range(len(members))
                                if len(members[k]) > 1},
                   "embed_failed": dict(failed)}, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

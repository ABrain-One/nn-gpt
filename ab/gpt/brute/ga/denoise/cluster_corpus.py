"""Merge embed_corpus.py shards and report distinct-architecture counts. CPU, no torch.

Two definitions, reported together because neither is "the" answer:
  EXACT   WL graph hash -- two models are the same iff their graphs are isomorphic.
  GRADED  continuous-WL cosine >= threshold -- the operating point DiversityGate rejects on,
          strictly coarser: every exact duplicate is a graded one, never the reverse.

THE COUNT IS A CURVE. diversity_gate.py's calibration fixes two anchors -- trivial variation
(width + same-family mutants) sits >=0.985, genuinely distinct architectures <=0.921 -- and
0.95 was chosen as a midpoint. The 0.921-0.985 band is UNRESOLVED by that calibration, and a
single threshold silently merges all of it. On the denoise corpus that band moved the answer
from 396 to 118 distinct architectures, so quoting one number without the curve overstates
the precision of the claim by a factor of three.

UNTRACEABLE MODELS ARE REPORTED, NEVER IMPUTED. On classification roughly a fifth of models
cannot be traced by torch.fx at all (data-dependent control flow in torchvision backbones).
They have no architectural identity, and any published "distinct architectures" figure has to
say whether they are excluded, counted as singletons, or fall back to the lexical hash. This
prints all three so the choice is explicit rather than accidental.

  python3 cluster_corpus.py --in DIR
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.90, 0.921, 0.93, 0.95, 0.97, 0.985, 0.99, 0.995])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np

    names, hashes, mats = [], [], []
    build_fail, trace_fail, elapsed = {}, {}, []
    for p in sorted(glob.glob(os.path.join(a.indir, "shard*.npz"))):
        z = np.load(p, allow_pickle=True)
        n = list(z["names"])
        names += n
        hashes += list(z["hashes"])
        v = z["vecs"]
        if v.size:
            mats.append(v)
        elif n:
            print(f"  WARNING {os.path.basename(p)}: {len(n)} names but no vectors "
                  f"(ragged widths) -- excluded from the graded count")
    for p in sorted(glob.glob(os.path.join(a.indir, "fail*.json"))):
        j = json.load(open(p))
        build_fail.update(j.get("build", {}))
        trace_fail.update(j.get("trace", {}))
        elapsed.append(j.get("elapsed_s", 0))

    total_attempted = len(names) + len(build_fail) + len(trace_fail)
    print(f"=== corpus diversity | {total_attempted} models attempted ===")
    print(f"  embedded successfully   {len(names)}")
    print(f"  build failures          {len(build_fail)}   (shape/contract)")
    print(f"  trace failures          {len(trace_fail)}   (torch.fx cannot trace)")
    if elapsed:
        print(f"  shard wall time         max {max(elapsed)/60:.1f} min, "
              f"sum {sum(elapsed)/60:.1f} min across {len(elapsed)} shards")

    if not names:
        print("\nnothing embedded")
        return 1

    exact_distinct = len(set(hashes))
    print(f"\n=== EXACT (WL graph hash) ===")
    print(f"  DISTINCT architectures  {exact_distinct}")
    print(f"  isomorphic duplicates   {len(names) - exact_distinct}"
          f"  ({(len(names)-exact_distinct)/len(names):.1%} of traced)")

    if not mats:
        print("\nno usable embedding matrix; graded count unavailable")
        return 0
    M = np.concatenate(mats, axis=0)
    if M.shape[0] != len(names):
        print(f"  NOTE {len(names)} names but {M.shape[0]} vectors; "
              f"graded count uses the vectors only")

    def cluster(t):
        reps, sizes = [], []
        for i in range(M.shape[0]):
            if reps:
                s = M[reps] @ M[i]
                j = int(np.argmax(s))
                if s[j] >= t:
                    sizes[j] += 1
                    continue
            reps.append(i)
            sizes.append(1)
        return len(reps), max(sizes)

    print(f"\n=== distinct-architecture count vs threshold ===")
    print(f"  {'cosine':>8}{'clusters':>10}{'largest':>9}  note")
    sweep = {}
    for t in a.thresholds:
        k, big = cluster(t)
        sweep[t] = k
        note = ("calibration: distinct archs sit at or BELOW this" if abs(t-0.921) < 1e-9
                else "calibration: trivial variants sit at or ABOVE this" if abs(t-0.985) < 1e-9
                else "gate's operating point" if abs(t-0.95) < 1e-9 else "")
        print(f"  {t:8.3f}{k:10d}{big:9d}  {note}", flush=True)
    if 0.921 in sweep and 0.985 in sweep:
        print(f"\n  => defensible range: {sweep[0.921]} to {sweep[0.985]} "
              f"distinct architectures")

    print(f"\n=== how the untraceable {len(trace_fail)} are counted ===")
    g = sweep.get(0.95, exact_distinct)
    print(f"  excluded (report on traced only)  {g}")
    print(f"  each counted as its own architecture   {g + len(trace_fail)}")
    print(f"  as a fraction of the corpus       "
          f"{len(trace_fail)/total_attempted:.1%} have NO architectural identity")

    for label, d in (("build", build_fail), ("trace", trace_fail)):
        if d:
            print(f"\n=== {label} failures ({len(d)}) ===")
            for k, v in collections.Counter(
                    e.split(":")[0] for e in d.values()).most_common(8):
                print(f"  {v:5d}  {k}")

    out = a.out or os.path.join(a.indir, "diversity.json")
    with open(out, "w") as f:
        json.dump({"attempted": total_attempted, "embedded": len(names),
                   "exact_distinct": exact_distinct,
                   "graded": {str(k): v for k, v in sweep.items()},
                   "build_failures": len(build_fail),
                   "trace_failures": len(trace_fail)}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

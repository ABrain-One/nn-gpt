"""Does the 0.95 threshold still hold now that we have 5,000 models instead of a few dozen?

WHAT 0.95 RESTS ON TODAY. metric_ablation/DECISION_LEDGER.md records the original calibration:
controlled mutants at depth +1..+16 scored 0.985-0.995, cross-family pairs scored 0.812-0.921,
and NOTHING landed in between -- a "clean empty gap". 0.95 is simply the centre of that gap.
It was never derived; it was placed where there was the most room on either side.

WHY THAT DESERVES RE-EXAMINATION. The gap was measured on a small set of HAND-BUILT pairs:
take a seed, apply k mutations, compare. Real generated models are not that. Fifteen generations
of crossover and stacked mutation produce architectures whose relationship to their seed is far
less clean, and the corpus sweep already hints the gap is not empty any more -- the distinct
count moves from 227 at cosine 0.95 to 396 at 0.985, which can only happen if hundreds of real
pairs sit inside the supposedly empty band.

THE GROUND TRUTH IS ALREADY ON DISK and did not exist when the threshold was set. Every GA cache
entry carries `name` (seed plus the mutation chain, e.g. DenoiseThunder+m+s+m) and `parent`.
That gives three populations with known labels, at a scale the original could not reach:

    PARENT-CHILD   one mutation apart. The strongest possible "same architecture" signal.
    SAME-FAMILY    share a seed root, not directly related. Same lineage, many mutations apart.
    CROSS-FAMILY   different seed roots. Known genuinely different.
    CROSSOVER      block-crossover hybrids (name contains X). Deliberately BETWEEN families --
                   the category the original calibration had no examples of at all.

If the populations still separate cleanly, 0.95 is vindicated on real data. If they overlap, the
threshold is arbitrating pairs it was never shown, and the honest report is a range rather than a
number -- which is what the corpus sweep has been saying all along.

  python recalibrate_scale.py --max-per-group 4000
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def seed_root(name):
    """'DenoiseThunder+m+s+m' -> 'DenoiseThunder'. Crossovers keep both parents."""
    return re.split(r"\+", name or "")[0]


def is_crossover(name):
    return "X" in seed_root(name)


def load_lineage():
    """{hash: {name, parent, src}} from every GA run's cache + models dir."""
    out = {}
    for cache in sorted(glob.glob(os.path.join(HERE, "ga_*_cache.json"))
                        + glob.glob(os.path.join(HERE, "seed_ga_*_cache.json"))):
        mdir = cache.replace("_cache.json", "_models")
        if not os.path.isdir(mdir):
            continue
        try:
            c = json.load(open(cache))
        except Exception:
            continue
        for h, v in c.items():
            if not isinstance(v, dict) or not v.get("name"):
                continue
            p = os.path.join(mdir, f"{h}.py")
            if not os.path.exists(p):
                continue
            try:
                src = open(p).read()
            except OSError:
                continue
            # First writer wins: the same hash in two runs is the same source by construction.
            out.setdefault(h, {"name": v["name"], "parent": v.get("parent"), "src": src})
    return out


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def describe(label, v):
    if not v:
        print(f"  {label:<14} (no pairs)")
        return
    print(f"  {label:<14} n={len(v):>6}   "
          f"min {min(v):.3f}  p01 {pct(v,.01):.3f}  p05 {pct(v,.05):.3f}  "
          f"med {pct(v,.5):.3f}  p95 {pct(v,.95):.3f}  max {max(v):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-models", type=int, default=1200,
                    help="cap embedded models; pairs grow quadratically")
    ap.add_argument("--max-per-group", type=int, default=4000, help="pairs sampled per group")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/recalibration.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    lin = load_lineage()
    print(f"=== recalibration at scale | {len(lin)} models with lineage on disk ===",
          flush=True)

    # Keep families that can actually contribute within-family pairs.
    byfam = collections.defaultdict(list)
    for h, v in lin.items():
        byfam[seed_root(v["name"])].append(h)
    fams = sorted(byfam, key=lambda f: -len(byfam[f]))
    keep = []
    for f in fams:
        keep += byfam[f][:max(2, a.max_models // max(1, len(fams)) + 6)]
    rng.shuffle(keep)
    keep = keep[:a.max_models]
    print(f"  embedding {len(keep)} models across {len({seed_root(lin[h]['name']) for h in keep})} "
          f"families", flush=True)

    import numpy as np
    from arch_embed import embed

    vec, ok = {}, 0
    for i, h in enumerate(keep, 1):
        try:
            ns = {}
            exec(compile(lin[h]["src"], "<m>", "exec"), ns)
            net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
            net.eval()
            v = np.asarray(embed(net), dtype=np.float64).ravel()
            n = np.linalg.norm(v)
            if n > 0 and np.isfinite(n):
                vec[h] = v / n
                ok += 1
            del net
        except Exception:
            pass
        if i % 100 == 0 or i == len(keep):
            print(f"  [{i}/{len(keep)}] embedded={ok}", flush=True)

    width = collections.Counter(v.shape[0] for v in vec.values())
    if len(width) > 1:
        w = width.most_common(1)[0][0]
        vec = {k: v for k, v in vec.items() if v.shape[0] == w}
        print(f"  dropped ragged widths, kept {len(vec)} at dim {w}")

    hs = [h for h in keep if h in vec]
    name = {h: lin[h]["name"] for h in hs}
    root = {h: seed_root(name[h]) for h in hs}
    byname = collections.defaultdict(list)
    for h in hs:
        byname[name[h]].append(h)

    def cos(x, y):
        return float(vec[x] @ vec[y])

    # ---- PARENT-CHILD: one mutation apart, the strongest "same" label available.
    pc = []
    for h in hs:
        par = lin[h].get("parent")
        if not par:
            continue
        for ph in byname.get(par, []):
            if ph != h:
                pc.append(cos(h, ph))
                break

    # ---- SAME-FAMILY (not parent-child) and CROSS-FAMILY.
    same, cross, xover = [], [], []
    fam_members = collections.defaultdict(list)
    for h in hs:
        fam_members[root[h]].append(h)
    fam_list = [f for f, m in fam_members.items() if len(m) > 1]
    for _ in range(a.max_per_group):
        if not fam_list:
            break
        f = rng.choice(fam_list)
        x, y = rng.sample(fam_members[f], 2)
        (xover if is_crossover(name[x]) or is_crossover(name[y]) else same).append(cos(x, y))
    roots = [f for f in fam_members if len(fam_members[f]) >= 1]
    for _ in range(a.max_per_group):
        if len(roots) < 2:
            break
        f1, f2 = rng.sample(roots, 2)
        if is_crossover(f1) or is_crossover(f2):
            continue
        cross.append(cos(rng.choice(fam_members[f1]), rng.choice(fam_members[f2])))

    print(f"\n=== pairwise cosine by KNOWN relationship ===")
    describe("parent-child", pc)
    describe("same-family", same)
    describe("crossover", xover)
    describe("cross-family", cross)

    # ---- Is the gap still empty? The original claimed (0.921, 0.985) held nothing.
    print(f"\n=== is the 0.921-0.985 gap still empty? ===")
    tot = 0
    for lab, v in (("parent-child", pc), ("same-family", same),
                   ("crossover", xover), ("cross-family", cross)):
        n = sum(1 for x in v if 0.921 < x < 0.985)
        tot += n
        if v:
            print(f"  {lab:<14} {n:>6} of {len(v):>6} pairs land inside  ({n/len(v):5.1%})")
    print(f"  {'TOTAL':<14} {tot:>6} pairs in the band the calibration called empty")

    # ---- Where would a threshold actually separate "same" from "different"?
    pos = pc + same                      # should be rejected as duplicates
    neg = cross                          # should be kept as distinct
    print(f"\n=== separation quality vs threshold ===")
    print(f"  {'thresh':>7}{'caught dup':>12}{'kept distinct':>15}{'Youden J':>10}")
    best = (None, -2)
    for t in (0.90, 0.921, 0.93, 0.95, 0.97, 0.985, 0.99, 0.995):
        tpr = sum(1 for x in pos if x >= t) / len(pos) if pos else 0
        tnr = sum(1 for x in neg if x < t) / len(neg) if neg else 0
        j = tpr + tnr - 1
        if j > best[1]:
            best = (t, j)
        mark = "  <- current" if abs(t - 0.95) < 1e-9 else ""
        print(f"  {t:7.3f}{tpr:11.1%}{tnr:14.1%}{j:10.3f}{mark}")
    print(f"\n  best separation at {best[0]} (Youden J = {best[1]:.3f})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"models_embedded": len(hs),
               "parent_child": pc, "same_family": same,
               "crossover": xover, "cross_family": cross,
               "best_threshold": best[0], "best_j": best[1]},
              open(a.out, "w"))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

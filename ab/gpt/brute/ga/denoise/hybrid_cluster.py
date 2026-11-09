"""Three-way comparison on the LEMUR zoo: fx (ours), AST (his), HYBRID (the live gate).

WHAT THE HYBRID ACTUALLY IS. diversity_gate.py::_fingerprint_hybrid takes the exact hash from
the AST of every method and the graded vector from the fx trace, because each measured better
at its own job: the AST covers 100% and sees train_setup/learn, ours separates lineage-labelled
pairs at Youden J 0.941 against 0.849. This script scores that exact combination against its
two parents, so the gate's operating point is a measured number rather than an argument.

NO NEW EMBEDDING IS NEEDED. Both halves already exist -- ast_zoo/ has the AST hashes and
vectors, cls_embed3/ has the fx hashes and vectors. The hybrid is a join on model name.

ALL THREE ARE CLUSTERED IN THE SAME ORDER, sorted by name. Greedy leader clustering is
order-dependent -- the same models in a different order give a different count -- so comparing
metrics run under different orders would confound the metric with the shuffle.

THE GRADED COMPARISON IS INTERSECT-ONLY, and that is not a technicality. fx cannot embed 24.6%
of the zoo, so a graded number over the AST's full coverage would be scoring the two on
different populations. Coverage is reported separately, where it belongs, and it is the AST's
real advantage.

  python3 hybrid_cluster.py --ast-in DIR --fx-in DIR
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from prof_cluster import load_dir                                   # noqa: E402


def gate_cluster(order, hash_of, vec_of, t):
    """Replay DiversityGate.check/add over `order`. -> (n_accepted, n_exact, n_graded).

    MIRRORS THE LIVE GATE, including the part that looks like a bug and is not: a candidate
    refused as an exact clone never has its VECTOR added (add() returns early when the hash is
    already present), so the reference set holds one vector per distinct hash, not one per
    model. Counting it any other way would measure a gate we do not run.

    hash_of may be None for a model (metric could not identify it) -- the gate admits those
    and lets training judge, so they are accepted without joining the reference set.
    """
    import numpy as np
    seen = set()
    n = len(order)
    d = len(next(iter(vec_of.values()))) if vec_of else 0
    R = np.empty((n, d), dtype=np.float32) if d else None
    k = 0
    acc = ex = gr = 0
    for nm in order:
        h = hash_of.get(nm)
        if h is not None and h in seen:
            ex += 1
            continue
        v = vec_of.get(nm)
        if v is not None and k and t is not None:
            if float((R[:k] @ v).max()) >= t:
                gr += 1
                continue
        acc += 1
        if h is not None:
            seen.add(h)
        if v is not None and R is not None:
            R[k] = v
            k += 1
    return acc, ex, gr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-in", default="/a/mm/sweep_out/ast_zoo")
    ap.add_argument("--fx-in", default="/a/mm/sweep_out/cls_embed3")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/hybrid_zoo.json")
    a = ap.parse_args()

    import json
    import numpy as np

    print("=== loading ===")
    an, ah, aM, afail = load_dir(a.ast_in)
    fn, fh, fM, ffail = load_dir(a.fx_in)
    print(f"  AST  {len(an)} hashed, {len(afail)} failed")
    print(f"  fx   {len(fn)} embedded, {len(ffail)} failed")

    ast_h = dict(zip(an, ah))
    fx_h = dict(zip(fn, fh))
    ast_v = {n: aM[i] for i, n in enumerate(an)} if len(aM) else {}
    fx_v = {n: fM[i] for i, n in enumerate(fn)} if len(fM) else {}

    union = sorted(set(an) | set(fn))
    inter = sorted(set(an) & set(fn))
    print(f"\n=== COVERAGE ===")
    print(f"  {'metric':<10}{'identified':>12}{'of union':>10}")
    print(f"  {'fx':<10}{len(fn):>12}{len(fn)/len(union):>9.1%}")
    print(f"  {'AST':<10}{len(an):>12}{len(an)/len(union):>9.1%}")
    # The hybrid's exact half falls back to fx when the AST cannot parse, so its coverage is
    # the union, not either parent. That is the whole point of splitting the two halves.
    print(f"  {'HYBRID':<10}{len(union):>12}{1.0:>9.1%}")
    print(f"  intersect (graded comparison runs here)  {len(inter)}")

    res = {"union": len(union), "intersect": len(inter),
           "fx_covered": len(fn), "ast_covered": len(an), "exact": {}, "graded": {}}

    print(f"\n=== EXACT IDENTITY (no threshold) ===")
    print(f"  {'metric':<10}{'population':>12}{'distinct':>10}{'dupes':>10}")
    for lab, hm, pop in (("fx", fx_h, fn), ("AST", ast_h, an),
                         # HYBRID = AST where it parsed, fx otherwise. Over the union, so it
                         # is the only one of the three with an opinion on every model.
                         ("HYBRID", {n: ast_h.get(n, fx_h.get(n)) for n in union}, union)):
        vals = [hm[n] for n in pop if hm.get(n) is not None]
        dis = len(set(vals))
        print(f"  {lab:<10}{len(pop):>12}{dis:>10}{len(vals)-dis:>10}"
              f"  ({(len(vals)-dis)/max(len(vals),1):.1%})")
        res["exact"][lab] = {"population": len(pop), "distinct": dis}

    print(f"\n=== EXACT, ON THE INTERSECT ONLY (like-for-like) ===")
    print(f"  {'metric':<10}{'distinct':>10}{'dupes':>10}")
    for lab, hm in (("fx", fx_h), ("AST", ast_h),
                    ("HYBRID", {n: ast_h.get(n, fx_h.get(n)) for n in inter})):
        vals = [hm[n] for n in inter if hm.get(n) is not None]
        dis = len(set(vals))
        print(f"  {lab:<10}{dis:>10}{len(vals)-dis:>10}  "
              f"({(len(vals)-dis)/max(len(vals),1):.1%})")
        res["exact"][lab + "_intersect"] = dis

    if not inter or not ast_v or not fx_v:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}  (no graded pass: missing vectors)")
        return 0

    print(f"\n=== AS THE GATE WOULD RUN IT (exact pre-gate + graded) ===")
    print(f"  population {len(inter)} (intersect), greedy in name order\n")
    print(f"  {'cos':>6}  {'metric':<10}{'accepted':>10}{'exact-dup':>11}{'near-dup':>10}")
    combos = (("fx", fx_h, fx_v), ("AST", ast_h, ast_v), ("HYBRID", ast_h, fx_v))
    for t in a.thresholds:
        for lab, hm, vm in combos:
            acc, ex, gr = gate_cluster(inter, hm, vm, t)
            print(f"  {t:6.3f}  {lab:<10}{acc:>10}{ex:>11}{gr:>10}", flush=True)
            res["graded"].setdefault(str(t), {})[lab] = {
                "accepted": acc, "exact_dup": ex, "near_dup": gr}
        print()

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

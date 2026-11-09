"""Merge prof_embed_corpus.py shards and report distinct-architecture counts. CPU, no torch.

TWO SLICES, ALWAYS, because the AST and fx runs do not measure the same population. fx lost
24.6% of the LEMUR zoo to TraceError; the AST metric loses almost nothing. Reporting only the
AST's full coverage against the fx number would credit the AST with a difference that is
mostly just the models fx never saw:

  ALL         every model the AST could hash -- the honest coverage of this metric.
  INTERSECT   only models BOTH metrics embedded -- the only apples-to-apples comparison,
              and the one that belongs next to the published 10,479 / 28.3% figures.

CLUSTERING IS REWRITTEN, NOT REUSED. uid_full.py and cluster_corpus.py both do `M[reps] @ M[i]`
inside the loop. That is fancy indexing, so it MATERIALISES the rep matrix on every iteration:
at ~10k reps and 4,096 dims that is a 160 MB copy per model, 19,397 times. It completed on the
denoise corpus because reps stayed small, and it would not have completed here. A contiguous
rep buffer makes the same product a view plus one BLAS call.

FLOAT32 THROUGHOUT. wl_embed returns float64; at 19,397 x 4,096 that is 635 MB before any
clustering, and the extra precision is meaningless for a cosine threshold quoted to 3 places.

  python3 prof_cluster.py --in DIR --fx-in DIR2
"""
import argparse
import collections
import glob
import json
import os


def load_dir(indir):
    """-> (names, hashes, M, failures). M rows correspond to names positionally."""
    import numpy as np
    names, hashes, mats, failed = [], [], [], {}
    for p in sorted(glob.glob(os.path.join(indir, "shard*.npz"))):
        z = np.load(p, allow_pickle=True)
        n = list(z["names"])
        v = z["vecs"]
        if n and not v.size:
            # Ragged widths mean the shard's vectors were not saved. Dropping the NAMES too
            # keeps names and rows aligned; keeping them would shift every later row.
            print(f"  WARNING {os.path.basename(p)}: {len(n)} names but no vectors "
                  f"(ragged) -- shard excluded")
            continue
        names += n
        hashes += list(z["hashes"])
        if v.size:
            mats.append(np.asarray(v, dtype=np.float32))
    for p in sorted(glob.glob(os.path.join(indir, "fail*.json"))):
        j = json.load(open(p))
        failed.update(j.get("build", {}))
        failed.update(j.get("trace", {}))
    M = np.concatenate(mats, axis=0) if mats else np.zeros((0, 0), np.float32)
    return names, hashes, M, failed


def cluster(M, thresholds):
    """Greedy leader clustering at each threshold. -> {t: (n_clusters, largest)}.

    One pass per threshold against a CONTIGUOUS rep buffer: R[:k] is a view, so the whole
    inner step is a single gemv with no allocation. See the module docstring for why the
    obvious `M[reps] @ M[i]` does not survive this corpus size.
    """
    import numpy as np
    out = {}
    n, d = M.shape
    for t in thresholds:
        R = np.empty((n, d), dtype=np.float32)
        sizes = []
        k = 0
        for i in range(n):
            if k:
                s = R[:k] @ M[i]
                j = int(np.argmax(s))
                if s[j] >= t:
                    sizes[j] += 1
                    continue
            R[k] = M[i]
            k += 1
            sizes.append(1)
        out[t] = (k, max(sizes) if sizes else 0)
        del R
        print(f"  {t:8.3f}{k:10d}{max(sizes) if sizes else 0:9d}", flush=True)
    return out


def report(label, names, hashes, M, thresholds):
    import numpy as np
    print(f"\n{'='*66}\n=== {label}  |  {len(names)} models ===")
    exact = len(set(hashes))
    print(f"  DISTINCT (exact WL hash)  {exact}")
    print(f"  isomorphic duplicates     {len(names) - exact}"
          f"  ({(len(names)-exact)/max(len(names),1):.1%})")
    if not len(M) or not thresholds:
        return {"models": len(names), "exact_distinct": exact, "graded": {}}
    print(f"\n  {'cosine':>8}{'clusters':>10}{'largest':>9}")
    sw = cluster(M, thresholds)
    return {"models": len(names), "exact_distinct": exact,
            "graded": {str(t): v[0] for t, v in sw.items()},
            "largest": {str(t): v[1] for t, v in sw.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--fx-in", default="/a/mm/sweep_out/cls_embed3",
                    help="the torch.fx run over the SAME rows; used for the INTERSECT slice "
                         "and the merge/split comparison")
    # SIX, NOT EIGHT, AND THEY OVERLAP THE fx CURVE ON PURPOSE. Each threshold is a full
    # sequential pass whose cost is memory traffic, not arithmetic: at ~12k reps the buffer is
    # 200 MB and gets read once per model, so one pass moves ~4 TB and two slices x 8
    # thresholds is over an hour of pure bandwidth. These six include every operating point
    # cluster_corpus.py reported (0.90 / 0.921 / 0.95 / 0.97 / 0.99), so the two curves stay
    # directly comparable; 0.85 is kept because the AST metric's cross-family floor sits lower
    # than ours and the curve would otherwise start above its interesting region.
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.85, 0.90, 0.921, 0.95, 0.97, 0.99])
    ap.add_argument("--no-graded", action="store_true",
                    help="skip the threshold sweep entirely. It is 30-50 min of pure memory "
                         "bandwidth and answers a DIFFERENT question (how the count varies "
                         "with the operating point) from coverage / exact identity / "
                         "cross-metric splits, which take about two minutes. Run without it "
                         "first; add the sweep only if the cheap numbers justify a curve.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.no_graded:
        a.thresholds = []

    import numpy as np

    print("=== loading AST shards ===")
    names, hashes, M, failed = load_dir(a.indir)
    attempted = len(names) + len(failed)
    print(f"  hashed   {len(names)}/{attempted}  ({len(names)/max(attempted,1):.1%})")
    print(f"  failed   {len(failed)}")
    for k, v in collections.Counter(e.split(":")[0]
                                    for e in failed.values()).most_common(8):
        print(f"    {v:5d}  {k}")

    if not names:
        print("\nnothing hashed")
        return 1

    res = {"attempted": attempted, "failed": len(failed),
           "all": report("ALL (AST coverage)", names, hashes, M, a.thresholds)}

    fx_names, fx_hashes, fx_M, fx_failed = [], [], None, {}
    if a.fx_in and os.path.isdir(a.fx_in):
        print("\n=== loading fx shards (for the INTERSECT slice) ===")
        fx_names, fx_hashes, fx_M, fx_failed = load_dir(a.fx_in)
        print(f"  fx embedded {len(fx_names)}")

    if fx_names:
        keep = set(fx_names)
        idx = [i for i, n in enumerate(names) if n in keep]
        res["intersect"] = report(
            "INTERSECT (both metrics embedded it)",
            [names[i] for i in idx], [hashes[i] for i in idx],
            M[idx] if len(M) else M, a.thresholds)

        # WHAT EACH METRIC SEES THAT THE OTHER DOES NOT. The professor's objection is that
        # fx ignores __init__/train_setup/learn, so the interesting number is groups fx calls
        # identical that the AST separates. The reverse direction is reported too and is not
        # symmetric noise: it is where fx's shape-aware trace distinguishes models whose
        # source is alpha-equivalent.
        ast_of = dict(zip(names, hashes))
        fx_of = dict(zip(fx_names, fx_hashes))
        both = [n for n in fx_of if n in ast_of]
        gb_fx, gb_ast = collections.defaultdict(set), collections.defaultdict(set)
        for n in both:
            gb_fx[fx_of[n]].add(ast_of[n])
            gb_ast[ast_of[n]].add(fx_of[n])
        split_by_ast = sum(1 for v in gb_fx.values() if len(v) > 1)
        split_by_fx = sum(1 for v in gb_ast.values() if len(v) > 1)
        print(f"\n=== CROSS-METRIC (on the {len(both)} both embedded) ===")
        print(f"  fx-identical groups the AST SEPARATES   {split_by_ast}"
              f"  of {len(gb_fx)}  ({split_by_ast/max(len(gb_fx),1):.1%})")
        print(f"     -> these differ in __init__/train_setup/learn, which fx cannot see")
        print(f"  AST-identical groups fx SEPARATES       {split_by_fx}"
              f"  of {len(gb_ast)}  ({split_by_fx/max(len(gb_ast),1):.1%})")
        print(f"     -> alpha-equivalent source whose traced graph still differs")
        res["cross"] = {"both": len(both), "fx_groups": len(gb_fx),
                        "split_by_ast": split_by_ast, "ast_groups": len(gb_ast),
                        "split_by_fx": split_by_fx}

        only_ast = len(set(names) - set(fx_names))
        print(f"\n  models ONLY the AST could hash          {only_ast}")
        print(f"  models ONLY fx could embed              "
              f"{len(set(fx_names) - set(names))}")
        res["only_ast"] = only_ast
        res["only_fx"] = len(set(fx_names) - set(names))

    ctrl_path = "/a/mm/incoming/neg_control.py"
    if os.path.exists(ctrl_path) and len(M):
        # The author's own negative control. On denoise it scored 0.159-0.466 against a corpus
        # whose internal floor was 0.258 -- a POPULATED gap, not the clean one the 200-model
        # sample suggested. Whether it separates on a corpus this diverse is the open question.
        try:
            import sys
            import tempfile
            for _p in (os.path.dirname(os.path.abspath(__file__)), "/a/mm/incoming"):
                if os.path.isdir(_p) and _p not in sys.path:
                    sys.path.insert(0, _p)
            import ast_model_graph as P
            from prof.uid_full import load_class, source_graph_ext
            C = load_class(tempfile.mkdtemp(prefix="ctrl_"), "neg_control",
                           open(ctrl_path).read(), attr="Mlp")
            g = source_graph_ext(C)
            vc = np.asarray(P.wl_embed(*P.adjacency(g), P.featurize(g)),
                            dtype=np.float32).ravel()
            vc /= np.linalg.norm(vc)
            cs = M @ vc
            print(f"\n=== negative control (his MLP) vs the zoo ===")
            print(f"  control min {cs.min():.3f}  median {np.median(cs):.3f}  "
                  f"max {cs.max():.3f}")
            over = int((cs >= 0.90).sum())
            print(f"  zoo models scoring >= 0.90 against an MLP: {over}"
                  f"  ({over/len(cs):.2%})")

            # THE CONTROL SCORE ALONE MEANS NOTHING. What makes a threshold defensible is a
            # GAP: the corpus's own internal floor sitting above the control's ceiling. On
            # denoise that gap looked clean on a 200-model sample (+0.272) and was POPULATED
            # on the full 1,468 (-0.245), which is exactly the error a one-sided number
            # invites. So the floor is measured here rather than assumed.
            #
            # BLOCKED, because the full Gram matrix is 19,397^2 -- 188M pairs, 1.5 GB at
            # float32 and 1.5e12 flops. Row blocks keep the working set at ~40 MB and let the
            # min be reduced incrementally; percentiles come from a fixed random subsample,
            # which is what they are estimating anyway.
            lo_floor, sample = 1.0, []
            rng = np.random.default_rng(0)
            B = 512
            for s in range(0, len(M), B):
                blk = M[s:s + B] @ M.T                       # (b, n)
                for r in range(blk.shape[0]):
                    blk[r, s + r] = 2.0                      # mask self-similarity
                lo_floor = min(lo_floor, float(blk.min()))
                sample.append(rng.choice(blk.ravel(), size=min(20000, blk.size),
                                         replace=False))
                del blk
            samp = np.concatenate(sample)
            samp = samp[samp <= 1.0]                         # drop the masked diagonal
            gap = lo_floor - float(cs.max())
            print(f"\n  zoo-internal pairwise: min {lo_floor:.3f}  "
                  f"p01 {np.percentile(samp,1):.3f}  p05 {np.percentile(samp,5):.3f}  "
                  f"median {np.median(samp):.3f}")
            print(f"  SEPARATION: lowest zoo-internal {lo_floor:.3f} vs highest control "
                  f"{cs.max():.3f}  -> gap {gap:+.3f}")
            print("  " + ("=> clean gap; a threshold in it is defensible."
                          if gap > 0 else
                          "=> POPULATED gap; no single threshold cleanly separates the "
                          "control from the corpus."))
            res["control"] = {"min": float(cs.min()), "median": float(np.median(cs)),
                              "max": float(cs.max()), "over_0.90": over,
                              "zoo_internal_min": lo_floor,
                              "zoo_internal_p01": float(np.percentile(samp, 1)),
                              "zoo_internal_median": float(np.median(samp)),
                              "gap": gap}
        except Exception as e:
            print(f"  control failed: {type(e).__name__}: {e}")

    out = a.out or os.path.join(a.indir, "diversity.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

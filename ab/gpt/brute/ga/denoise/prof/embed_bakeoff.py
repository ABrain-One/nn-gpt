"""Which embedding actually discriminates better -- ours or the professor's?

WHY THE OBVIOUS COMPARISON IS THE WRONG ONE. At cosine 0.95 our metric yields 227 clusters on
this corpus and his yields 84, and it is tempting to read that as ours being finer and
therefore better. It is not evidence. A metric can be finer because it resolves real
architectural differences, or because it is noisy; cluster count alone cannot tell those
apart. The claim "ours discriminates better" was asserted from that number and should not
have been.

THE TEST THAT DOES ANSWER IT is the one used to recalibrate our own threshold: score pairs
whose relationship is KNOWN and ask which metric separates them. Every GA cache entry records
`name` (seed plus mutation chain) and `parent`, giving four labelled populations:

    parent-child   one mutation apart          -> should score HIGH (same architecture)
    same-family    shared seed, many mutations -> should score HIGH
    cross-family   different seed roots        -> should score LOW  (different architecture)
    crossover      block-crossover hybrids     -> deliberately between

Separation is summarised by Youden's J (true-positive rate + true-negative rate - 1) at each
threshold, so the two metrics are compared on the same pairs with the same statistic rather
than on a granularity number that flatters whichever happens to be finer.

The professor's negative control (an MLP) is included as a third anchor: whatever else is
true, a model from an entirely different family should score below every within-corpus pair,
and a metric where it does not has an interpretability problem regardless of its J.

Both embeddings are computed on the SAME models in the SAME process, so nothing differs
except the metric under test.

  python embed_bakeoff.py --max-models 900 --max-per-group 3000
"""
import argparse
import ast
import collections
import glob
import importlib.util
import inspect
import json
import os
import random
import re
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
GA_DIR = os.path.dirname(HERE)
for _p in (HERE, GA_DIR, "/a/mm/incoming", "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ast_model_graph as P                                        # noqa: E402
from uid_full import source_graph_ext, load_class                  # noqa: E402

IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def seed_root(name):
    return re.split(r"\+", name or "")[0]


def is_crossover(name):
    return "X" in seed_root(name)


def load_lineage():
    out = {}
    for cache in sorted(glob.glob(os.path.join(GA_DIR, "ga_*_cache.json"))
                        + glob.glob(os.path.join(GA_DIR, "seed_ga_*_cache.json"))):
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
                out.setdefault(h, {"name": v["name"], "parent": v.get("parent"),
                                   "src": open(p).read()})
            except OSError:
                pass
    return out


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-models", type=int, default=900)
    ap.add_argument("--max-per-group", type=int, default=3000)
    ap.add_argument("--control", default="/a/mm/incoming/neg_control.py")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/embed_bakeoff.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import numpy as np
    import torch                                                   # noqa: F401
    from arch_embed import embed as ours_embed

    lin = load_lineage()
    keys = list(lin)
    rng.shuffle(keys)
    keys = keys[:a.max_models]
    print(f"=== embedding bake-off | {len(keys)} lineage-labelled models ===", flush=True)

    tmp = tempfile.mkdtemp(prefix="bake_")
    OUR, HIS, NAME = {}, {}, {}
    for i, h in enumerate(keys, 1):
        src, nm = lin[h]["src"], lin[h]["name"]
        try:                                            # theirs: AST, no build
            K = load_class(tmp, h, src)
            g = source_graph_ext(K)
            vh = np.asarray(P.wl_embed(*P.adjacency(g), P.featurize(g)),
                            dtype=np.float64).ravel()
        except Exception:
            continue
        try:                                            # ours: build + fx trace
            ns = {}
            exec(compile(src, "<m>", "exec"), ns)
            net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
            net.eval()
            vo = np.asarray(ours_embed(net), dtype=np.float64).ravel()
            del net
        except Exception:
            continue
        no, nh = np.linalg.norm(vo), np.linalg.norm(vh)
        if no == 0 or nh == 0 or not (np.isfinite(no) and np.isfinite(nh)):
            continue
        OUR[h], HIS[h], NAME[h] = vo / no, vh / nh, nm
        if i % 100 == 0 or i == len(keys):
            print(f"  [{i}/{len(keys)}] both-embedded={len(OUR)}", flush=True)

    # keep only models BOTH metrics produced a vector for, so the pair sets are identical
    hs = [h for h in keys if h in OUR and h in HIS]
    for D in (OUR, HIS):
        w = collections.Counter(D[h].shape[0] for h in hs).most_common(1)[0][0]
        hs = [h for h in hs if D[h].shape[0] == w]
    print(f"\n  {len(hs)} models embedded by BOTH metrics\n", flush=True)

    root = {h: seed_root(NAME[h]) for h in hs}
    byname = collections.defaultdict(list)
    for h in hs:
        byname[NAME[h]].append(h)
    fam = collections.defaultdict(list)
    for h in hs:
        fam[root[h]].append(h)

    pairs = {"parent-child": [], "same-family": [], "crossover": [], "cross-family": []}
    for h in hs:
        par = lin[h].get("parent")
        if par:
            for ph in byname.get(par, []):
                if ph != h:
                    pairs["parent-child"].append((h, ph))
                    break
    multi = [f for f, m in fam.items() if len(m) > 1]
    for _ in range(a.max_per_group):
        if not multi:
            break
        f = rng.choice(multi)
        x, y = rng.sample(fam[f], 2)
        pairs["crossover" if (is_crossover(NAME[x]) or is_crossover(NAME[y]))
              else "same-family"].append((x, y))
    roots = [f for f in fam if not is_crossover(f)]
    for _ in range(a.max_per_group):
        if len(roots) < 2:
            break
        f1, f2 = rng.sample(roots, 2)
        pairs["cross-family"].append((rng.choice(fam[f1]), rng.choice(fam[f2])))

    def scores(D, ps):
        return [float(D[x] @ D[y]) for x, y in ps]

    res = {}
    for label, D in (("OURS (continuous-WL on fx trace)", OUR),
                     ("PROF (continuous-WL on AST graph)", HIS)):
        print(f"=== {label} ===")
        s = {k: scores(D, v) for k, v in pairs.items() if v}
        for k, v in s.items():
            print(f"  {k:<14} n={len(v):>5}  min {min(v):.3f}  p05 {pct(v,.05):.3f}  "
                  f"med {pct(v,.5):.3f}  max {max(v):.3f}")
        pos = s.get("parent-child", []) + s.get("same-family", [])
        neg = s.get("cross-family", [])
        best = (None, -2)
        print(f"  {'thresh':>8}{'catch dup':>11}{'keep distinct':>15}{'Youden J':>10}")
        for t in (0.80, 0.85, 0.90, 0.921, 0.95, 0.97, 0.985, 0.99):
            tpr = sum(1 for x in pos if x >= t) / len(pos) if pos else 0
            tnr = sum(1 for x in neg if x < t) / len(neg) if neg else 0
            j = tpr + tnr - 1
            if j > best[1]:
                best = (t, j)
            print(f"  {t:8.3f}{tpr:10.1%}{tnr:14.1%}{j:10.3f}")
        print(f"  -> best {best[0]} (J = {best[1]:.3f})\n")
        res[label] = {"best_threshold": best[0], "best_j": best[1],
                      "dist": {k: [min(v), pct(v, .5), max(v)] for k, v in s.items()}}

    # negative control as a third anchor
    if os.path.exists(a.control):
        print("=== negative control (his MLP) vs the corpus ===")
        try:
            C = load_class(tmp, "ctrl", open(a.control).read(), attr="Mlp")
            gC = source_graph_ext(C)
            vch = np.asarray(P.wl_embed(*P.adjacency(gC), P.featurize(gC))).ravel()
            vch /= np.linalg.norm(vch)
            ch = [float(vch @ HIS[h]) for h in hs]
            print(f"  PROF: control {min(ch):.3f}-{max(ch):.3f}   "
                  f"corpus-internal min {min(scores(HIS, pairs['cross-family'])):.3f}")
            ns = {}
            exec(compile(open(a.control).read(), "<c>", "exec"), ns)
            netc = ns["Mlp"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
            netc.eval()
            vco = np.asarray(ours_embed(netc)).ravel()
            vco /= np.linalg.norm(vco)
            co = [float(vco @ OUR[h]) for h in hs]
            print(f"  OURS: control {min(co):.3f}-{max(co):.3f}   "
                  f"corpus-internal min {min(scores(OUR, pairs['cross-family'])):.3f}")
        except Exception as e:
            print(f"  control failed: {type(e).__name__}: {e}")

    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

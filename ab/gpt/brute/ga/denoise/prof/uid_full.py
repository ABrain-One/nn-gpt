"""Architectural UID over a whole LEMUR task, using the professor's AST approach.

WHY THIS REPLACES OUR torch.fx IDENTITY. Ours traces `forward` and nothing else, so it cannot
see train_setup or learn. Measured on our corpus: of 44 groups our hash calls identical, 34
(77%) differ in their training code. The clearest case is a released pair --

    DenoiseUnet5 vs DenoiseUnet5F      identical under our hash
      loss       MSE          vs  Charbonnier
      clip_norm  0.1          vs  0.5
      score      37.44 dB     vs  34.76 dB          2.68 dB apart

-- in a corpus where 73% of models sit inside one dB. The AST approach reads every method the
model defines, so optimizer, loss, clipping and the update loop all reach the identity.

It is also strictly cheaper: no model is built and nothing is traced, so the 24.6% of the
LEMUR zoo that torch.fx cannot handle (3,478 TraceError from data-dependent control flow)
simply is not a failure mode here. Measured 22.7 models/sec against ~4/sec for ours.

THE ONE FIX APPLIED. source_graph() collects methods via inspect.getmro(cls), which reaches
Net and its base classes but NOT the helper nn.Module classes models define alongside Net. On
200 models that silently merged 39 groups whose architectures genuinely differ -- e.g. two
models with byte-identical Net bodies whose ELUAvgBlock used kernels [3,5,1] against [3,3,1].
Collecting module-level classes as well takes the distinct count from 115 to 199 and the bad
merges from 39 to 0.

CALIBRATION comes from the author's own negative control (neg_control.py, an MLP). On our
corpus that model scores 0.285-0.407 against every denoise architecture, while our models
score 0.679-1.000 against each other. The gap is empty, which is what makes a threshold on
this metric defensible -- notably unlike the 0.921-0.985 gap our own metric was calibrated
into, which real generated models later populated with 1,450 pairs.

  python uid_full.py --task img-denoising --db /a/mm/nn-dataset/db/ab.nn.db
"""
import argparse
import ast
import collections
import importlib.util
import inspect
import json
import os
import sqlite3
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ast_model_graph as P                                        # noqa: E402


def source_graph_ext(cls):
    """P.source_graph plus the module-level classes the model defines.

    The only change from the reference implementation, and it is load-bearing: helper blocks
    live beside Net rather than in its MRO, so without this every architectural difference
    inside a block is invisible.
    """
    g, attrs, uses = P._Graph(), {}, {}
    root = cls.__module__.split(".")[0]
    klasses = list(inspect.getmro(cls))
    mod = sys.modules.get(cls.__module__)
    if mod is not None:
        for v in vars(mod).values():
            if inspect.isclass(v) and v.__module__ == cls.__module__ and v not in klasses:
                klasses.append(v)
    seen, fns = set(), []
    for klass in klasses:
        if klass.__module__.split(".")[0] != root:
            continue
        for _, f in vars(klass).items():
            if inspect.isfunction(f) and id(f) not in seen:
                seen.add(id(f))
                fns.append(f)
    for f in fns:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        except (OSError, SyntaxError, IndentationError):
            continue
        P._Dataflow(g, attrs, uses).visit(tree)
    for name, consumers in uses.items():
        for producer in attrs.get(name, []):
            for c in consumers:
                g.link(producer, c, 0)
    return g


def load_class(tmpdir, name, code, attr="Net"):
    """inspect.getsource needs a real file; DB code exec'd in memory has none.

    Anyone applying this to nn.code hits this immediately -- exec(compile(...)) produces
    functions whose source cannot be recovered, so source_graph silently returns an empty
    graph and every model hashes the same. Materialising to a temp file avoids it.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:60]
    p = os.path.join(tmpdir, f"m_{safe}.py")
    with open(p, "w") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location(f"m_{safe}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return getattr(m, attr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--task", default="img-denoising")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", default="/a/mm/incoming/neg_control.py")
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/prof_uid.json")
    a = ap.parse_args()

    import numpy as np

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code, n.arch_uid FROM nn n WHERE n.name IN "
        "(SELECT DISTINCT nn FROM stat WHERE task=?)", (a.task,)).fetchall()
    con.close()
    if a.limit:
        rows = rows[:a.limit]
    print(f"=== AST-based UID | {len(rows)} {a.task} models ===", flush=True)

    tmp = tempfile.mkdtemp(prefix="uid_")
    H, V, OURS, failed = {}, {}, {}, {}
    t0 = time.time()
    for i, (nm, code, uid) in enumerate(rows, 1):
        try:
            K = load_class(tmp, nm, code)
            g = source_graph_ext(K)
            if not g.keys:
                raise ValueError("empty graph (source not recoverable)")
            H[nm] = P.wl_hash(g)
            V[nm] = P.wl_embed(*P.adjacency(g), P.featurize(g))
            if uid:
                OURS[nm] = uid
        except Exception as e:
            failed[nm] = f"{type(e).__name__}: {e}"[:110]
        if i % 200 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  [{i}/{len(rows)}] ok={len(H)} failed={len(failed)} "
                  f"{el:.0f}s ({i/max(el,1e-9):.1f}/s)", flush=True)

    print(f"\n=== COVERAGE ===")
    print(f"  hashed              {len(H)}/{len(rows)}  ({len(H)/max(len(rows),1):.1%})")
    print(f"  failed              {len(failed)}")
    if failed:
        for k, v in collections.Counter(e.split(":")[0]
                                        for e in failed.values()).most_common(5):
            print(f"    {v:5d}  {k}")
    print(f"  our torch.fx uid    {len(OURS)}/{len(rows)}  "
          f"({len(OURS)/max(len(rows),1):.1%})")

    print(f"\n=== EXACT IDENTITY ===")
    print(f"  distinct (AST, all methods)   {len(set(H.values()))}")
    if OURS:
        print(f"  distinct (ours, forward only) {len(set(OURS.values()))}")
        # models ours calls identical that the AST separates = the training-code differences
        gb = collections.defaultdict(set)
        for nm, u in OURS.items():
            if nm in H:
                gb[u].add(H[nm])
        split = sum(1 for v in gb.values() if len(v) > 1)
        print(f"  groups ours merges that AST separates  {split}"
              f"   <- differ in train_setup/learn/__init__")

    if not V:
        return 1
    names = list(V)
    M = np.stack([V[n] for n in names])
    S = M @ M.T
    np.fill_diagonal(S, -1)
    pw = S[np.triu_indices(len(names), k=1)]

    print(f"\n=== SIMILARITY DISTRIBUTION ===")
    print(f"  within corpus   min {pw.min():.3f}  p05 {np.percentile(pw,5):.3f}  "
          f"median {np.median(pw):.3f}  max {pw.max():.3f}")

    ctrl = None
    if os.path.exists(a.control):
        try:
            C = load_class(tmp, "neg_control", open(a.control).read(), attr="Mlp")
            gC = source_graph_ext(C)
            vc = P.wl_embed(*P.adjacency(gC), P.featurize(gC))
            cs = np.array([float(vc @ V[n]) for n in names])
            ctrl = (float(cs.min()), float(np.median(cs)), float(cs.max()))
            print(f"  negative control (MLP) min {cs.min():.3f}  "
                  f"median {np.median(cs):.3f}  max {cs.max():.3f}")
            gap = pw.min() - cs.max()
            print(f"  SEPARATION: lowest within-corpus {pw.min():.3f} vs highest "
                  f"control {cs.max():.3f}  -> gap {gap:+.3f}")
            print("  " + ("=> clean gap; a threshold in it is defensible."
                          if gap > 0 else
                          "=> POPULATED gap; no single threshold cleanly separates."))
        except Exception as e:
            print(f"  control failed: {type(e).__name__}: {e}")

    print(f"\n=== distinct architectures vs threshold ===")
    order = sorted(range(len(names)))
    sweep = {}
    print(f"  {'cosine':>8}{'clusters':>10}{'largest':>9}")
    for t in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995):
        reps, sizes = [], []
        for i in order:
            if reps:
                s = M[reps] @ M[i]
                j = int(np.argmax(s))
                if s[j] >= t:
                    sizes[j] += 1
                    continue
            reps.append(i)
            sizes.append(1)
        sweep[t] = len(reps)
        print(f"  {t:8.3f}{len(reps):10d}{max(sizes):9d}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"task": a.task, "models": len(rows), "hashed": len(H),
               "failed": len(failed), "distinct_exact": len(set(H.values())),
               "ours_distinct": len(set(OURS.values())) if OURS else None,
               "graded": {str(k): v for k, v in sweep.items()},
               "control": ctrl}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

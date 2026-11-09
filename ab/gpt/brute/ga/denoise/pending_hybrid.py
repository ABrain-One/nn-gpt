"""Are the unmerged GA models actually distinct? Replays the hybrid gate over them.

WHY THIS IS NOT ALREADY KNOWN. Each GA variant ran with its OWN diversity store, so a
candidate was only ever checked against models that variant had already accepted. Nothing
ever compared `cheap` against `distant`, or either against `astuid`. Two runs seeded from the
same corpus can converge on the same architecture independently and both keep it, and the
merge would silently import the pair.

THE MASTER CORPUS IS LOADED FIRST, as the reference set. A pending model that re-enacts
something already in master is a duplicate no matter which run produced it.

ORDER IS BY VARIANT, THEN NAME, and it matters: greedy leader clustering is order-dependent,
so whichever variant goes first gets the benefit of the doubt on any mutual collision. The
per-variant "lost to an earlier variant" column makes that visible instead of hiding it in a
total.

  python pending_hybrid.py --master DB --variants cheap distant astuid
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def codes(db, names=None):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT name, code FROM nn ORDER BY name").fetchall()
    con.close()
    return [(n, c) for n, c in rows if names is None or n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--variants", nargs="+", default=["cheap", "distant", "astuid"])
    ap.add_argument("--ga-db-root", default="/a/mm/sweep_out/ga_db")
    ap.add_argument("--cos", type=float, default=0.95)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/pending_hybrid.json")
    a = ap.parse_args()

    from diversity_gate import DiversityGate

    master = codes(a.master)
    master_names = {n for n, _ in master}
    print(f"=== reference: {len(master)} models already in master ===", flush=True)

    # ast_exact=True + graded=True is the hybrid: exact identity from the AST of every
    # method, graded cosine from the fx trace. store_path=None so this NEVER writes -- it is
    # a measurement of the pending set, not a gate run that should influence anything.
    gate = DiversityGate(store_path=None, graded=True, cos_max=a.cos,
                         ast_exact=True, log=lambda *_: None)

    t0 = time.time()
    seeded = 0
    for i, (nm, code) in enumerate(master, 1):
        try:
            _, fp = gate.check(code)
            gate.add(nm, fp)
            seeded += 1
        except Exception:
            pass
        if i % 250 == 0:
            print(f"  seeded {i}/{len(master)}  {time.time()-t0:.0f}s", flush=True)
    print(f"  reference set holds {len(gate.accepted)} exact hashes, "
          f"{len(gate.embeds)} vectors ({seeded} seeded, {time.time()-t0:.0f}s)\n", flush=True)

    origin = {}                      # accepted pending name -> variant that got it
    res = {"master": len(master), "cos": a.cos, "variants": {}}
    rows = []

    for v in a.variants:
        db = os.path.join(a.ga_db_root, v, "ab.nn.db")
        if not os.path.exists(db):
            print(f"  !! {db} missing, skipping")
            continue
        pend = [(n, c) for n, c in codes(db) if n not in master_names]
        acc = exact = near = 0
        vs_master = vs_pending = 0
        examples = []
        t1 = time.time()
        for j, (nm, code) in enumerate(pend, 1):
            try:
                ok, fp = gate.check(code)
            except Exception:
                # An untraceable candidate is admitted by the gate's own contract; counting
                # it as a duplicate would understate distinctness on a technicality.
                ok, fp = True, (None, None)
            m = gate.last_match
            if ok:
                gate.add(nm, fp)
                origin[nm] = v
                acc += 1
            else:
                if m and m.get("kind") == "exact-clone":
                    exact += 1
                else:
                    near += 1
                key = (m or {}).get("key")
                if key is None or key in master_names:
                    vs_master += 1
                else:
                    vs_pending += 1
                    if len(examples) < 6:
                        examples.append(
                            f"{nm} ~ {key} [{origin.get(key,'?')}] "
                            f"{(m or {}).get('cos',0):.3f}")
            if j % 100 == 0:
                print(f"  {v}: {j}/{len(pend)}  acc={acc}  {time.time()-t1:.0f}s",
                      flush=True)
        print(f"\n  {v:8} pending {len(pend):4}  DISTINCT {acc:4}  "
              f"exact-dup {exact:3}  near-dup {near:3}   "
              f"(vs master {vs_master}, vs earlier variant {vs_pending})")
        for e in examples:
            print(f"       collision: {e}")
        res["variants"][v] = {"pending": len(pend), "distinct": acc,
                              "exact_dup": exact, "near_dup": near,
                              "vs_master": vs_master, "vs_pending": vs_pending}
        rows.append((v, len(pend), acc, exact, near, vs_master, vs_pending))

    tp = sum(r[1] for r in rows)
    ta = sum(r[2] for r in rows)
    print(f"\n{'='*70}")
    print(f"  {'variant':<10}{'pending':>9}{'distinct':>10}{'exact':>8}{'near':>7}"
          f"{'vs master':>11}{'vs peer':>9}")
    for v, p, ac, e, n, vm, vp in rows:
        print(f"  {v:<10}{p:>9}{ac:>10}{e:>8}{n:>7}{vm:>11}{vp:>9}")
    print(f"  {'TOTAL':<10}{tp:>9}{ta:>10}"
          f"{sum(r[3] for r in rows):>8}{sum(r[4] for r in rows):>7}"
          f"{sum(r[5] for r in rows):>11}{sum(r[6] for r in rows):>9}")
    print(f"\n  of {tp} unmerged models, {ta} are architecturally new "
          f"({ta/max(tp,1):.1%}); master would go {len(master)} -> {len(master)+ta}")
    res["total"] = {"pending": tp, "distinct": ta,
                    "master_after": len(master) + ta}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

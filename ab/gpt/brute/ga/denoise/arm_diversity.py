"""Which generator actually supplies the architectural variety? A cumulative ablation.

THE QUESTION. The corpus collapses to 84 clusters at cosine 0.95 despite holding ~1,550
models, and 779 freshly trained GA models added only 29 architectures. If the variety is
carried by the 36 hand-written families rather than by the 1,284 generated descendants, then
adding families is the only lever that moves it, and more GA compute is not.

HOW IT IS MEASURED. Greedy leader clustering is order-dependent -- whichever arm goes first
claims every shared cluster -- so a single ordering cannot separate "this arm is diverse" from
"this arm went first". Every arm therefore gets measured three ways:

  ALONE       cluster the arm by itself: its intrinsic spread.
  FIRST       cluster it first, then the others: an upper bound on its contribution.
  LAST        cluster it after everything else: its MARGINAL contribution -- how many
              clusters exist only because this arm exists. This is the number that decides
              where the next model should come from.

An arm with a large ALONE and a near-zero LAST is redundant: it re-derives variety the others
already had. An arm with a small ALONE and a large LAST is a genuine source, however few
models it has.
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import naming  # noqa: E402  (needs HERE on sys.path first)


def arm_of(n):
    # 'LLM-' plus the legacy 'LLM_' spelling (470 DB rows predate the rename),
    # plus the bare g1_/g2_ candidate-id names from the earliest imports.
    if naming.matches_family(n, "LLM-") or n.startswith(("g1_", "g2_")) \
            or naming.arm_of(n):
        return "LLM"
    if n.startswith(("DenoiseGA-", "DnSeedGA-", "DenoiseSkip-", "DnSkip-",
                     "DnNewFam-", "DnSearch-")):
        return "GA"
    return "hand"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--task", default="img-denoising")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/arm_diversity.json")
    a = ap.parse_args()

    import numpy as np
    from diversity_gate import DiversityGate

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code FROM nn n WHERE n.name IN "
        "(SELECT DISTINCT nn FROM stat WHERE task=?) ORDER BY n.name", (a.task,)).fetchall()
    con.close()

    by = collections.defaultdict(list)
    for n, c in rows:
        by[arm_of(n)].append((n, c))
    print(f"=== {len(rows)} {a.task} models ===")
    for k in ("hand", "GA", "LLM"):
        print(f"  {k:5} {len(by[k]):5}")

    # FINGERPRINT ONCE. The gate recomputes an AST parse and an fx trace per check, which is
    # ~4 models/sec; six orderings x three thresholds would repeat that 18 times for no reason.
    print("\nfingerprinting once ...", flush=True)
    g = DiversityGate(store_path=None, graded=True, cos_max=0.95,
                      ast_exact=True, log=lambda *_: None)
    fp = {}
    for i, (n, c) in enumerate(rows, 1):
        try:
            fp[n] = g._fingerprint(c)
        except Exception:
            fp[n] = (None, None)
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    ok = sum(1 for v in fp.values() if v[0] is not None)
    vec = sum(1 for v in fp.values() if v[1] is not None)
    print(f"  hashed {ok}/{len(rows)}, vectors {vec}\n")

    def cluster(order, t):
        """-> (n_clusters, {arm: clusters this arm OPENED}). Mirrors the live gate."""
        seen, reps, opened = set(), [], collections.Counter()
        R = np.empty((len(order), 4128 if not vec else
                      len(next(v[1] for v in fp.values() if v[1] is not None))),
                     dtype=np.float32)
        k = 0
        for n in order:
            h, e = fp.get(n, (None, None))
            if h is not None and h in seen:
                continue
            if e is not None and k and float((R[:k] @ e).max()) >= t:
                continue
            reps.append(n)
            opened[arm_of(n)] += 1
            if h is not None:
                seen.add(h)
            if e is not None:
                R[k] = e
                k += 1
        return len(reps), opened

    names = {k: [n for n, _ in v] for k, v in by.items()}
    res = {"counts": {k: len(v) for k, v in by.items()}, "t": {}}

    for t in a.thresholds:
        print(f"{'='*64}\n=== cosine {t} ===")
        alone = {k: cluster(names[k], t)[0] for k in ("hand", "GA", "LLM")}
        total, _ = cluster([n for n, _ in rows], t)
        print(f"  {'arm':6}{'models':>8}{'ALONE':>8}{'FIRST':>8}{'LAST':>8}"
              f"{'marginal %':>12}")
        row = {}
        for k in ("hand", "GA", "LLM"):
            others = [n for j in ("hand", "GA", "LLM") if j != k for n in names[j]]
            _, of = cluster(names[k] + others, t)
            first = of[k]
            _, ol = cluster(others + names[k], t)
            last = ol[k]
            print(f"  {k:6}{len(names[k]):>8}{alone[k]:>8}{first:>8}{last:>8}"
                  f"{last/max(total,1):>11.1%}")
            row[k] = {"models": len(names[k]), "alone": alone[k],
                      "first": first, "last": last}
        print(f"  {'ALL':6}{len(rows):>8}{'':>8}{'':>8}{total:>8}")
        # Cost of variety, per arm, in models trained per cluster it uniquely contributes.
        print(f"\n  models trained per MARGINAL cluster:")
        for k in ("hand", "GA", "LLM"):
            m, l = row[k]["models"], row[k]["last"]
            print(f"    {k:6} {m/l:>8.1f}" if l else f"    {k:6} {'inf':>8}")
        res["t"][str(t)] = {"total": total, "arms": row}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

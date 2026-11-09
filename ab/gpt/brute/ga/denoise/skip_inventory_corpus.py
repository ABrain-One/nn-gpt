"""Corpus-wide skip-connection inventory -- the SAMPLING FRAME for the skip ablation.

skip_inventory.py analyses one .py file at a time against the hand-written DenoiseUnet
seeds. The ablation needs the same classification over the whole corpus, and 80 of the
models we care about exist only as DB rows with no file on disk, so this drives the same
detectors from `nn.code` instead of from paths.

WHAT IT IS FOR. The ablation is ADDITION-ONLY (skip_operators.py has add_global_residual /
add_block_residual / add_lateral_skip and no removal counterpart), so a valid base is a
model that LACKS the skip type under test. This script finds those bases and reports how
many exist per cell, which is the difference between a factorial design that is populated
and one that silently collapses to a few models.

It also replaces the project's only quantitative skip claim -- "+3.41 dB", from a single
uncontrolled observation that DenoiseUnet2F lacked the global residual and gained one --
with an observational baseline over thousands of models. That baseline is NOT the answer
either (it is confounded; see PLAN_three_tasks.md), but it sets the expectation the
controlled experiment has to beat.

  python skip_inventory_corpus.py --out skip_inventory_corpus.json
"""
import argparse
import ast
import collections
import json
import os
import sqlite3
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skip_inventory as SI                                    # noqa: E402

SCALE = 48.0


def classify(src):
    """-> dict of skip-type booleans, or None when the source will not parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    try:
        return {
            "GLOBAL": bool(SI._has_global_residual(tree)),
            "LATERAL": bool(SI._has_lateral(src)),
            "BLOCKRES": bool(SI._has_block_residual(tree)),
            "DENSE": bool(SI._has_dense(src)),
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db")
    ap.add_argument("--out", default=os.path.join(HERE, "skip_inventory_corpus.json"))
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    best = {}
    for nn, acc in c.execute("select nn,accuracy from stat where epoch=5"):
        best[nn] = max(best.get(nn, 0.0), acc)

    rows, unparsed = {}, 0
    for name, code in c.execute("select name,code from nn"):
        sig = classify(code)
        if sig is None:
            unparsed += 1
            continue
        sig["ep5_dB"] = round(best[name] * SCALE, 3) if name in best else None
        rows[name] = sig
    print(f"classified {len(rows)} models ({unparsed} unparsed)")

    # how many models sit in each of the 8 GLOBAL x LATERAL x BLOCKRES cells
    cells = collections.Counter(
        (r["GLOBAL"], r["LATERAL"], r["BLOCKRES"]) for r in rows.values())
    print("\ncell population (GLOBAL, LATERAL, BLOCKRES):")
    for k in sorted(cells, key=lambda k: -cells[k]):
        scored = [r["ep5_dB"] for r in rows.values()
                  if (r["GLOBAL"], r["LATERAL"], r["BLOCKRES"]) == k and r["ep5_dB"]]
        med = f"{st.median(scored):.2f}" if scored else "-"
        print(f"  G={int(k[0])} L={int(k[1])} B={int(k[2])} : {cells[k]:5d} models  median {med} dB")

    # observational per-type gap -- REPORTED WITH ITS CONFOUND, not as a result
    print("\nobservational gaps (CONFOUNDED -- capacity and family vary with skip presence):")
    for t in ("GLOBAL", "LATERAL", "BLOCKRES", "DENSE"):
        w = [r["ep5_dB"] for r in rows.values() if r[t] and r["ep5_dB"]]
        wo = [r["ep5_dB"] for r in rows.values() if not r[t] and r["ep5_dB"]]
        if w and wo:
            print(f"  {t:9s} with n={len(w):5d} median {st.median(w):.2f} | "
                  f"without n={len(wo):5d} median {st.median(wo):.2f} | "
                  f"median gap {st.median(w)-st.median(wo):+.2f} dB")

    # bases available for an addition-only ablation: must LACK the type under test
    print("\nusable ablation bases (lack the type, so it can be ADDED):")
    for t in ("GLOBAL", "BLOCKRES", "LATERAL"):
        cand = [n for n, r in rows.items() if not r[t] and (r["ep5_dB"] or 0) >= 32.0]
        print(f"  lacking {t:9s} and >=32 dB: {len(cand)}")

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

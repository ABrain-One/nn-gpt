"""Measure MACs for every hand-written seed, and rank the arms on a defensible basis.

WHY THIS EXISTS. The headline comparison "does the GA beat hand design" was being made
against numbers that are not comparable in two separate ways, and I made both mistakes in one
message:

  MIXED EPOCH BUDGETS. DenoiseSubPixel's 35.246 dB is a 50-EPOCH result from the canonical
  sweep's per-model DBs. GA and LLM models train 5 epochs. Comparing them ranks training
  budget, not architecture.

  ILLEGAL BASELINES. The best hand-written model at 5 epochs is DenoiseUnet7 at 35.150 dB --
  but DenoiseUnet7F measures 30152 MMac, 3.3x over the 9053 cap, and the plain variants are
  the FULL-WIDTH originals, so it is larger still. A baseline that could never ship is not a
  baseline. This is the same error that produced the DenoiseUNet2 retraction: a model written
  up as an edge-legal seed until someone measured it.

Of the 33 hand-written seeds only 4 had ever been measured. This measures all of them with
the GA's own counter (edge_cost._mac_hooks), so "best legal baseline at N epochs" stops being
an assumption.

  python seed_mac_audit.py --names /a/mm/sweep_out/seed_names.txt --cap 9053
"""
import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def macs_of(src, hw=256):
    """(MMac, Mparams) on CPU via the GA's own counter, or raises."""
    import torch
    from edge_cost import _mac_hooks
    ns = {}
    exec(compile(src, "<seed>", "exec"), ns)
    net = ns["Net"]((1, 3, hw, hw), (3,), dict(PRM), "cpu")
    net.eval()
    handles, macs = _mac_hooks(net)
    try:
        with torch.no_grad():
            net(torch.rand(1, 3, hw, hw))
    finally:
        for h in handles:
            h.remove()
    return macs["v"] / 1e6, sum(p.numel() for p in net.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="/a/mm/sweep_out/seed_names.txt")
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cap", type=float, default=9053.0)
    ap.add_argument("--epoch", type=int, default=5, help="epoch to compare all arms at")
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/seed_macs.json")
    a = ap.parse_args()

    names = [l.strip() for l in open(a.names) if l.strip()]
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    def db_at(nm):
        r = c.execute("SELECT MAX(accuracy)*48 FROM stat WHERE nn=? AND epoch=?",
                      (nm, a.epoch)).fetchone()
        return r[0] if r and r[0] is not None else None

    rows = []
    print(f"=== MAC audit of {len(names)} hand-written seeds | cap {a.cap:.0f} MMac "
          f"| dB at epoch {a.epoch} ===\n")
    print(f"{'seed':<24} {'MMac':>9} {'Mprm':>7}  {'dB@%d' % a.epoch:>8}  verdict")
    for nm in names:
        p = os.path.join(NN_DIR, nm + ".py")
        d = db_at(nm)
        if not os.path.exists(p):
            print(f"{nm:<24} {'-':>9} {'-':>7}  {d if d is None else round(d,3):>8}  "
                  f"NO SOURCE (in DB only)")
            rows.append({"name": nm, "mmac": None, "db": d, "legal": None,
                         "note": "source file absent"})
            continue
        try:
            mm, mp = macs_of(open(p).read())
        except Exception as e:
            print(f"{nm:<24} {'?':>9} {'?':>7}  {d if d is None else round(d,3):>8}  "
                  f"UNMEASURABLE ({type(e).__name__})")
            rows.append({"name": nm, "mmac": None, "db": d, "legal": False,
                         "note": f"{type(e).__name__}: {e}"[:120]})
            continue
        legal = mm <= a.cap
        print(f"{nm:<24} {mm:>9.0f} {mp:>7.2f}  "
              f"{('%8.3f' % d) if d is not None else '       -'}  "
              f"{'legal' if legal else 'OVER by %.0f' % (mm - a.cap)}")
        rows.append({"name": nm, "mmac": round(mm, 1), "mparams": round(mp, 3),
                     "db": d, "legal": legal})

    legal = [r for r in rows if r.get("legal") and r.get("db") is not None]
    over = [r for r in rows if r.get("legal") is False]
    print(f"\n  legal {len(legal)}   over/unmeasurable {len(over)}   "
          f"of {len(rows)}")

    print(f"\n=== BEST LEGAL HAND-WRITTEN BASELINE at epoch {a.epoch} ===")
    for r in sorted(legal, key=lambda x: -x["db"])[:5]:
        print(f"  {r['db']:7.3f} dB  {r['name']:<24} {r['mmac']:>8.0f} MMac")

    # The generated arms, on the SAME epoch and the same legality rule.
    for pat, lab in (("DenoiseGA-%", "GA"), ("LLM_%", "LLM")):
        r = c.execute("SELECT nn, MAX(accuracy)*48 FROM stat WHERE nn LIKE ? AND epoch=?",
                      (pat, a.epoch)).fetchone()
        if r and r[1] is not None:
            print(f"  {r[1]:7.3f} dB  {lab:<24} (best {lab} at epoch {a.epoch})")

    print("\nNOTE: GA/LLM models are MAC-gated at generation time, so their best is legal by "
          "construction. Hand-written seeds were not, which is what this audit fixes.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

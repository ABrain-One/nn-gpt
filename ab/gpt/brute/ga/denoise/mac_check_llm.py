"""Measure MACs for LLM-produced models. CPU only.

WHY THIS EXISTS. The GA enforces the 9053 MMac edge budget at two points (--seed-mac-cap on
its seeds, --edge-filter/--mac-cap on its children). The LLM path enforces it at NONE:
`screen_and_train.py` contains no MAC check at all, and `elite_archive.py` prints
"MAC CAP NOT ENFORCED: cost backend unimportable" because `ab.gpt` is shadowed by
nn-dataset's `ab` package under the pod-local LEMUR root.

So every LLM model trained so far entered the DB, and the elite pool, WITHOUT anyone
checking whether it fits the edge budget the whole project is defined by. The prompt states
the limit, but a prompt is a request, not a constraint -- exactly the gap that let
DenoiseUNet2 (54618 MMac, ~6x over) be written up as an edge-legal seed until the GA
measured it.

This measures them. Source-level reading cannot: MACs depend on channel widths and spatial
resolution through the whole graph.

  python mac_check_llm.py --stage2-dirs /a/mm/sweep_out/llm_stage2 --cap 9053
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def macs_of(src, hw=256):
    """(MMac, Mparams) on CPU, or raises.

    Uses edge_cost._mac_hooks -- the GA's OWN counter -- rather than thop/fvcore. Two
    reasons: neither profiler is installed in this image (the first attempt reported all 9
    models UNMEASURABLE), and more importantly a different counter would produce numbers not
    comparable to the GA's, which is the whole point of measuring against the same 9053 cap.
    It hooks Conv2d/ConvTranspose2d/Linear and ignores norms, activations and element-wise
    ops as negligible.

    The model is instantiated by exec here rather than through mutator.lemur_io, so this
    runs CPU-only without the LEMUR root or the DB stack.
    """
    import torch
    from edge_cost import _mac_hooks
    ns = {}
    exec(compile(src, "<cand>", "exec"), ns)
    net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
    net.eval()
    handles, macs = _mac_hooks(net)
    try:
        with torch.no_grad():
            net(torch.rand(1, 3, hw, hw))
    finally:
        for h in handles:
            h.remove()
    n = sum(p.numel() for p in net.parameters())
    return macs["v"] / 1e6, n / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dirs", nargs="+",
                    default=["/a/mm/sweep_out/llm_stage2"])
    ap.add_argument("--run-dirs", nargs="+",
                    default=["/a/mm/sweep_out/llm_gen2/cold/run",
                             "/a/mm/sweep_out/llm_gen2/seeded/run"])
    ap.add_argument("--cap", type=float, default=9053.0)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/llm_macs.json")
    a = ap.parse_args()

    src = {}
    for r in a.run_dirs:
        for p in glob.glob(os.path.join(r, "cand", "*", "candidate.py")):
            src[os.path.basename(os.path.dirname(p))] = p

    ids = []
    for sd in a.stage2_dirs:
        for d in sorted(glob.glob(os.path.join(sd, "full", "*"))):
            cid = os.path.basename(d)
            if cid in src:
                ids.append(cid)

    print(f"=== MAC audit of {len(ids)} trained LLM models | cap {a.cap:.0f} MMac ===\n")
    print(f"{'candidate':<34} {'MMac':>10} {'Mparams':>9}  verdict")
    rows, over, unknown = [], 0, 0
    for cid in ids:
        try:
            m, p = macs_of(open(src[cid]).read())
        except Exception as e:
            m = p = None
            print(f"{cid:<34} {'?':>10} {'?':>9}  UNMEASURABLE ({type(e).__name__})")
            unknown += 1
        if m is not None:
            v = "OK" if m <= a.cap else f"OVER by {m - a.cap:.0f}"
            if m > a.cap:
                over += 1
            print(f"{cid:<34} {m:>10.1f} {p:>9.2f}  {v}")
        rows.append({"id": cid, "mmac": m, "mparams": p,
                     "legal": (m is not None and m <= a.cap)})

    print(f"\n  within cap   {sum(1 for r in rows if r['legal'])}")
    print(f"  OVER cap     {over}")
    print(f"  unmeasurable {unknown}   (treated as NOT verified, never as legal)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    # Non-zero exit when anything is over or unverified, so a pipeline step can gate on it.
    return 1 if (over or unknown) else 0


if __name__ == "__main__":
    raise SystemExit(main())

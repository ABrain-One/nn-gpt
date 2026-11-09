"""What ACTUALLY breaks the LLM's architectures at the first forward pass.

The stored preflight `detail` strings were classified by regex and suggested ~47% of
forward failures were simple channel mismatches. Re-running the candidates says otherwise:
two thirds of the live errors never name both the declared and the received shape, so a
channel repair has nothing to solve for (measured: repair v2 recovered 1 of 120).

This re-executes each reject and buckets the REAL exception, so the next fix is chosen
against evidence instead of against a string match. CPU only, read-only.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def classify(err):
    e = err or ""
    if "expected weight to be at least 1" in e or re.search(r"weight of size \[0[,\]]", e):
        return "zero-sized layer (channel arithmetic collapsed to 0)"
    if "Given groups=1" in e and "expected input" in e:
        return "conv in_channels vs actual input (repairable shape)"
    if "Given groups=1" in e:
        return "conv weight invalid, input not named"
    if "Sizes of tensors must match except in dimension" in e:
        return "concat: non-channel dim mismatch (spatial)"
    if "must match the size of tensor" in e:
        return "elementwise add/mul: shape mismatch"
    if "has no attribute" in e:
        return "missing attribute (module never created)"
    if "shape '" in e and "is invalid for input of size" in e:
        return "view/reshape with wrong element count"
    if "Output size is too small" in e or "Calculated padded input size" in e:
        return "downsampled too far (spatial size hit zero)"
    if "out of memory" in e.lower():
        return "OOM"
    if e.startswith("shape "):
        return "output shape != input shape (contract violation)"
    return "other: " + e.split(":")[0][:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run3", default="/a/mm/sweep_out/run3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/a/mm/sweep_out/ablation/forward_census.json")
    a = ap.parse_args()
    import torch

    todo = []
    for k in range(1, 21):
        for f in glob.glob(f"{a.run3}/cycle{k}/pre/preflight.json"):
            for r in json.load(open(f)):
                if r.get("ok") or r.get("stage") != "forward":
                    continue
                p = f"{a.run3}/cycle{k}/run/cand/{r['id']}/candidate.py"
                if os.path.exists(p):
                    todo.append((r["id"], p))
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} forward-stage rejects to re-execute", flush=True)

    cnt = collections.Counter()
    samples = {}
    for i, (cid, path) in enumerate(todo, 1):
        src = open(path).read()
        err = None
        try:
            ns = {}
            exec(compile(src, "<f>", "exec"), ns)
            net = ns["Net"](in_shape=(1, 3, 64, 64), prm={"lr": 1e-4}, device="cpu")
            net.eval()
            with torch.no_grad():
                y = net(torch.randn(2, 3, 64, 64))
            if tuple(y.shape) != (2, 3, 64, 64):
                err = f"shape {tuple(y.shape)}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        k = "PASSES NOW" if err is None else classify(err)
        cnt[k] += 1
        samples.setdefault(k, (err or "")[:160])
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}]", flush=True)

    tot = sum(cnt.values())
    print(f"\n=== FORWARD-FAILURE CENSUS ({tot} candidates re-executed) ===")
    for k, v in cnt.most_common():
        print(f"  {v:5d} ({100*v/tot:5.1f}%)  {k}")
        if samples.get(k):
            print(f"           e.g. {samples[k][:110]}")
    json.dump({"counts": dict(cnt), "samples": samples}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

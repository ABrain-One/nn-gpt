"""Measure channel-repair v1 vs v2 on run 3's real forward-stage rejects.

Answers one question with evidence rather than argument: how many of the 434 candidates
that failed their first forward pass are recoverable, and how many does each repair
actually recover? Runs the repaired source to a real forward pass -- a patch that parses
but still crashes is not a repair.

CPU only, no GPU, no DB writes, no changes to preflight.py. Read-only over run 3's outputs.
"""
import argparse
import glob
import json
import os
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build_model(src, size=64):
    ns = {}
    exec(compile(src, "<repair>", "exec"), ns)
    if "Net" not in ns:
        raise RuntimeError("no Net class")
    return ns["Net"](in_shape=(1, 3, size, size), prm={"lr": 1e-4}, device="cpu")


def forward_ok(src, size=64):
    """-> (True, None) or (False, error string)."""
    import torch
    try:
        net = build_model(src, size)
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 3, size, size))
        if tuple(y.shape) != (2, 3, size, size):
            return False, f"shape {tuple(y.shape)}"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run3", default="/a/mm/sweep_out/run3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=6,
                    help="repair attempts per candidate; each fixes ONE conv, so a model "
                         "with several bad skips needs several passes")
    ap.add_argument("--out", default="/a/mm/sweep_out/ablation/chan_repair_eval.json")
    a = ap.parse_args()

    import preflight as PF
    import chan_repair2 as R2

    # collect the forward-stage rejects
    todo = []
    for k in range(1, 21):
        for f in glob.glob(f"{a.run3}/cycle{k}/pre/preflight.json"):
            for r in json.load(open(f)):
                if r.get("ok") or r.get("stage") != "forward":
                    continue
                p = f"{a.run3}/cycle{k}/run/cand/{r['id']}/candidate.py"
                # .orig is written when a repair already rewrote the file; start from it
                p_orig = p + ".orig"
                src_path = p_orig if os.path.exists(p_orig) else p
                if os.path.exists(src_path):
                    todo.append((r["id"], src_path, r.get("detail") or ""))
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} forward-stage rejects with source on disk", flush=True)

    stats = collections.Counter()
    reasons = collections.Counter()
    records = []
    for i, (cid, path, detail) in enumerate(todo, 1):
        src0 = open(path).read()
        ok0, err0 = forward_ok(src0)
        if ok0:
            stats["already_ok"] += 1          # passes now (env differs from run time)
            continue

        # ---- v1: the shipped regex repair, same budget the pipeline uses ----
        s1, err1, n1 = src0, err0, 0
        for _ in range(a.max_iters):
            nxt, note = PF.repair_channels(s1, err1 or "", 1)
            if nxt is None:
                break
            s1 = nxt
            n1 += 1
            ok1, err1 = forward_ok(s1)
            if ok1:
                break
        v1_ok = forward_ok(s1)[0] if n1 else False

        # ---- v2: runtime shape lookup + AST patch ----
        s2, err2, n2, why = src0, err0, 0, None
        for _ in range(a.max_iters):
            nxt, note = R2.repair(s2, err2 or "", build_model)
            if nxt is None:
                why = note
                break
            s2 = nxt
            n2 += 1
            ok2, err2 = forward_ok(s2)
            if ok2:
                break
        v2_ok = forward_ok(s2)[0] if n2 else False

        stats["total"] += 1
        stats["v1_fired"] += 1 if n1 else 0
        stats["v1_fixed"] += 1 if v1_ok else 0
        stats["v2_fired"] += 1 if n2 else 0
        stats["v2_fixed"] += 1 if v2_ok else 0
        if not v2_ok and why:
            reasons[why.split(":")[0][:48]] += 1
        records.append({"id": cid, "v1_iters": n1, "v1_ok": v1_ok,
                        "v2_iters": n2, "v2_ok": v2_ok, "v2_reason": why})
        if i % 25 == 0:
            print(f"  [{i}/{len(todo)}] v1 fixed {stats['v1_fixed']}  "
                  f"v2 fixed {stats['v2_fixed']}", flush=True)

    print("\n=== CHANNEL REPAIR: v1 (shipped regex) vs v2 (runtime + AST) ===")
    t = max(stats["total"], 1)
    print(f"  candidates evaluated : {stats['total']}  (already passing: {stats['already_ok']})")
    print(f"  v1 fired / fixed     : {stats['v1_fired']} / {stats['v1_fixed']}  "
          f"({100*stats['v1_fixed']/t:.1f}%)")
    print(f"  v2 fired / fixed     : {stats['v2_fired']} / {stats['v2_fixed']}  "
          f"({100*stats['v2_fixed']/t:.1f}%)")
    print("\n  why v2 still declines:")
    for r, n in reasons.most_common(8):
        print(f"    {n:4d}  {r}")
    json.dump({"stats": dict(stats), "reasons": dict(reasons), "records": records},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

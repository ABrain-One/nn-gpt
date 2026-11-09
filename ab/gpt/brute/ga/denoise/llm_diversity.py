"""WL exact-hash + graded-cosine diversity census over the LLM-generated candidates.

Uses the SAME machinery the GA uses to reject clones -- arch_embed.both_source and
DiversityGate -- so "how diverse is the LLM output" is answered on the identical scale as
"how diverse is the GA output", rather than by a new ad-hoc metric.

TWO OPERATING POINTS, and they answer different questions:

  EXACT WL graph hash    Two models collide only if their traced graphs are isomorphic with
                         identical per-node op labels. Width changes, channel counts and
                         variable renaming DO shift the label set, so this is a strict
                         literal-clone test. This is what the GA runs by default.

  GRADED cosine >= 0.95  Continuous-WL embedding similarity. Calibrated in metric_ablation:
                         trivial variation sits >= 0.985, genuinely distinct architectures
                         <= 0.921, so 0.95 separates them with margin. NEVER used in any run
                         so far -- the GA's launchers leave graded off. Reported here because
                         for a GENERATOR that can leave the family (an LLM, unlike a mutation
                         operator) it is the meaningful near-duplicate test: the exact hash
                         will happily pass 400 minor rewrites of one U-Net.

The greedy-gate simulation replays DiversityGate in candidate order, which is what the GA
would actually have done, so the survivor count is directly comparable to a GA run's.

Order matters for greedy acceptance, so the arm-by-arm figures are computed on each arm
ALONE (not by filtering a pooled pass) -- otherwise whichever arm ran first would look more
diverse purely for being first.

  python llm_diversity.py --run-dirs .../cold/run .../seeded/run --out /a/mm/sweep_out/llm_div
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def arm_of(cid):
    return "cold" if "_cold_" in cid else ("seeded" if "_seeded_" in cid else "?")


def strat_of(cid):
    parts = cid.split("_", 2)
    return parts[2] if len(parts) > 2 else "?"


def fingerprint_all(cands, log=print):
    """[(cid, hash, embed_or_None)] -- one trace per candidate."""
    import arch_embed
    out, t0 = [], time.time()
    for i, (cid, path) in enumerate(cands, 1):
        code = open(path).read()
        try:
            h, e = arch_embed.both_source(code)
        except Exception as ex:
            h, e = None, None
            if i <= 5:
                log(f"    trace error {cid}: {type(ex).__name__}: {ex}"[:160])
        out.append((cid, h, e))
        if i % 25 == 0 or i == len(cands):
            ok = sum(1 for _, hh, _ in out if hh)
            log(f"  [{i}/{len(cands)}] traced ok={ok} ({time.time()-t0:.0f}s)")
    return out


def greedy_gate(fps, graded, cos_max=0.95):
    """Replay DiversityGate in order; return (accepted_ids, rejected_ids)."""
    from diversity_gate import DiversityGate
    g = DiversityGate(graded=graded, cos_max=cos_max, log=lambda *a, **k: None)
    acc, rej = [], []
    for cid, h, e in fps:
        if h is None:
            acc.append(cid)          # untraceable -> gate lets it through, as in the GA
            continue
        if _check(g, h, e):
            g.add(cid, (h, e))
            acc.append(cid)
        else:
            rej.append(cid)
    return acc, rej


def _check(g, h, e):
    """DiversityGate.check() takes source; we already have the fingerprint, so replicate
    its decision rule directly rather than re-tracing all 516 models a second time."""
    import numpy as np
    if h in g.accepted:
        return False
    if g.graded and e is not None and g.embeds:
        if max(float(np.dot(e, ae)) for ae in g.embeds) >= g.cos_max:
            return False
    return True


def report(fps, label, log=print):
    n = len(fps)
    traced = [(c, h, e) for c, h, e in fps if h is not None]
    log(f"\n=== {label} | {n} candidates, {len(traced)} traceable "
        f"({len(traced)/n:.1%}) ===")
    if not traced:
        return
    hashes = Counter(h for _, h, _ in traced)
    uniq = len(hashes)
    log(f"  unique WL exact graphs      {uniq}  of {len(traced)} traced  "
        f"({uniq/len(traced):.1%})")
    dup = {h: c for h, c in hashes.items() if c > 1}
    log(f"  colliding graph groups      {len(dup)}  covering "
        f"{sum(dup.values())} candidates")
    if dup:
        top = sorted(dup.items(), key=lambda kv: -kv[1])[:5]
        for h, c in top:
            ids = [cid for cid, hh, _ in traced if hh == h][:4]
            log(f"    x{c:<4d} {h[:16]}  e.g. {', '.join(ids)}")
    return traced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="/a/mm/sweep_out/llm_div")
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cands = []
    for r in a.run_dirs:
        for p in sorted(glob.glob(os.path.join(r, "cand", "*", "candidate.py"))):
            cands.append((os.path.basename(os.path.dirname(p)), p))
    cands.sort()
    if a.limit:
        cands = cands[:a.limit]
    print(f"=== LLM diversity census | {len(cands)} candidates ===", flush=True)

    fps = fingerprint_all(cands)

    # persist first -- tracing is the expensive part and must never need repeating
    with open(os.path.join(a.out, "fingerprints.json"), "w") as f:
        json.dump([{"id": c, "hash": h, "has_embed": e is not None}
                   for c, h, e in fps], f, indent=2)

    report(fps, "ALL (cold + seeded)")
    per_arm = defaultdict(list)
    for c, h, e in fps:
        per_arm[arm_of(c)].append((c, h, e))
    for arm in ("cold", "seeded"):
        if per_arm[arm]:
            report(per_arm[arm], f"arm = {arm}")

    print("\n=== greedy DiversityGate replay (what the GA would have kept) ===")
    for graded in (False, True):
        mode = f"graded cos>={a.cos_max}" if graded else "exact hash only"
        acc, rej = greedy_gate(fps, graded, a.cos_max)
        print(f"  {mode:<26} accept {len(acc):4d}  reject {len(rej):4d}  "
              f"({len(acc)/len(fps):.1%} kept)")
        for arm in ("cold", "seeded"):
            sub = per_arm[arm]
            if not sub:
                continue
            aa, rr = greedy_gate(sub, graded, a.cos_max)
            print(f"      {arm:<8} alone      accept {len(aa):4d}  reject {len(rr):4d}  "
                  f"({len(aa)/len(sub):.1%} kept)")
        with open(os.path.join(a.out,
                  f"gate_{'graded' if graded else 'exact'}.json"), "w") as f:
            json.dump({"accepted": acc, "rejected": rej}, f, indent=2)

    print("\n=== unique graphs per strategy ===")
    by_s = defaultdict(list)
    for c, h, e in fps:
        by_s[strat_of(c)].append(h)
    for s, hs in sorted(by_s.items()):
        t = [h for h in hs if h]
        u = len(set(t))
        print(f"  {s:<24} {u:4d} unique / {len(t):4d} traced / {len(hs):4d} total")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())

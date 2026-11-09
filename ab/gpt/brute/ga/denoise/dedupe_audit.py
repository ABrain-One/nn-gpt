"""Measure how much of the corpus's distinct-model count survives the GRADED diversity check.

WHY THIS EXISTS. The graded half of DiversityGate builds every candidate through
mutator/lemur_io.py:169, which is unconditionally

    params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

and a probe pod on an ordinary worker (w9-mi, zero GPUs requested) reports
NVIDIA_VISIBLE_DEVICES=all with 4 visible cards -- so preflight has been constructing
candidates on GPUs it never requested, on every node, not just the DGX. When such a card is
contended the trace raises OutOfMemoryError, which diversity_gate.py CATCHES and degrades to
"untraceable for the graded check; exact check still applies". The gate then falls back to
arch_uid2 alone and admits near-duplicates as novel. It never crashes and never warns twice.

The historical preflight stdout is gone (pod logs expired; the one surviving host-side log was
truncated by a `tee` on relaunch), so WHICH runs degraded cannot be reconstructed. This script
sidesteps provenance entirely: it re-runs both halves of the gate over the corpus as it stands
today and reports the gap between them. That gap is the answer regardless of how it got there.

ORDER MATTERS AND IS PRESERVED. DiversityGate is greedy -- it accepts a candidate if it is far
enough from everything ALREADY accepted, so the surviving set depends on insertion order. Rows
are processed by ascending nn.id, which is the order they entered the DB, so the greedy replay
approximates what a correctly-functioning gate would have kept at the time.

  CUDA_VISIBLE_DEVICES= python dedupe_audit.py --db db/ab.nn.db --cos-max 0.90
"""
import argparse
import json
import os
import sqlite3
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cos-max", type=float, default=0.90,
                    help="the gate's threshold; pairs above this are near-duplicates")
    ap.add_argument("--out", default="/a/mm/sweep_out/dedupe_audit.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows (debug aid)")
    a = ap.parse_args()

    # Assert the CPU claim rather than trusting it. This whole script exists because a
    # "CPU only" banner was printed by a process that was in fact allocating on a contended
    # GPU; repeating that mistake here would silently corrupt the audit of that mistake.
    import torch
    assert not torch.cuda.is_available(), (
        "CUDA is visible -- rerun with CUDA_VISIBLE_DEVICES= . The graded trace would "
        "otherwise allocate on a shared card and degrade exactly as the bug under audit did.")
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import arch_embed
    from arch_uid2 import arch_uid

    con = sqlite3.connect(a.db)
    q = "SELECT id, name, code FROM nn ORDER BY id"
    if a.limit:
        q += f" LIMIT {a.limit}"
    rows = con.execute(q).fetchall()
    print(f"=== dedupe audit | {len(rows)} rows | cos_max={a.cos_max} | CPU ===", flush=True)

    def fam(n):
        for p in ("LLM_", "DenoiseGA-", "DnSeedGA-", "DenoiseSkip-"):
            if n.startswith(p):
                return p
        return "other"

    exact_seen = {}          # arch_uid2 -> first name holding it
    kept_vecs = []           # embeddings of the greedily-accepted set
    kept_names = []
    exact_dupes, graded_dupes, untraceable = [], [], []
    t0 = time.time()

    for i, (rid, name, code) in enumerate(rows):
        if i % 100 == 0 and i:
            print(f"  [{i}/{len(rows)}] exact_kept={len(exact_seen)} "
                  f"graded_kept={len(kept_names)} untraceable={len(untraceable)} "
                  f"{time.time()-t0:.0f}s", flush=True)
        try:
            uid = arch_uid(code)
        except Exception:
            uid = None
        if uid is not None:
            if uid in exact_seen:
                exact_dupes.append({"name": name, "dup_of": exact_seen[uid], "fam": fam(name)})
                continue                      # the exact half already rejects these
            exact_seen[uid] = name

        try:
            _h, emb = arch_embed.both_source(code)
        except Exception:
            emb = None
        if emb is None:
            # The gate ACCEPTS untraceable candidates ("exact check still applies"). Counted
            # separately: these are not proven distinct, only unjudged.
            untraceable.append({"name": name, "fam": fam(name)})
            continue

        v = np.asarray(emb, dtype=np.float64).ravel()
        n = np.linalg.norm(v)
        if n == 0:
            untraceable.append({"name": name, "fam": fam(name)})
            continue
        v = v / n

        if kept_vecs:
            sims = np.asarray(kept_vecs) @ v
            j = int(np.argmax(sims))
            if sims[j] > a.cos_max:
                graded_dupes.append({"name": name, "dup_of": kept_names[j],
                                     "cos": round(float(sims[j]), 4), "fam": fam(name)})
                continue
        kept_vecs.append(v)
        kept_names.append(name)

    n_exact = len(exact_seen)
    n_graded = len(kept_names)
    by_fam = {}
    for d in graded_dupes:
        by_fam[d["fam"]] = by_fam.get(d["fam"], 0) + 1

    print(f"\n{'':-<66}")
    print(f"rows in nn                          {len(rows):>6}")
    print(f"  exact (arch_uid2) duplicates      {len(exact_dupes):>6}   already excluded by the working half")
    print(f"  distinct by arch_uid2             {n_exact:>6}   <- the number currently claimed")
    print(f"  of those, untraceable             {len(untraceable):>6}   unjudged by the graded half, admitted anyway")
    print(f"  ALSO distinct at cos<={a.cos_max}        {n_graded:>6}   <- what a working graded gate keeps")
    print(f"  near-duplicates admitted          {len(graded_dupes):>6}   <- INFLATION")
    if n_exact:
        print(f"  inflation rate                    {100*len(graded_dupes)/n_exact:>5.1f}%")
    print(f"{'':-<66}")
    print("near-duplicates by family:")
    for k in sorted(by_fam, key=lambda k: -by_fam[k]):
        print(f"  {k:<16} {by_fam[k]:>6}")
    print("\nclosest 15 admitted near-duplicates:")
    for d in sorted(graded_dupes, key=lambda d: -d["cos"])[:15]:
        print(f"  cos={d['cos']:.4f}  {d['name'][:44]:<46} ~ {d['dup_of'][:40]}")

    json.dump({"cos_max": a.cos_max, "n_rows": len(rows), "n_exact_distinct": n_exact,
               "n_graded_distinct": n_graded, "n_untraceable": len(untraceable),
               "n_exact_dupes": len(exact_dupes), "n_graded_dupes": len(graded_dupes),
               "by_family": by_fam, "graded_dupes": graded_dupes,
               "untraceable": untraceable, "exact_dupes": exact_dupes},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

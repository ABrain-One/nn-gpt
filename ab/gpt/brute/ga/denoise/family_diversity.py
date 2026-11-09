"""Per-family architectural diversity: does LLM generation produce less redundancy than GA?

WHY NOT REUSE THE CORPUS-WIDE PASS. dedupe_audit.py replays ONE greedy gate over all 2,958
rows in nn.id order. Greedy acceptance is order-dependent -- whoever arrives first is kept and
everything near it is rejected -- so a family that entered the DB early looks artificially
diverse and one that entered late looks artificially redundant. Comparing families under that
pass measures insertion history as much as architecture.

WHAT THIS DOES INSTEAD. Each family is deduped ONLY against itself, in its own greedy pass.
Cross-family order effects vanish, and the question becomes exactly the one worth asking:
given N architectures from generator X, how many survive a novelty threshold? Families are
also compared at MATCHED SAMPLE SIZE by subsampling every family down to the smallest one --
greedy rejection rate rises with set size (more chances to be near something already kept),
so DenoiseGA-'s 2,049 models would look more redundant than LLM_'s 470 even if the two
generators were identical. That confound is the whole ballgame and is corrected by
bootstrapping equal-size subsets.

THRESHOLD ROBUSTNESS. Reported at cos in {0.85, 0.90, 0.95}. A diversity claim that only holds
at one hand-picked threshold is not a finding, so the comparison must survive all three or be
reported as threshold-dependent.

EMBEDDING RESOLUTION CAVEAT, CARRIED IN THE OUTPUT. 76 corpus pairs sit at cosine exactly
1.0000 while being arch_uid2-distinct: the WL embedding cannot separate them at all. Absolute
survival counts therefore understate true diversity by an unknown amount. RELATIVE comparison
between families is still sound, because every family is measured with the same instrument.

  CUDA_VISIBLE_DEVICES= python family_diversity.py --db .../ab.nn.db
"""
import argparse
import json
import os
import random
import sqlite3
import sys
import time

FAMS = ("LLM_", "DenoiseGA-", "DnSeedGA-", "DenoiseSkip-")


def family_of(name):
    for p in FAMS:
        if name.startswith(p):
            return p
    return "other"


def greedy_keep(vecs, cos_max):
    """Greedy novelty filter -- the same rule DiversityGate applies. Returns kept indices."""
    import numpy as np
    kept = []
    mat = None
    for i, v in enumerate(vecs):
        if mat is not None and len(kept):
            if float(np.max(mat[:len(kept)] @ v)) > cos_max:
                continue
        if mat is None:
            mat = np.zeros((len(vecs), v.shape[0]), dtype=np.float64)
        mat[len(kept)] = v
        kept.append(i)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cache", default="/a/mm/sweep_out/family_embeds.npz")
    ap.add_argument("--out", default="/a/mm/sweep_out/family_diversity.json")
    ap.add_argument("--boots", type=int, default=20, help="subsample repeats for matched-N")
    ap.add_argument("--seed", type=int, default=0)
    # Matched-N is set by the SMALLEST family in the comparison, so including a tiny family
    # costs every other family its statistical power: with `other` (58 hand-written seeds) in
    # the set, N=58 and the bootstrap sd is ~5 points; dropping it lifts N to 279 and roughly
    # halves the error bars on the LLM-vs-GA contrast that is the actual claim. Run it both
    # ways -- `other` is a useful human-designed reference point, just not at the same N.
    ap.add_argument("--drop", default="", help="comma-separated families to exclude")
    a = ap.parse_args()
    drop = {s for s in a.drop.split(",") if s}

    import torch
    assert not torch.cuda.is_available(), (
        "CUDA visible -- rerun with CUDA_VISIBLE_DEVICES= . A contended card makes the graded "
        "trace throw OutOfMemoryError, which arch_embed swallows into a silent None.")
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import arch_embed
    from arch_uid2 import arch_uid

    names, vecs = [], []
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        names, vecs = list(z["names"]), list(z["vecs"])
        print(f"loaded {len(names)} cached embeddings from {a.cache}")
    else:
        con = sqlite3.connect(a.db)
        rows = con.execute("SELECT id, name, code FROM nn ORDER BY id").fetchall()
        seen, t0 = set(), time.time()
        for i, (_rid, name, code) in enumerate(rows):
            if i % 200 == 0 and i:
                print(f"  [{i}/{len(rows)}] embedded={len(names)} {time.time()-t0:.0f}s", flush=True)
            try:
                uid = arch_uid(code)
            except Exception:
                uid = None
            if uid is not None:
                if uid in seen:            # exact duplicates are already excluded upstream
                    continue
                seen.add(uid)
            try:
                _h, e = arch_embed.both_source(code)
            except Exception:
                e = None
            if e is None:
                continue                   # untraceable: the gate cannot judge, so neither can we
            v = np.asarray(e, dtype=np.float64).ravel()
            n = np.linalg.norm(v)
            if n == 0:
                continue
            names.append(name)
            vecs.append(v / n)
        np.savez_compressed(a.cache, names=np.array(names, dtype=object),
                            vecs=np.array(vecs, dtype=np.float64))
        print(f"embedded {len(names)} architectures -> {a.cache}")

    byfam = {}
    for n, v in zip(names, vecs):
        byfam.setdefault(family_of(n), []).append(v)
    fams = {k: v for k, v in byfam.items() if len(v) >= 30 and k not in drop}
    print(f"\nfamilies with >=30 traceable architectures: "
          f"{ {k: len(v) for k, v in fams.items()} }")
    nmin = min(len(v) for v in fams.values())
    print(f"matched sample size N={nmin} (smallest family)\n")

    rng = random.Random(a.seed)
    out = {"n_min": nmin, "boots": a.boots, "thresholds": {}}
    for cm in (0.85, 0.90, 0.95):
        print(f"=== cos_max {cm} ===")
        print(f"{'family':<15}{'N_all':>7}{'kept_all':>9}{'rate_all':>9}"
              f"{'kept@N':>9}{'sd':>7}{'rate@N':>9}")
        out["thresholds"][str(cm)] = {}
        for fam in sorted(fams, key=lambda f: -len(fams[f])):
            vs = fams[fam]
            kept_all = len(greedy_keep(vs, cm))
            sub = []
            for b in range(a.boots):
                idx = list(range(len(vs)))
                rng.shuffle(idx)                     # also randomises greedy order -> unbiased
                sub.append(len(greedy_keep([vs[i] for i in idx[:nmin]], cm)))
            mean = sum(sub) / len(sub)
            sd = (sum((x - mean) ** 2 for x in sub) / len(sub)) ** 0.5
            print(f"{fam:<15}{len(vs):>7}{kept_all:>9}{100*kept_all/len(vs):>8.1f}%"
                  f"{mean:>9.1f}{sd:>7.1f}{100*mean/nmin:>8.1f}%")
            out["thresholds"][str(cm)][fam] = {
                "n_all": len(vs), "kept_all": kept_all,
                "rate_all": round(kept_all / len(vs), 4),
                "kept_matched_mean": round(mean, 2), "kept_matched_sd": round(sd, 2),
                "rate_matched": round(mean / nmin, 4)}
        print()

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print("\nREAD rate@N, NOT rate_all: greedy rejection rises with set size, so the larger "
          "family is penalised on rate_all regardless of its true diversity.")


if __name__ == "__main__":
    main()

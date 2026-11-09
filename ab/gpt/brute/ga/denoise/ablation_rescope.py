"""Rescope the ablation manifest so every retained run is interpretable.

THE PROBLEM THIS FIXES. The manifest was built before the corpus inventory existed, so its
placement arms were sized by what skip_operators could INSERT rather than by what could be
MEASURED. The corpus inventory then put the real effect sizes at +0.22 dB (GLOBAL), +0.21 dB
(LATERAL) and +0.03 dB (BLOCKRES) -- observational, and therefore if anything inflated by the
capacity/family confound. Against the measured 0.14 dB run-to-run noise floor (paired
difference sd 0.198 dB), that makes the minimum detectable effect the binding constraint:

    contrast                        n    SEM     MDE(80% power)
    pooled over 8 bases            24    0.040   0.11 dB   <- fine
    one placement on 2 bases        6    0.081   0.23 dB   <- larger than the effect
    one placement on 1 base         3    0.114   0.32 dB   <- an anecdote

WHAT CHANGES.

1. place_block: DROPPED ENTIRELY (13 variants). Ten distinct placements spread over thirteen
   variants -- the best is on 2 of 8 bases and six sit on a SINGLE base. No re-scoping saves
   this arm: there is no subset of it that is crossed over enough bases to estimate anything,
   and BLOCKRES had the smallest observational gap of the three (+0.03 dB) anyway.

2. place_lateral: RESTRICTED TO THE FULLY-CROSSED PLACEMENTS and given more repeats.
   L@identity (8/8 bases), L@t (5/8) and L@s (5/8) share a common set of 5 bases, so on those
   5 the three placements are fully crossed and can be compared to EACH OTHER -- which is the
   actual question ("where should a skip go"), and which a single placement cannot answer no
   matter how many repeats it gets. L@h (2/8), L@e0 (1/8) and L@e1 (1/8) are dropped.
   Repeats rise 3 -> 5, taking the placement MDE to 0.11 dB.

3. factorial: UNCHANGED. 64 variants x 3 repeats already resolves 0.11 dB pooled over the 8
   bases, and it is the only part that can populate G=1,L=1,B=1 -- a cell holding just 5 of
   the corpus's 2,958 models, which is precisely what observation cannot reach.

HONEST ACCOUNTING: this is not the ~35% compute cut it looks like. Dropping 13 block variants
removes 39 runs, but adding repeats to the retained lateral placements adds 30 back, so the
total falls 297 -> 267 (-10%). Wall-clock barely moves either, because ablation_train.sh
shards by BASE and the retained placements concentrate on 5 of the 8 bases. What is bought is
not time: it is that every run left in the design can support a claim.

  python ablation_rescope.py --write
"""
import argparse
import collections
import json
import math
import os

KEEP_LATERAL = ("L@identity", "L@t", "L@s")
FACTORIAL_REPEATS = 3
PLACEMENT_REPEATS = 5
SD_DIFF = 0.198          # sqrt(2) * the 0.14 dB measured noise floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/shared/ssd/home/b-b-ahmed/thesis/sweep_out/ablation/manifest.json")
    ap.add_argument("--write", action="store_true", help="without this, only report")
    a = ap.parse_args()

    rows = json.load(open(a.manifest))
    valid = [r for r in rows if r.get("valid") and r.get("path")]
    bases = sorted({r["base"] for r in valid})

    per_cell = collections.defaultdict(set)
    for r in valid:
        if r["kind"] == "place_lateral":
            per_cell[r["cell"]].add(r["base"])
    common = set.intersection(*[per_cell[c] for c in KEEP_LATERAL]) if all(
        c in per_cell for c in KEEP_LATERAL) else set()

    kept = []
    for r in valid:
        if r["kind"] == "factorial":
            kept.append(dict(r, repeats=FACTORIAL_REPEATS))
        elif r["kind"] == "place_lateral" and r["cell"] in KEEP_LATERAL and r["base"] in common:
            kept.append(dict(r, repeats=PLACEMENT_REPEATS))
        # place_block and uncrossed laterals are dropped

    old_runs = len(valid) * 3
    new_runs = sum(r["repeats"] for r in kept)
    print(f"bases: {len(bases)} | variants {len(valid)} -> {len(kept)} | "
          f"runs {old_runs} -> {new_runs} ({100*(new_runs-old_runs)/old_runs:+.0f}%)")
    print(f"fully-crossed lateral bases: {len(common)}")

    by_kind = collections.Counter(r["kind"] for r in kept)
    print("\nretained by kind:")
    for k, n in by_kind.items():
        reps = FACTORIAL_REPEATS if k == "factorial" else PLACEMENT_REPEATS
        print(f"  {k:<16} {n:>3} variants x {reps} repeats = {n*reps:>3} runs")

    # Per-shard load, since ablation_train.sh assigns whole BASES to shards (i % NSHARDS).
    load = collections.Counter()
    for r in kept:
        load[r["base"]] += r["repeats"]
    ns = 8
    shard = collections.Counter()
    for i, b in enumerate(bases):
        shard[i % ns] += load.get(b, 0)
    mx = max(shard.values()) if shard else 0
    print(f"\nper-shard runs (NSHARDS=8): {sorted(shard.values(), reverse=True)}")
    print(f"critical path: {mx} runs x ~47 min = {mx*47/60:.1f} h")

    print("\npower of the retained design:")
    for lbl, n in (("factorial, pooled over 8 bases", 8 * FACTORIAL_REPEATS),
                   ("lateral placement, 5 bases", len(common) * PLACEMENT_REPEATS)):
        sem = SD_DIFF / math.sqrt(n)
        print(f"  {lbl:<32} n={n:<3} SEM {sem:.3f}  MDE {2.8*sem:.2f} dB")
    print("  (observational effects to beat: GLOBAL +0.22, LATERAL +0.21, BLOCKRES +0.03 dB)")

    if a.write:
        # The runner reads `repeats` per row if present; REPEATS stays as the env fallback.
        bak = a.manifest + ".prescope"
        if not os.path.exists(bak):
            os.replace(a.manifest, bak)
            print(f"\noriginal preserved at {bak}")
        json.dump(kept, open(a.manifest, "w"), indent=1)
        print(f"wrote {len(kept)} variants -> {a.manifest}")
    else:
        print("\n(dry run -- pass --write to apply)")


if __name__ == "__main__":
    main()

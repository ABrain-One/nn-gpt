"""GLOBAL-skip arm, asked from the removable side: Gkeep vs Grm.

WHY THIS SCRIPT EXISTS SEPARATELY FROM ablation_factorial.py. That script analyses ONE results
directory and its degeneracy screen is written around the G0/G1 cell names. This contrast spans TWO
runs -- Gkeep lives in sweep_out/ablation (the G0 cells of the original factorial) and Grm lives in
sweep_out/ablation_g2 -- so it needs its own loader. Folding it into the 2x2 script would mean
teaching that script about a factor whose two levels come from different directories, which is more
coupling than the one number is worth.

THE FACTOR, AND ITS SIGN. Every one of the 99 generated bases ALREADY ends in a global residual
(`return clamp(out_conv(d0) + identity, 0, 1)`), so "add one" has no valid treatment -- the first
attempt doubled the existing residual and all 96 runs collapsed onto the 33.021 dB no-op, which is
why that arm is void (see ablation_gen_gremove.py for the full account). Asked the other way round:

    Gkeep  the base as generated, residual present   -> the G0 cells of sweep_out/ablation
    Grm    residual stripped out                     -> sweep_out/ablation_g2

    contrast = Gkeep - Grm,  paired within (base, B-level, L-level, repeat)
    POSITIVE means the global residual HELPS.

WHAT THIS CONTRAST DOES AND DOES NOT LICENSE. It measures whether the shortcut these architectures
already have is load-bearing. It says NOTHING about whether a second, additional global skip would
help -- that is the question the doubling bug made unanswerable, and no run in either directory
addresses it. Reporting this as "global skips help" would overclaim; it is "removing the existing
global skip hurts".

THE MEAN IS NOT THE HEADLINE. Stripping the residual sends a minority of configurations into
collapse (tens of dB, not tenths), so the arithmetic mean is a mixture of "slightly worse" and
"destroyed". Both are reported: the mean for completeness, a trimmed mean and the median as the
robust centre, and the collapse count explicitly, because "removal costs 1.1 dB on average" and
"removal destroys 7% of configurations outright" are different engineering facts and the second is
the one that should drive a decision.
"""
import argparse
import collections
import glob
import json
import os
import statistics as st

ROOT = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = os.path.join(ROOT, "sweep_out")
NOOP_DB = 33.021          # PSNR of returning the noisy input unchanged, 256x256 centre crop
COLLAPSE_DB = 30.0        # below this a run has not "scored worse", it has failed


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "result_shard*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def level(cell):
    """-> ('Gkeep'|'Grm'|None, '<B><L> suffix')  from a cell name."""
    if cell.startswith("Grm"):
        return "Grm", cell[3:]
    if cell.startswith("G0") and len(cell) == 6:
        return "Gkeep", cell[2:]
    return None, None          # G1* is the void arm; L@* is the placement arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dir", default=os.path.join(SWEEP, "ablation", "train"))
    ap.add_argument("--rm-dir", default=os.path.join(SWEEP, "ablation_g2", "train"))
    a = ap.parse_args()

    rows = load(a.keep_dir) + load(a.rm_dir)
    blocks = collections.defaultdict(dict)
    for r in rows:
        lvl, suffix = level(r.get("cell", ""))
        if lvl is None or r.get("status") != "OK" or r.get("reported_dB") is None:
            continue
        blocks[(r["base"], suffix, r.get("repeat"))][lvl] = r["reported_dB"]

    paired = {k: v for k, v in blocks.items() if "Gkeep" in v and "Grm" in v}
    print("=" * 92)
    print("GLOBAL RESIDUAL — Gkeep vs Grm  (positive = the residual helps)")
    print("=" * 92)
    print(f"\n{len(paired)} paired blocks | "
          f"{len({b for b, _, _ in paired})} bases | "
          f"{len({s for _, s, _ in paired})} (B,L) cells")

    # SAME EXCLUSION RULE AS THE 2x2, applied for the same reason. ablation_factorial.py drops a base
    # whose repeats of one configuration disagree by whole decibels: that base is diverging, not
    # expressing a topology effect, and one broken base can dictate a pooled contrast. Applying the
    # rule to the untouched (Gkeep) side only, so the exclusion cannot be caused by the treatment.
    keep_sd = collections.defaultdict(list)
    for (b, s, _), v in paired.items():
        keep_sd[(b, s)].append(v["Gkeep"])
    unstable = sorted({b for (b, _), vs in keep_sd.items() if len(vs) >= 2 and st.stdev(vs) > 1.0})
    # A base that is barely above the no-op even untouched has no headroom to lose and its
    # "difference" is noise about a floor; report it rather than silently pooling it.
    per_keep = collections.defaultdict(list)
    for (b, _, _), v in paired.items():
        per_keep[b].append(v["Gkeep"])
    weak = sorted({b for b, vs in per_keep.items() if st.mean(vs) < NOOP_DB + 1.0})
    for label, group in (("UNSTABLE (a repeat sd > 1.0 dB on the untouched side)", unstable),
                         ("BARELY FUNCTIONAL untouched (Gkeep mean < no-op + 1 dB)", weak)):
        if group:
            print(f"\n  {label} -- excluded from the pooled contrast:")
            for b in group:
                print(f"    {b[:46]}  Gkeep mean {st.mean(per_keep[b]):.2f} dB")
    drop = set(unstable) | set(weak)
    if drop:
        paired_all = paired
        paired = {k: v for k, v in paired.items() if k[0] not in drop}
        print(f"  pooling {len(paired)} of {len(paired_all)} blocks "
              f"({len({b for b, _, _ in paired})} of {len(per_keep)} bases)")

    d = [v["Gkeep"] - v["Grm"] for v in paired.values()]
    mean = st.mean(d)
    sd = st.stdev(d)
    sem = sd / len(d) ** 0.5
    k = max(1, len(d) // 10)
    trimmed = st.mean(sorted(d)[k:-k])
    helps = sum(1 for x in d if x > 0)

    print(f"\n  mean            {mean:+.3f} dB   sem {sem:.3f}  "
          f"95% CI [{mean - 1.96 * sem:+.3f}, {mean + 1.96 * sem:+.3f}]  "
          f"{'RESOLVED' if abs(mean) > 1.96 * sem else 'not resolved'}")
    print(f"  median          {st.median(d):+.3f} dB")
    print(f"  10% trimmed     {trimmed:+.3f} dB   <- robust centre, collapses excluded")
    print(f"  sign agreement  {helps}/{len(d)} blocks scored worse with the residual removed")
    print(f"  range           {min(d):+.2f} .. {max(d):+.2f} dB")

    # The tail is the decision-relevant part, so name it rather than letting the mean absorb it.
    dead = {k: v for k, v in paired.items() if v["Grm"] < COLLAPSE_DB}
    print(f"\n  configurations DESTROYED by removal (< {COLLAPSE_DB:.0f} dB): "
          f"{len(dead)}/{len(paired)} = {len(dead) / len(paired):.1%}")
    if dead:
        by_cell = collections.Counter(s for _, s, _ in dead)
        print(f"    by (B,L) cell: {dict(by_cell)}")
        worst = min(dead.items(), key=lambda kv: kv[1]["Grm"])
        print(f"    worst: {worst[0][0][:34]} {worst[0][1]} r{worst[0][2]}  "
              f"{worst[1]['Gkeep']:.2f} -> {worst[1]['Grm']:.2f} dB")
    at_noop = sum(1 for v in paired.values() if abs(v["Grm"] - NOOP_DB) < 0.01)
    print(f"  landed ON the {NOOP_DB} dB no-op after removal: {at_noop}")

    print("\n  per-base (mean over that base's cells and repeats)")
    print(f"    {'base':<40}{'Gkeep':>9}{'Grm':>9}{'diff':>9}{'blocks':>8}")
    per = collections.defaultdict(list)
    for (b, _, _), v in paired.items():
        per[b].append(v)
    for b, vs in sorted(per.items(), key=lambda kv: -st.mean([x["Gkeep"] - x["Grm"] for x in kv[1]])):
        gk = st.mean([x["Gkeep"] for x in vs])
        gr = st.mean([x["Grm"] for x in vs])
        print(f"    {b[:38]:<40}{gk:>9.3f}{gr:>9.3f}{gk - gr:>+9.3f}{len(vs):>8}")

    print(f"""
READING IT. The global residual every base already carries is load-bearing: removing it costs
{-mean:+.3f} dB on the mean and {-trimmed:+.3f} dB on the trimmed centre, and {helps} of {len(d)} paired blocks got
worse. But the mean is a mixture -- {len(dead)} configurations did not degrade, they FAILED, and one
fell to {min(v['Grm'] for v in paired.values()):.2f} dB. Quote the trimmed figure as the typical cost and the collapse
count as the risk; the arithmetic mean alone overstates the first and hides the second.

This says the EXISTING shortcut is necessary. It does not say a second one would help -- that
question died with the doubled-residual arm and remains unmeasured.""")


if __name__ == "__main__":
    main()

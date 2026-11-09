"""Analyse the completed 2x2x2 skip-connection factorial.

THE DESIGN IS WITHIN-BASE AND THE ANALYSIS MUST BE PAIRED. Eight base architectures were each
rebuilt in all eight combinations of GLOBAL / LATERAL / BLOCKRES, three repeats per cell: 192 runs.
Pooling all 192 as if they were independent would drown every effect in between-base variance --
bases differ by whole decibels while the effects sit near a tenth of one. So each effect is computed
as a CONTRAST INSIDE a base first, and only then averaged across the eight bases, with the standard
error taken across bases. That is what makes a 0.1 dB effect detectable at all.

WHY THIS BEATS THE OBSERVATIONAL ESTIMATE. Section 19's corpus scan put the marginal gaps at
GLOBAL +0.173, LATERAL +0.531, BLOCKRES -0.155 dB, but those are confounded: models carrying a
lateral skip also differ in width, depth, activation and normalisation. Here everything except the
skip wiring is held fixed within a base, so a difference is attributable to the topology.

THE MAC CONFOUND IS CHECKED, NOT ASSUMED. Adding a skip can add computation, and a PSNR gain bought
with extra MACs is not a topology finding. mac_delta_pct is reported per cell so a reader can see
whether the winning configuration is also the more expensive one.

THE NOISE FLOOR USED HERE IS THE MEASURED ONE. The design was scoped against an assumed 0.14 dB
per-run sd; the confirmation run measured 0.207 dB on AWGN, which raises the pooled minimum
detectable effect from 0.11 to about 0.16 dB. Effects below that are reported as not resolved rather
than as zero.

  python ablation_factorial.py
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics as st

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
MEASURED_SD = 0.207          # per-run sd, measured in the confirmation run (was assumed 0.14)
NOOP_DB = 33.021             # the do-nothing PSNR on this benchmark (§05, 256x256 centre crop)

# THE GLOBAL ARM OF THIS FACTORIAL IS VOID, AND THE ANALYSIS MUST SAY SO RATHER THAN AVERAGE IT IN.
# All 96 G1 runs scored EXACTLY 33.021 dB -- the no-op floor -- across 8 different bases and 4
# different cells. The cause is in the generated source, not the training: the bases already ended in
#     return clamp(out_conv(d0) + identity, 0, 1)          # identity = x, a global residual
# and the G1 injection wrapped that in a SECOND one:
#     return x + clamp(out_conv(d0) + identity, 0, 1)
# A doubled residual with a clamp between them has the no-op as an attractor -- drive the inner term
# to <=0, the clamp zeroes it, and the output is the input exactly. Every G1 run found it.
# So G1 measures "what happens when the variant generator breaks the model", not "what a global skip
# does". Reporting a GLOBAL main effect of -1.269 dB from these rows would be reporting a codegen bug
# as a scientific result. The G0 half is clean (0 of 96 at the floor) and yields a valid 2x2 on
# LATERAL x BLOCKRES, which is what this script now analyses.


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "result_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return [r for r in rows if r.get("status") == "OK" and r.get("best_dB") is not None
            and r.get("kind") == "factorial"]


def mean_ci(xs):
    """mean, sem, and a 95% interval using the t multiplier for small n."""
    n = len(xs)
    m = st.mean(xs)
    if n < 2:
        return m, 0.0, (m, m)
    sem = st.stdev(xs) / math.sqrt(n)
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(n, 2.31)
    return m, sem, (m - t * sem, m + t * sem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(SWEEP, "ablation", "train"))
    ap.add_argument("--manifest", default=os.path.join(SWEEP, "ablation", "manifest.json"))
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    man = man if isinstance(man, list) else man.get("rows", man.get("variants", []))
    macs = {(r["base"], r["cell"]): r.get("mac_delta_pct", 0.0)
            for r in man if r.get("kind") == "factorial"}

    rows = load(a.dir)

    # ---- degeneracy screen, before any effect is computed --------------------------------
    deg = [r for r in rows if abs(r["best_dB"] - NOOP_DB) < 0.0005]
    if deg:
        by_cell = collections.Counter(r["cell"] for r in deg)
        print(f"DEGENERACY SCREEN: {len(deg)}/{len(rows)} runs sit EXACTLY at the "
              f"{NOOP_DB} dB no-op floor")
        print(f"  by cell: {dict(by_cell)}")
        dead_levels = set()
        for flag in "GBL":
            on = [r for r in rows if f"{flag}1" in r["cell"]]
            if on and all(abs(r["best_dB"] - NOOP_DB) < 0.0005 for r in on):
                dead_levels.add(flag)
        if dead_levels:
            print(f"  EVERY run with {'/'.join(sorted(dead_levels))}=1 is degenerate -> that factor "
                  f"is VOID and is dropped from the design.")
            print(f"  These variants return their input; see the module header for the codegen cause.")
        rows = [r for r in rows if not any(f"{f}1" in r["cell"] for f in dead_levels)]
        factors = [f for f in "GLB" if f not in dead_levels]
        print(f"  analysing the surviving {len(rows)} runs as a "
              f"{'x'.join(['2']*len(factors))} on {'/'.join(factors)}\n")
    else:
        factors = list("GLB")
    # cell mean per (base, cell) over repeats
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["base"], r["cell"])].append(r["best_dB"])
    bases = sorted({b for b, _ in cells})
    names = sorted({r["cell"] for r in rows})

    # A base whose repeats disagree by whole decibels is DIVERGING, not expressing a topology
    # effect, and averaging it in lets one broken base dictate a pooled contrast. Reported and
    # excluded: DnSeedGA-146bae38 returned 13.85 and 28.79 dB for the SAME configuration.
    unstable = sorted({b for (b, c), v in cells.items() if len(v) >= 2 and st.stdev(v) > 1.0})
    if unstable:
        print("UNSTABLE BASES (a repeat sd above 1.0 dB in some cell) -- excluded from pooling:")
        for b in unstable:
            worst = max((st.stdev(v), c) for (bb, c), v in cells.items() if bb == b and len(v) >= 2)
            print(f"  {b[:46]}  worst cell {worst[1]}, repeat sd {worst[0]:.2f} dB")
        print()

    complete = [b for b in bases if all((b, c) in cells for c in names)]
    print(f"{len(rows)} factorial runs | {len(bases)} bases | "
          f"{len(complete)} with every cell measured")
    complete = [b for b in complete if b not in unstable]
    if len(complete) < 3:
        print("not enough clean, complete bases")
        return
    reps = collections.Counter(len(v) for v in cells.values())
    print(f"  repeats per cell: {dict(reps)}")

    # within-cell sd -> the real per-run noise in THIS experiment
    sds = [st.stdev(v) for v in cells.values() if len(v) >= 2]
    clean = {k: v for k, v in cells.items() if k[0] in complete and len(v) >= 2}
    pooled = math.sqrt(sum((len(v) - 1) * st.variance(v) for v in clean.values())
                       / sum(len(v) - 1 for v in clean.values()))
    print(f"  pooled within-cell sd: {pooled:.3f} dB "
          f"(confirmation run measured 0.207; design assumed 0.140)")

    mean = {k: st.mean(v) for k, v in cells.items()}

    print(f"\n=== CELL MEANS by base (dB), and the MAC cost of each cell ===")
    print(f"  {'base':<26}" + "".join(f"{c:>9}" for c in names))
    for b in complete:
        print(f"  {b[:25]:<26}" + "".join(f"{mean[(b,c)]:>9.3f}" for c in names))
    print(f"  {'mean MAC delta %':<26}" +
          "".join(f"{st.mean([macs.get((b,c),0.0) for b in complete]):>9.1f}" for c in names))

    # ---- main effects, paired within base then averaged across bases -------------------
    def effect(flag):
        """within-base contrast: mean(flag=1 cells) - mean(flag=0 cells), per base."""
        out = []
        for b in complete:
            on = [mean[(b, c)] for c in names if f"{flag}1" in c]
            off = [mean[(b, c)] for c in names if f"{flag}0" in c]
            out.append(st.mean(on) - st.mean(off))
        return out

    mde = 2.8 * pooled / math.sqrt(len(complete) * 4 * 3)   # 4 cells x 3 repeats per side, n bases
    print(f"\n=== MAIN EFFECTS (paired within base, averaged over {len(complete)} bases) ===")
    print(f"  minimum detectable effect at this design ~ {mde:.3f} dB\n")
    print(f"  {'effect':<28}{'dB':>8}{'sem':>8}{'95% CI':>20}   verdict")
    results = {}
    LABELS = {"G": ("GLOBAL  input->output", +0.173),
              "L": ("LATERAL enc->dec", +0.531),
              "B": ("BLOCKRES within-block", -0.155)}
    for flag in factors:
        label, obs = LABELS[flag]
        e = effect(flag)
        m, sem, (lo, hi) = mean_ci(e)
        sig = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved"
        results[flag] = {"dB": m, "sem": sem, "ci": [lo, hi], "per_base": e,
                         "observational": obs, "resolved": sig == "RESOLVED"}
        print(f"  {label:<28}{m:>+8.3f}{sem:>8.3f}   [{lo:>+6.3f},{hi:>+6.3f}]   {sig}")
        print(f"  {'  observational (§19)':<28}{obs:>+8.3f}{'':>8}   "
              f"{'confounded -- width/depth/activation vary too':>20}")
        print(f"  {'  per-base':<28}" + " ".join(f"{x:+.2f}" for x in e))
        agree = sum(1 for x in e if (x > 0) == (m > 0))
        print(f"  {'  sign agreement':<28}{agree}/{len(e)} bases share the pooled sign\n")

    # ---- interactions ------------------------------------------------------------------
    def inter(f1, f2):
        out = []
        for b in complete:
            def cm(v1, v2):
                sel = [mean[(b, c)] for c in names if f"{f1}{v1}" in c and f"{f2}{v2}" in c]
                return st.mean(sel)
            out.append((cm(1, 1) - cm(1, 0)) - (cm(0, 1) - cm(0, 0)))
        return out

    print(f"=== TWO-WAY INTERACTIONS (does one skip's benefit depend on another?) ===")
    print(f"  {'pair':<28}{'dB':>8}{'sem':>8}{'95% CI':>20}   verdict")
    import itertools
    for f1, f2 in itertools.combinations(factors, 2):
        e = inter(f1, f2)
        m, sem, (lo, hi) = mean_ci(e)
        sig = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved"
        results[f"{f1}x{f2}"] = {"dB": m, "sem": sem, "ci": [lo, hi], "resolved": sig == "RESOLVED"}
        print(f"  {f1} x {f2:<24}{m:>+8.3f}{sem:>8.3f}   [{lo:>+6.3f},{hi:>+6.3f}]   {sig}")

    # ---- best and worst configuration, pooled ------------------------------------------
    print(f"\n=== CONFIGURATIONS RANKED (mean over {len(complete)} bases, each base centred) ===")
    centred = {c: st.mean([mean[(b, c)] - st.mean([mean[(b, x)] for x in names])
                           for b in complete]) for c in names}
    print(f"  {'cell':<10}{'dB vs base mean':>18}{'MAC delta %':>14}")
    for c, v in sorted(centred.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<10}{v:>+18.3f}{st.mean([macs.get((b,c),0.0) for b in complete]):>14.1f}")
    best, worst = max(centred, key=centred.get), min(centred, key=centred.get)
    print(f"\n  best {best} vs worst {worst}: {centred[best]-centred[worst]:+.3f} dB "
          f"-- the whole span the skip topology controls")

    out = {"n_runs": len(rows), "n_bases": len(complete), "pooled_within_cell_sd": pooled,
           "mde_dB": mde, "effects": results,
           "cell_centred_dB": centred,
           "span_dB": centred[best] - centred[worst], "best_cell": best, "worst_cell": worst}
    p = os.path.join(a.dir, "factorial_report.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

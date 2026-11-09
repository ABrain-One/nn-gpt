"""Does selecting on synthetic AWGN cost you real-noise performance, and by how much?

This refines crossdata_report.py, which measured every cell ONCE. With 3 repeats per cell the
questions that single draw could not answer become answerable:

  1. WHAT IS THE PER-DATASET NOISE FLOOR? Pooled within-cell sd, computed separately for AWGN,
     MIDD and SIDD. The project's 0.14 dB floor was measured on AWGN; MIDD's centre-crop test
     path and SIDD's different test set have no reason to share it, and every claim about rank
     agreement is bounded by these numbers.

  2. IS THE 1.90 dB REGRET REAL, OR ONE BAD DRAW? In the single-draw run AWGN's #1 pick
     (LLM_Pg33_seeded-8a7b5df) scored 34.56 on MIDD -- last of the 8 selected models -- while
     AWGN's #2 and #3 picks scored 36.14 and 36.15, i.e. MIDD ranks 3 and 2. So regret@1 rests
     entirely on one model at one rank. If 8a7b5df regresses toward the mean here, the honest
     headline is regret@3, which was 0.31 dB.

  3. REGRET AS A FUNCTION OF SHORTLIST SIZE. Reporting only k=1 overstates the problem: nobody
     ships the single top-scoring architecture without looking at the runners-up. regret@k is
     the practically meaningful curve, and it is bootstrapped so it carries an interval.

  4. HOW OFTEN DOES AWGN EVEN PICK THE SAME MODEL TWICE? Selection stability under resampling.
     If AWGN's #1 is not reproducible across repeats of its OWN metric, that alone settles the
     matter -- an unstable selector cannot be a valid one, independently of transfer.

WINNER'S CURSE CUTS THE OTHER WAY, AND IS REPORTED. The single-draw MIDD champion was chosen
BY its single noisy MIDD measurement, so its true value is expected to sit below 36.46 and the
1.90 dB regret is biased UPWARD. Re-measuring the champion is what corrects that, so the
comparison of single-draw vs re-measured dB is printed per model rather than buried.

MIDD ABSOLUTE dB REMAINS NON-QUOTABLE -- midd.py takes one 256x256 centre crop per full-res
test image with a model-dependent bias its own docstring measures at 0.16-0.32 dB. Fine for
ranking; never to be set against SplitterNet's published 37.92 dB.

  python confirm_report.py
"""
import argparse
import glob
import json
import math
import os
import statistics as st

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
DATASETS = ("denoise", "midd", "sidd")
LBL = {"denoise": "AWGN", "midd": "MIDD", "sidd": "SIDD"}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def load(d):
    """cells[(model, dataset)] = [dB, ...] over repeats."""
    cells = {}
    for f in glob.glob(os.path.join(d, "result_shard*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "OK" and r.get("best_dB") is not None:
                cells.setdefault((r["model"], r["dataset"]), []).append(r["best_dB"])
    return cells


def lcg(seed):
    """Deterministic PRNG. Math.random/random.seed-free so a rerun reproduces the intervals
    exactly; a bootstrap CI that moves between runs of the same data is not auditable."""
    s = seed

    def nxt(n):
        nonlocal s
        s = (1103515245 * s + 12345) % (1 << 31)
        return s % n
    return nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(SWEEP, "confirm"))
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    sub = json.load(open(os.path.join(a.dir, "subset.json")))
    single = sub.get("single_draw_dB", {})
    cells = load(a.dir)
    models = sorted({m for m, _ in cells})
    print(f"{len(cells)} cells with data | {len(models)} models")

    full = [m for m in models if all(len(cells.get((m, d), [])) >= 2 for d in DATASETS)]
    print(f"{len(full)} models have >=2 repeats on ALL THREE datasets")
    if len(full) < 4:
        print("not enough yet -- rerun when confirm-train has progressed")
        for m in models:
            print(f"  {m[:46]:<48} " + "  ".join(
                f"{LBL[d]}:{len(cells.get((m,d),[]))}" for d in DATASETS))
        return

    # ---- 1. per-dataset noise floor -------------------------------------------------
    print("\n=== 1. PER-DATASET NOISE FLOOR (pooled within-cell sd over repeats) ===")
    floors = {}
    for d in DATASETS:
        sds, ns = [], 0
        for m in full:
            v = cells[(m, d)]
            if len(v) >= 2:
                sds.append((len(v) - 1) * st.variance(v))
                ns += len(v) - 1
        floors[d] = math.sqrt(sum(sds) / ns) if ns else float("nan")
        per = [round(st.stdev(cells[(m, d)]), 3) for m in full if len(cells[(m, d)]) >= 2]
        print(f"  {LBL[d]:<5} pooled sd {floors[d]:.3f} dB   (per-model sd: "
              f"min {min(per):.3f} max {max(per):.3f})  paired-diff sd "
              f"{floors[d]*math.sqrt(2):.3f} dB")
    print("  reference: the AWGN floor measured earlier in this project was 0.14 dB")

    mean = {(m, d): st.mean(cells[(m, d)]) for m in full for d in DATASETS}

    # ---- 2. winner's curse ----------------------------------------------------------
    print("\n=== 2. SINGLE DRAW vs RE-MEASURED MEAN (winner's curse check) ===")
    print(f"{'':<46}{'AWGN':>16}{'MIDD':>16}{'SIDD':>16}")
    for m in sorted(full, key=lambda m: -mean[(m, 'midd')]):
        cols = ""
        for d in DATASETS:
            was = (single.get(m) or {}).get(d)
            now = mean[(m, d)]
            cols += f"  {was:6.2f}->{now:6.2f}" if was else f"  {'':>6}   {now:6.2f}"
        print(f"  {m[:44]:<44}{cols}")
    for d in DATASETS:
        dd = [mean[(m, d)] - single[m][d] for m in full if m in single and d in single[m]]
        if dd:
            print(f"  {LBL[d]:<5} mean shift {st.mean(dd):+.3f} dB "
                  f"(negative = the single draw was optimistic)")

    # ---- 3. rank transfer, attenuation-corrected ------------------------------------
    print("\n=== 3. RANK TRANSFER among these top models (n={}) ===".format(len(full)))
    print("  Corrected rho divides by sqrt(reliability_x * reliability_y): the observed value")
    print("  is capped by how well each metric agrees with ITSELF, so uncorrected numbers")
    print("  understate transfer. >1.0 after correction means indistinguishable from perfect.")
    rel = {}
    for d in DATASETS:
        # split-half: repeat 0 vs repeat 1 ranking agreement, Spearman-Brown corrected
        v0 = [cells[(m, d)][0] for m in full]
        v1 = [cells[(m, d)][1] for m in full]
        r = spearman(v0, v1)
        rel[d] = (2 * r / (1 + r)) if (r is not None and r > -1) else None
        rr = "n/a" if r is None else f"{r:+.3f}"
        rc = "n/a" if rel[d] is None else f"{rel[d]:+.3f}"
        print(f"  {LBL[d]:<5} split-half rho {rr}  -> reliability {rc}")
    for x, y in (("denoise", "midd"), ("denoise", "sidd"), ("midd", "sidd")):
        r = spearman([mean[(m, x)] for m in full], [mean[(m, y)] for m in full])
        corr = ""
        if r is not None and rel[x] and rel[y] and rel[x] > 0 and rel[y] > 0:
            c = r / math.sqrt(rel[x] * rel[y])
            corr = f"   corrected {c:+.3f}" + ("  (>=1: indistinguishable from perfect)" if c >= 1 else "")
        print(f"  {LBL[x]} -> {LBL[y]:<5} rho={r:+.3f}{corr}")

    # ---- 4. regret@k, bootstrapped --------------------------------------------------
    print("\n=== 4. SELECTION REGRET vs SHORTLIST SIZE (bootstrap over repeats) ===")
    print("  regret@k = (best real-noise dB in the set) - (best real-noise dB among the k")
    print("  models AWGN ranks highest). k=1 is 'trust AWGN's winner'; k=3 is 'shortlist 3")
    print("  on AWGN, then test those on real noise'.")
    nxt = lcg(20260806)

    def draw():
        return {(m, d): cells[(m, d)][nxt(len(cells[(m, d)]))] for m in full for d in DATASETS}

    ks = [k for k in (1, 2, 3, 5) if k <= len(full)]
    boot = {(k, d): [] for k in ks for d in ("midd", "sidd")}
    winner = {}
    for _ in range(a.boot):
        s = draw()
        order = sorted(full, key=lambda m: -s[(m, "denoise")])
        winner[order[0]] = winner.get(order[0], 0) + 1
        for d in ("midd", "sidd"):
            best = max(s[(m, d)] for m in full)
            for k in ks:
                boot[(k, d)].append(best - max(s[(m, d)] for m in order[:k]))
    for d in ("midd", "sidd"):
        print(f"\n  --- {LBL[d]} ---")
        for k in ks:
            v = sorted(boot[(k, d)])
            lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v))]
            pz = sum(1 for x in v if x <= floors[d]) / len(v)
            print(f"    regret@{k}  {st.mean(v):5.2f} dB  95% CI [{lo:.2f}, {hi:.2f}]   "
                  f"P(regret <= {LBL[d]} noise floor) = {pz:.0%}")

    print("\n=== 5. IS AWGN's WINNER EVEN REPRODUCIBLE? ===")
    tot = sum(winner.values())
    for m, c in sorted(winner.items(), key=lambda kv: -kv[1]):
        print(f"  {c/tot:5.1%}  {m[:52]}")
    print(f"  -> AWGN's #1 pick changes across resamples of its own metric in "
          f"{1 - max(winner.values())/tot:.0%} of draws")

    out = {"n_models": len(full), "floors": floors, "reliability": rel,
           "mean_dB": {f"{m}|{d}": mean[(m, d)] for m in full for d in DATASETS},
           "n_repeats": {f"{m}|{d}": len(cells[(m, d)]) for m in full for d in DATASETS},
           "regret": {f"k{k}|{d}": {"mean": st.mean(boot[(k, d)]),
                                    "ci": [sorted(boot[(k, d)])[int(.025*a.boot)],
                                           sorted(boot[(k, d)])[int(.975*a.boot)]]}
                      for k in ks for d in ("midd", "sidd")},
           "awgn_winner_share": {m: c / tot for m, c in winner.items()}}
    o = os.path.join(a.dir, "confirm_report.json")
    json.dump(out, open(o, "w"), indent=1)
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()

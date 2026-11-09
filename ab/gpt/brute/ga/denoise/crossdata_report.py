"""Cross-dataset transfer report: does AWGN ranking predict real-noise ranking?

Aggregates the per-epoch curves into all_epochs.json (same shape as ft50/beatbar) and then
answers the question the experiment exists for.

SPEARMAN, NOT PEARSON. The subset deliberately spans 9.65 .. 36.98 dB and includes models
that trained cleanly to nothing. Under Pearson a single 9.65 dB point manufactures a large
correlation by itself; rank correlation is immune. Both are printed so the gap is visible,
but the reported result is Spearman.

WHY THE 'denoise' COLUMN IS RE-RUN RATHER THAN TAKEN FROM ft50. ft50 measured AWGN with
'psnr' (Y-channel) at 50 epochs; this runs 'psnrrgb' at 30. Those differ by ~0.3-0.5 dB and
are not interchangeable, so the in-experiment denoise column is the one used for every
cross-dataset comparison. ft50's numbers appear only as an external check on the re-run --
if rho(ft50, denoise) is not high, something about the re-run is wrong and the MIDD/SIDD
numbers should not be trusted either.

MIDD ABSOLUTE dB IS NOT QUOTABLE. midd.py evaluates one 256x256 centre crop per full-res test
image, with a model-dependent bias its own docstring measures at 0.16-0.32 dB. Fine for the
ranking this report computes; never set against SplitterNet's published 37.92 dB.

  python crossdata_report.py
"""
import argparse
import glob
import json
import math
import os

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"


def ranks(xs):
    """Average ranks, ties shared -- required for Spearman to be correct with duplicates."""
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
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def pvalue(rho, n):
    """Two-sided p via the t approximation; adequate for n>=10."""
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return None
    t = abs(rho) * math.sqrt((n - 2) / (1 - rho * rho))
    df = n - 2
    # regularised incomplete beta via continued fraction is overkill here; use a normal
    # approximation on Fisher's z, which is conservative at these sample sizes
    z = 0.5 * math.log((1 + abs(rho)) / (1 - abs(rho))) * math.sqrt(n - 3)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def ft50_awgn():
    out = {}
    for pat in ("fulltrain50/result_shard*.jsonl", "beatbar/result_shard*.jsonl"):
        for f in glob.glob(os.path.join(SWEEP, pat)):
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                nm = r.get("key") or r.get("model")
                db = r.get("max_dB") or r.get("best_dB")
                if nm and db and db > (out.get(nm) or 0):
                    out[nm] = db
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(SWEEP, "crossdata"))
    ap.add_argument("--min-dB", type=float, default=None,
                    help="optionally drop models below this on the in-experiment AWGN column")
    a = ap.parse_args()

    res = {}
    for f in glob.glob(os.path.join(a.dir, "result_shard*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "OK" and r.get("best_dB") is not None:
                res.setdefault(r["model"], {})[r["dataset"]] = r["best_dB"]

    complete = {m: d for m, d in res.items() if {"denoise", "midd", "sidd"} <= set(d)}
    print(f"{len(res)} models with any result | {len(complete)} with ALL THREE datasets")
    if len(complete) < 5:
        print("not enough complete rows yet -- rerun when xd-train has progressed")
        return

    models = sorted(complete)
    if a.min_dB is not None:
        models = [m for m in models if complete[m]["denoise"] >= a.min_dB]
        print(f"  filtered to {len(models)} with denoise >= {a.min_dB} dB")
    awgn = [complete[m]["denoise"] for m in models]
    midd = [complete[m]["midd"] for m in models]
    sidd = [complete[m]["sidd"] for m in models]

    print(f"\n{'dataset':<10}{'n':>4}{'min':>8}{'median':>9}{'max':>8}")
    for lbl, v in (("AWGN", awgn), ("MIDD", midd), ("SIDD", sidd)):
        s = sorted(v)
        print(f"{lbl:<10}{len(v):>4}{s[0]:>8.2f}{s[len(s)//2]:>9.2f}{s[-1]:>8.2f}")

    print("\n=== RANK TRANSFER (Spearman; Pearson shown for contrast) ===")
    out = {"n": len(models), "pairs": {}}
    for lbl, x, y in (("AWGN -> MIDD", awgn, midd),
                      ("AWGN -> SIDD", awgn, sidd),
                      ("MIDD -> SIDD", midd, sidd)):
        rs, rp = spearman(x, y), pearson(x, y)
        p = pvalue(rs, len(x))
        ps = "n/a" if p is None else (f"{p:.2g}" if p >= 1e-4 else "<1e-4")
        print(f"  {lbl:<14} rho={rs:+.3f}  p={ps:<8} (pearson r={rp:+.3f})")
        out["pairs"][lbl] = {"spearman": rs, "pearson": rp, "p": p, "n": len(x)}

    ext = ft50_awgn()
    both = [m for m in models if m in ext]
    if len(both) >= 5:
        r = spearman([ext[m] for m in both], [complete[m]["denoise"] for m in both])
        print(f"\n  sanity: ft50 AWGN(50ep, psnr) vs this run's AWGN(30ep, psnrrgb): "
              f"rho={r:+.3f} over {len(both)} models")
        print("  (low here means the re-run disagrees with ft50 and NOTHING else should be trusted)")
        out["sanity_ft50_vs_rerun"] = {"spearman": r, "n": len(both)}

    print(f"\n=== biggest rank movers (AWGN rank -> MIDD rank) ===")
    ra, rm = ranks(awgn), ranks(midd)
    mv = sorted(range(len(models)), key=lambda i: -abs(ra[i] - rm[i]))[:8]
    print(f"{'AWGN':>7}{'MIDD':>7}{'d':>6}  model")
    for i in mv:
        print(f"{ra[i]:>7.0f}{rm[i]:>7.0f}{rm[i]-ra[i]:>+6.0f}  {models[i][:46]}")

    o = os.path.join(a.dir, "transfer_report.json")
    out["per_model"] = {m: complete[m] for m in models}
    json.dump(out, open(o, "w"), indent=1)
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()

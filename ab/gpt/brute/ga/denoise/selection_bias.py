"""Quantify the selection optimism in the corpus's full-resolution PSNR numbers.

WHY THIS EXISTS. The denoise loader exposes exactly two splits, train/ and validation/, so GA
fitness was scored on the same 50 validation frames the corpus reports. Every PSNR in the corpus
is therefore an IN-SAMPLE number for a search that selected on it. The decision (2026-08-30) was
to keep comparability with the 4,262 models already measured rather than re-split and orphan them,
and to REPORT the bias instead. This script is that report.

WHAT CAN AND CANNOT BE MEASURED. True selection bias needs data the search never saw, and no such
split exists retroactively. What the stored `per_frame` records DO support is the winner's-curse
decomposition, which is the same phenomenon measured within the frame set:

  1. RANK STABILITY. Split the 50 frames into halves A and B. Rank all models on A, rank on B,
     report Spearman. This bounds how much of any published ranking is frame-sampling noise
     rather than architecture quality -- if A and B disagree, the leaderboard is partly an
     artefact of which 50 images the challenge happened to ship.

  2. SELECTION OPTIMISM. Pick the top-K by half A, then read those same models on half B. Compare
     against the honest top-K of B. The shortfall is the amount by which selecting and reporting
     on one set inflates the reported number -- a direct, if conservative, estimate of what our
     corpus-wide figures overstate. Conservative because selecting on 25 frames is weaker
     selection than the GA's 1500-candidate search over all 50.

Both are computed over many random splits, because a single split is itself noise. Splits are
drawn from a FIXED seed so the reported numbers are reproducible.

  python3 selection_bias.py --glob '/a/mm/sweep_out/**/fullres*.jsonl' --topk 10 --splits 200
"""
import argparse
import glob as globmod
import json
import random
import statistics


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load(pattern):
    """Latest OK record per model, keeping only models with a complete per-frame breakdown."""
    per = {}
    for path in globmod.glob(pattern, recursive=True):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") not in (None, "OK"):
                continue
            name, frames = rec.get("model"), rec.get("per_frame")
            if not name or not isinstance(frames, dict) or not frames:
                continue
            # A model measured twice keeps its better run, matching how the corpus reports it.
            prev = per.get(name)
            if prev is None or sum(frames.values()) > sum(prev.values()):
                per[name] = frames
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/a/mm/sweep_out/**/fullres*.jsonl")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default="/a/mm/sweep_out/selection_bias.json")
    a = ap.parse_args()

    per = load(a.glob)
    if not per:
        raise SystemExit("no per-frame records matched " + a.glob)

    # Only frames present for EVERY model can be split, or half-means would be computed over
    # different image sets per model and the comparison would be meaningless.
    common = set.intersection(*(set(f) for f in per.values()))
    models = sorted(per)
    print(f"models {len(models)}  frames common to all {len(common)}")
    if len(common) < 4:
        raise SystemExit("too few common frames to split")
    frames = sorted(common)

    def mean_on(name, subset):
        f = per[name]
        return sum(f[k] for k in subset) / len(subset)

    rng = random.Random(a.seed)
    rhos, opt, top_overlap = [], [], []
    for _ in range(a.splits):
        shuffled = frames[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        A, B = shuffled[:half], shuffled[half:]
        va = [mean_on(m, A) for m in models]
        vb = [mean_on(m, B) for m in models]
        rhos.append(spearman(va, vb))

        idx = sorted(range(len(models)), key=lambda i: -va[i])[:a.topk]
        # what half B says about the models half A picked ...
        picked_on_b = sum(vb[i] for i in idx) / len(idx)
        # ... versus what half B would have picked for itself.
        honest = sorted(vb, reverse=True)[:a.topk]
        opt.append(sum(honest) / len(honest) - picked_on_b)

        best_b = set(sorted(range(len(models)), key=lambda i: -vb[i])[:a.topk])
        top_overlap.append(len(best_b & set(idx)) / a.topk)

    def summarise(v):
        v = sorted(v)
        return {"mean": statistics.fmean(v), "p05": v[int(0.05 * len(v))],
                "p95": v[int(0.95 * len(v))]}

    res = {"models": len(models), "frames": len(frames), "splits": a.splits,
           "topk": a.topk, "rank_stability_spearman": summarise(rhos),
           "selection_optimism_db": summarise(opt),
           "topk_overlap_frac": summarise(top_overlap)}

    print()
    print(f"half-vs-half rank stability (Spearman) : {res['rank_stability_spearman']['mean']:.3f}"
          f"  [{res['rank_stability_spearman']['p05']:.3f},"
          f" {res['rank_stability_spearman']['p95']:.3f}]")
    print(f"selection optimism, top-{a.topk} (dB)      : "
          f"{res['selection_optimism_db']['mean']:.3f}"
          f"  [{res['selection_optimism_db']['p05']:.3f},"
          f" {res['selection_optimism_db']['p95']:.3f}]")
    print(f"top-{a.topk} overlap between halves       : "
          f"{res['topk_overlap_frac']['mean']:.2f}")

    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()

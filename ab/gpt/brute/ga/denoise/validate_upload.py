"""Pre-commit validation of the denoise upload batches. Denoise only; nothing else is touched.

Checks, per model in a batch list:

  1. SOURCE EXISTS AND IS PYTHON -- ab/nn/nn/<name>.py present, parses, defines `Net`.
     A model that cannot be imported is worse than a missing one: it breaks any tool that walks
     the directory.
  2. EPOCH COVERAGE IS COMPLETE AND CONTIGUOUS -- a 5-epoch run must have 1..5, a 50-epoch run
     1..50, with no gaps. A gap means the training record is partial, and a partial curve cannot
     be used for the epoch-3 selection signal (rho +0.728) that all our model selection relies on.
     Expected length comes from the data itself (`epoch_max`), not from an assumption.
  3. EVERY EPOCH FILE IS A NON-EMPTY LIST OF RECORDS carrying a PSNR signal (`metric_psnr`, or
     `accuracy` as fallback). PSNR = accuracy * 48 (ab/nn/metric/psnr.py).
  4. MAX AND AVG PSNR ARE COMPUTABLE, and max is cross-checked against the `best_accuracy` the
     record claims. A model whose stats do not yield a usable maximum is not rankable.
  5. DEGENERACY -- max PSNR at or below the 33.021 dB no-op floor means the model never learned
     to beat returning the input unchanged. Reported, not auto-rejected: that is a judgement call.

The 147 stat dirs already published and since deleted locally are NOT considered -- they are gone
from disk, so they cannot appear in a batch, and they are deliberately left alone.

Usage:
    python validate_upload.py                 # all planned batches
    python validate_upload.py --batch 1
    python validate_upload.py --csv out.csv   # per-model max/avg PSNR table
"""
import argparse
import ast
import collections
import csv
import json
import os
import re
import sys

REPO = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset"
NN_DIR = os.path.join(REPO, "ab", "nn", "nn")
STAT_DIR = os.path.join(REPO, "ab", "nn", "stat", "train")
HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(HERE, "docs", "upload_batches")
NOOP_DB = 33.021
PSNR_SCALE = 48.0


def psnr_of(rec):
    for k in ("metric_psnr", "accuracy"):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return v * PSNR_SCALE
    return None


def check_source(name):
    p = os.path.join(NN_DIR, f"{name}.py")
    if not os.path.isfile(p):
        return "source missing"
    if not p.endswith(".py"):
        return "not a .py file"
    try:
        tree = ast.parse(open(p, encoding="utf8", errors="replace").read())
    except SyntaxError as e:
        return f"source SyntaxError line {e.lineno}"
    if not any(isinstance(n, ast.ClassDef) and n.name == "Net" for n in tree.body):
        return "source has no top-level class Net"
    return None


def check_stats(name, stat_dirs):
    """-> (problems, summary) ; summary has max/avg psnr and epoch counts per stat dir."""
    problems, summary = [], []
    for d in stat_dirs:
        full = os.path.join(STAT_DIR, d)
        if not os.path.isdir(full):
            problems.append(f"{d}: stat dir missing")
            continue
        files = [f for f in os.listdir(full) if f.endswith(".json")]
        if not files:
            problems.append(f"{d}: no .json epoch files")
            continue
        eps = {}
        for f in files:
            m = re.fullmatch(r"(\d+)\.json", f)
            if not m:
                problems.append(f"{d}: unexpected file {f}")
                continue
            eps[int(m.group(1))] = os.path.join(full, f)

        psnrs, claimed_max, epoch_max = [], None, None
        for e in sorted(eps):
            try:
                recs = json.load(open(eps[e]))
            except Exception as ex:
                problems.append(f"{d}/{e}.json: unreadable ({type(ex).__name__})")
                continue
            if not isinstance(recs, list) or not recs:
                problems.append(f"{d}/{e}.json: not a non-empty list")
                continue
            got = False
            for r in recs:
                if not isinstance(r, dict):
                    continue
                v = psnr_of(r)
                if v is not None:
                    psnrs.append(v)
                    got = True
                if isinstance(r.get("best_accuracy"), (int, float)):
                    b = r["best_accuracy"] * PSNR_SCALE
                    claimed_max = b if claimed_max is None else max(claimed_max, b)
                if isinstance(r.get("epoch_max"), int):
                    epoch_max = max(epoch_max or 0, r["epoch_max"])
            if not got:
                problems.append(f"{d}/{e}.json: no metric_psnr/accuracy in any record")

        # contiguity, against the length the data itself declares
        want = epoch_max if epoch_max else (max(eps) if eps else 0)
        missing = sorted(set(range(1, want + 1)) - set(eps))
        if missing:
            problems.append(f"{d}: missing epoch(s) {missing[:6]}"
                            f"{'...' if len(missing) > 6 else ''} of 1..{want}")
        extra = sorted(e for e in eps if e > want)
        if extra:
            problems.append(f"{d}: epoch file(s) beyond epoch_max={want}: {extra[:4]}")

        if not psnrs:
            problems.append(f"{d}: no PSNR values at all -- not rankable")
            continue
        mx, av = max(psnrs), sum(psnrs) / len(psnrs)
        if claimed_max is not None and abs(mx - claimed_max) > 0.01:
            problems.append(f"{d}: max PSNR {mx:.3f} disagrees with best_accuracy {claimed_max:.3f}")
        summary.append({"model": name, "stat_dir": d, "epochs": len(eps),
                        "epoch_max": want, "max_psnr": round(mx, 4),
                        "avg_psnr": round(av, 4),
                        "degenerate": mx <= NOOP_DB + 0.005})
    return problems, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    lists = sorted(os.listdir(BATCH_DIR)) if os.path.isdir(BATCH_DIR) else []
    if a.batch:
        lists = [f"batch_{a.batch:03d}.txt"]
    if not lists:
        sys.exit("no batch lists found -- run upload_batches.py --plan first")

    all_sum, fails = [], collections.defaultdict(list)
    for lf in lists:
        models, stats = [], collections.defaultdict(list)
        for line in open(os.path.join(BATCH_DIR, lf)):
            p = line.strip()
            if p.startswith("ab/nn/nn/"):
                models.append(os.path.basename(p)[:-3])
            elif p.startswith("ab/nn/stat/train/"):
                d = os.path.basename(p)
                m = re.search(r"_(psnr|psnrrgb|ssimrgb)_(.+)$", d)
                if m:
                    stats[m.group(2)].append(d)
        print(f"\n=== {lf}: {len(models)} models ===")
        nprob = 0
        for n in models:
            probs = []
            e = check_source(n)
            if e:
                probs.append(e)
            if n not in stats:
                probs.append("no stat dir in this batch list")
            else:
                p2, s2 = check_stats(n, stats[n])
                probs += p2
                all_sum += s2
            if probs:
                nprob += 1
                for p in probs:
                    fails[p.split(":")[-1].strip()[:60] if ":" in p else p].append(n)
                if nprob <= 6:
                    print(f"  PROBLEM {n[:44]}: {probs[0]}")
        print(f"  models with problems: {nprob}/{len(models)}")

    print(f"\n{'='*70}\nSUMMARY over {len(all_sum)} (model, stat-dir) pair(s)")
    if fails:
        print("\nproblem classes:")
        for k, v in sorted(fails.items(), key=lambda x: -len(x[1]))[:12]:
            print(f"  {len(v):>5}  {k}")
    else:
        print("\n  NO PROBLEMS FOUND")
    if all_sum:
        ec = collections.Counter(s["epochs"] for s in all_sum)
        print(f"\nepoch coverage: {dict(sorted(ec.items()))}")
        deg = [s for s in all_sum if s["degenerate"]]
        print(f"max PSNR: min {min(s['max_psnr'] for s in all_sum):.3f} / "
              f"max {max(s['max_psnr'] for s in all_sum):.3f}")
        print(f"avg PSNR: min {min(s['avg_psnr'] for s in all_sum):.3f} / "
              f"max {max(s['avg_psnr'] for s in all_sum):.3f}")
        print(f"at or below the {NOOP_DB} dB no-op floor: {len(deg)}")
    if a.csv and all_sum:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_sum[0].keys()))
            w.writeheader()
            w.writerows(all_sum)
        print(f"\nwrote {a.csv}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

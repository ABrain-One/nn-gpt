"""Build all_epochs.json + max/avg/std summaries from per-epoch curve files, after the fact.

ft50 was launched before the richer summary block existed, so its result records carry
best_dB/final_dB but not max/avg/std, and it writes no merged per-model file. The RAW data
is complete either way -- one <epoch>.json per epoch with 27 fields -- so the aggregates are
recoverable without touching a running job. That is the point of this script: ft50 is
mid-flight across 8 pods and rewriting its runner would leave the sweep behaving as neither
the old nor the new version.

Safe to re-run at any time, including while the sweep is still writing: it only reads
<epoch>.json files and writes all_epochs.json + a summary table beside them.

  python aggregate_curves.py --dir /a/mm/sweep_out/fulltrain50/curves
"""
import argparse
import glob
import json
import os


def summarise(model_dir):
    per = {}
    for f in glob.glob(os.path.join(model_dir, "*.json")):
        base = os.path.basename(f)[:-5]
        if not base.isdigit():
            continue                              # skip all_epochs.json itself
        try:
            trials = json.load(open(f))
        except Exception:
            continue
        if isinstance(trials, list) and trials:
            per[int(base)] = max(trials, key=lambda d: d.get("accuracy") or 0)
    if not per:
        return None
    eps = sorted(per)
    db = [(per[e].get("accuracy") or 0) * 48 for e in eps]
    tail = db[-10:] if len(db) >= 10 else db
    mean = sum(db) / len(db)
    bi = max(range(len(db)), key=lambda i: db[i])
    return {
        "n_epochs": len(db),
        "max_dB": round(db[bi], 3),
        "best_epoch": eps[bi],
        "final_dB": round(db[-1], 3),
        "avg_dB": round(mean, 3),
        "avg_last10_dB": round(sum(tail) / len(tail), 3),
        "std_dB": round((sum((v - mean) ** 2 for v in db) / len(db)) ** 0.5, 3),
        "min_dB": round(min(db), 3),
        "curve": [[e, round(d, 3)] for e, d in zip(eps, db)],
        "_per_epoch": {str(e): per[e] for e in eps},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the curves/ directory")
    ap.add_argument("--bar", type=float, default=36.714,
                    help="reference PSNR to compare against (corpus best at 50 epochs)")
    ap.add_argument("--out", default=None, help="summary table json (default: <dir>/../summary.json)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.dir.rstrip("/")), "summary.json")

    rows = []
    for d in sorted(glob.glob(os.path.join(a.dir, "*"))):
        if not os.path.isdir(d):
            continue
        s = summarise(d)
        if not s:
            continue
        name = os.path.basename(d)
        per = s.pop("_per_epoch")
        with open(os.path.join(d, "all_epochs.json"), "w") as fh:
            json.dump({"model": name, "epochs": per,
                       "summary": {k: v for k, v in s.items() if k != "curve"}}, fh, indent=1)
        rows.append(dict(model=name, **s))

    rows.sort(key=lambda r: -r["max_dB"])
    beat = [r for r in rows if r["max_dB"] > a.bar]
    print(f"{len(rows)} models aggregated; {len(beat)} exceed {a.bar} dB\n")
    print(f"{'max':>7s} {'avg':>7s} {'last10':>7s} {'final':>7s} {'ep':>4s} {'n':>4s}  model")
    for r in rows[:25]:
        flag = "  <-- beats bar" if r["max_dB"] > a.bar else ""
        print(f"{r['max_dB']:7.2f} {r['avg_dB']:7.2f} {r['avg_last10_dB']:7.2f} "
              f"{r['final_dB']:7.2f} {r['best_epoch']:4d} {r['n_epochs']:4d}  "
              f"{r['model'][:44]}{flag}")
    json.dump({"bar": a.bar, "n": len(rows), "beat_bar": len(beat), "models": rows},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}  (+ all_epochs.json in each model dir)")


if __name__ == "__main__":
    main()

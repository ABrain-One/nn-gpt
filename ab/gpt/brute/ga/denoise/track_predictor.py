"""Track the accuracy predictor's LIVE performance on the production run.

The predictor was trained on the ablation archs (arms A/B/C/D). Production explores NEW
territory (Lite/3F/4F/8F mutants not in its training set), so this measures whether the
predictor's ranking/accuracy actually holds out-of-distribution. It re-scores every trained
production model (the predictor is deterministic, seeded from source hash → same score the
funnel used) and compares to the real ep5 PSNR in the cache.

Reports, for the models that were actually TRAINED (we can't see actuals for skipped ones):
  - regressor: Spearman rank-corr + MAE of predicted vs actual PSNR (viable, psnr>0)
  - gate: did p_dead flag the diverged (0.0) models?
Saves predictor_tracking.json (+ scatter) and appends a timestamped row to
predictor_tracking_history.csv so you can watch it evolve as the run grows.

Run (CPU-only, safe alongside the GPU run):
    cd /a/mm/nn-dataset
    PYTHONPATH=. python .../track_predictor.py
"""
import csv, glob, json, os, sys, time, math

# Be a LIGHT CPU citizen: this runs alongside the GPU production run (12 dataloader
# workers). Torch defaults to one thread per core (16) per op, which oversubscribes the
# box and steals from the run. Single-threaded is plenty for zero-cost proxies on tiny models.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "predictor"))
sys.path.insert(0, "/a/mm/nn-dataset")

MODELS_DIR = os.path.join(_HERE, "seed_ga_P1_models")
CACHE = os.path.join(_HERE, "seed_ga_P1_cache.json")
PSNR_SCALE = 48.0


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else None


def main():
    import predict
    cache = json.load(open(CACHE))
    rows = []           # (checksum, actual_psnr, pred_psnr, p_dead, edge_reject)
    scored = 0
    for fp in glob.glob(os.path.join(MODELS_DIR, "*.py")):
        cks = os.path.basename(fp)[:-3]
        rec = cache.get(cks)
        if not rec:
            continue
        src = open(fp).read()
        try:
            s = predict.score(src, rec.get("macs"), rec.get("params"), rec.get("name"))
        except Exception:
            continue
        rows.append(dict(checksum=cks, name=rec.get("name"),
                         actual=rec.get("psnr") or 0.0,
                         pred=s["psnr_pred"], p_dead=s["p_dead"],
                         edge_reject=bool(rec.get("edge_reject"))))
        scored += 1

    viable = [r for r in rows if r["actual"] > 0 and not r["edge_reject"]]
    dead = [r for r in rows if r["actual"] <= 0 and not r["edge_reject"]]
    xs = [r["pred"] for r in viable]
    ys = [r["actual"] for r in viable]
    rho = _spearman(xs, ys)
    mae = (sum(abs(r["pred"] - r["actual"]) for r in viable) / len(viable)) if viable else None
    # gate check: mean p_dead on diverged vs viable (should be higher on diverged)
    pdead_dead = (sum(r["p_dead"] for r in dead) / len(dead)) if dead else None
    pdead_viable = (sum(r["p_dead"] for r in viable) / len(viable)) if viable else None

    out = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), scored=scored,
               viable=len(viable), diverged=len(dead),
               spearman_pred_vs_actual=(round(rho, 3) if rho is not None else None),
               mae_db=(round(mae, 3) if mae is not None else None),
               mean_pdead_on_diverged=(round(pdead_dead, 3) if pdead_dead is not None else None),
               mean_pdead_on_viable=(round(pdead_viable, 3) if pdead_viable is not None else None))
    json.dump({**out, "rows": rows}, open(os.path.join(_HERE, "predictor_tracking.json"), "w"), indent=2)

    hist = os.path.join(_HERE, "predictor_tracking_history.csv")
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        if new:
            w.writeheader()
        w.writerow(out)

    print("=== accuracy-predictor tracking (live, on production models) ===")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  -> predictor_tracking.json + predictor_tracking_history.csv")

    # optional scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(xs, ys, s=14, alpha=0.6)
        lo, hi = min(xs + ys), max(xs + ys)
        plt.plot([lo, hi], [lo, hi], "r--", lw=1)
        plt.xlabel("predicted ep5 PSNR (dB)")
        plt.ylabel("actual ep5 PSNR (dB)")
        plt.title(f"predictor vs actual (n={len(viable)}, ρ={rho:.2f})" if rho else "predictor vs actual")
        plt.tight_layout()
        plt.savefig(os.path.join(_HERE, "predictor_tracking_scatter.png"), dpi=110)
        print("  -> predictor_tracking_scatter.png")
    except Exception as e:
        print(f"  (scatter skipped: {e})")


if __name__ == "__main__":
    main()

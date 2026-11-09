"""Multi-winner SIDD generalization sweep. For each Gate 1 winner in the manifest,
train the architecture from scratch on SIDD @50ep (same recipe as MAI) and record
SIDD PSNR + delta vs MAI. Resumable: skips models already in the results file.

Run from nn-dataset root (sidd loader resolves sidd/ relative to CWD):
    cd /a/mm/nn-dataset
    PYTHONPATH=. python <this> --manifest .../sidd_sweep_manifest.json
"""
import argparse, json, os, sys, time

sys.path.insert(0, "/a/mm/nn-dataset")
import ab.nn.api as api

TASK, METRIC, PSNR_SCALE = "img-denoising", "psnr", 48.0
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "sidd_sweep_manifest.json"))
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "sidd_sweep_results.json"))
    # seed the sweep with the already-completed A0 9F+m+m so the final table is whole
    ap.add_argument("--seed-done", default="")
    args = ap.parse_args()

    models = json.load(open(args.manifest))
    results = {}
    if os.path.exists(args.out):
        results = {r["name"]: r for r in json.load(open(args.out))}
    if args.seed_done and os.path.exists(args.seed_done):
        prev = json.load(open(args.seed_done))
        results.setdefault(prev["name"], {**prev, "arm": "A0", "deployable": True})

    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.workers}

    for i, m in enumerate(models, 1):
        name = m["name"]
        if name in results and results[name].get("sidd_psnr", 0) > 0:
            print(f"[{i}/{len(models)}] {name} — cached ({results[name]['sidd_psnr']} dB)", flush=True)
            continue
        src = open(m["src"]).read()
        print(f"[{i}/{len(models)}] {name} ({m['arm']}, MAI {m['mai']:.2f}) — SIDD @{args.epoch}ep ...", flush=True)
        t0 = time.time(); psnr = 0.0; err = None
        try:
            _n, acc, _a2t, _q = api.check_nn(src, TASK, "sidd", METRIC, prm,
                                             save_to_db=False, prefix="SiddSweep")
            psnr = acc * PSNR_SCALE
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            print("   FAILED:", err, flush=True)
        rec = dict(name=name, arm=m["arm"], deployable=m.get("deployable"),
                   mai_psnr=m["mai"], sidd_psnr=round(psnr, 3),
                   delta=round(psnr - m["mai"], 3), train_sec=round(time.time() - t0), error=err)
        results[name] = rec
        json.dump(list(results.values()), open(args.out, "w"), indent=2)
        print(f"   -> SIDD {psnr:.3f} dB  (Δ{rec['delta']:+.2f} vs MAI)  [{rec['train_sec']}s]", flush=True)

    # final table
    print("\n=== SIDD SWEEP RESULT ===", flush=True)
    rows = sorted(results.values(), key=lambda r: -(r.get("sidd_psnr") or 0))
    print(f"{'arm':4}{'model':40}{'MAI':>7}{'SIDD':>8}{'Δ':>7}{'dep':>5}")
    for r in rows:
        print(f"{r['arm']:4}{r['name'][:39]:40}{(r.get('mai_psnr') or 0):7.2f}"
              f"{(r.get('sidd_psnr') or 0):8.2f}{(r.get('delta') or 0):+7.2f}"
              f"{('Y' if r.get('deployable') else 'n'):>5}")
    json.dump(list(results.values()), open(args.out, "w"), indent=2)
    print("-> " + args.out, flush=True)


if __name__ == "__main__":
    main()

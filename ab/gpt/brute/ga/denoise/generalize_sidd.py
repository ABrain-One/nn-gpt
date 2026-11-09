"""Cross-dataset generalization check: train a Gate 1 winner ARCHITECTURE on SIDD
from scratch (same recipe as MAI full-train) and report PSNR, to compare against its
MAI PSNR. Tests whether the searched architecture is dataset-robust or MAI-overfit.

Run from the nn-dataset root (the sidd loader resolves sidd/ relative to CWD):
    cd /a/mm/nn-dataset
    PYTHONPATH=. python <this> --src <winner.py> --name 9F+m+m --epoch 50
"""
import argparse, json, os, sys, time

sys.path.insert(0, "/a/mm/nn-dataset")
import ab.nn.api as api

TASK, METRIC, PSNR_SCALE = "img-denoising", "psnr", 48.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to the winner architecture .py")
    ap.add_argument("--name", default="winner")
    ap.add_argument("--dataset", default="sidd")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mai-psnr", type=float, default=None, help="the model's MAI PSNR, for the delta")
    ap.add_argument("--out", default="generalize_sidd_results.json")
    args = ap.parse_args()

    src = open(args.src).read()
    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.workers}

    print(f"=== SIDD generalization: {args.name} on '{args.dataset}' @{args.epoch}ep ===", flush=True)
    t0 = time.time()
    psnr = 0.0; err = None
    try:
        _n, acc, _a2t, _q = api.check_nn(src, TASK, args.dataset, METRIC, prm,
                                         save_to_db=False, prefix="SiddGen")
        psnr = acc * PSNR_SCALE
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
        print("  FAILED:", err, flush=True)
    dt = time.time() - t0

    rec = {"name": args.name, "dataset": args.dataset, "epoch": args.epoch,
           "sidd_psnr": round(psnr, 3), "mai_psnr": args.mai_psnr,
           "delta": (round(psnr - args.mai_psnr, 3) if args.mai_psnr else None),
           "train_sec": round(dt), "error": err}
    print("=== RESULT ===", flush=True)
    print(json.dumps(rec, indent=2), flush=True)
    json.dump(rec, open(args.out, "w"), indent=2)
    print("-> " + args.out, flush=True)


if __name__ == "__main__":
    main()

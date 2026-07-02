"""Denoise NAS search driver.

Reads candidate model `.py` files produced by the brute mutator
(`config.MUTATED_MODELS_OUTPUT_ROOT/<seed>/*.py`), evaluates each via the
nn-dataset training hook `ab.nn.api.check_nn`, and ranks them by PSNR.

Two modes:
  proxy : short-epoch screen of ALL candidates (cheap), save_to_db=False, writes a
          ranked results JSON.
  full  : re-train the top-K from a results JSON at full epochs, save_to_db=True
          (persists discovered winners to the LEMUR DB).

PSNR note: stored accuracy x48 = real dB (ab/nn/metric/psnr.py).
Baselines: best manual F = 35.56 dB (DenoiseUnet7F); LiteDenoiseNet = 37.66 dB (MAI2026).

Run from this dir with nn-dataset on PYTHONPATH, e.g.:
  PYTHONPATH=/home/s493053/thesis/nn-dataset \
    /home/s493053/thesis/nn-dataset/.venv/bin/python denoise_search.py proxy --epoch 4 --limit 40
  ... full --topk 3 --epoch 50 --results denoise_search_results.json
"""
import argparse
import glob
import json
import os
import time
import traceback

# LEMUR root is derived from the CWD at ab.nn import time (ab/nn/util/Const.py walks up
# from cwd to a dir with ab/+README.md). Force CWD to the nn-dataset repo BEFORE importing
# ab.nn so we always hit the populated DB, regardless of where this script is invoked from.
_NN_DATASET_ROOT = os.environ.get("NN_DATASET_ROOT", "/home/s493053/thesis/nn-dataset")
os.chdir(_NN_DATASET_ROOT)

import ab.nn.api as api

from mutator import config

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0  # stored accuracy * 48 = real dB
BASELINES = {"DenoiseUnet7F (best F)": 35.56, "LiteDenoiseNet (MAI2026)": 37.66}
DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "denoise_search_results.json")


def find_candidates(root, limit=None):
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    return files[:limit] if limit else files


def evaluate(code, epoch, batch, lr, save_to_db, num_workers=8):
    prm = {"lr": lr, "batch": batch, "transform": "denoise", "epoch": epoch,
           "num_workers": num_workers}
    name, acc, acc_to_time, code_quality = api.check_nn(
        code, TASK, DATASET, METRIC, prm,
        save_to_db=save_to_db, prefix="DnSearch",
    )
    return {"db_name": name, "accuracy": acc, "psnr_db": acc * PSNR_SCALE,
            "acc_to_time": acc_to_time, "code_quality": code_quality}


def run_proxy(args):
    cands = find_candidates(args.candidates_dir, args.limit)
    print(f"Found {len(cands)} candidate(s) under {args.candidates_dir}")
    if not cands:
        print("No candidates — run the mutator first.")
        return
    results = []
    for i, path in enumerate(cands, 1):
        rel = os.path.relpath(path, args.candidates_dir)
        print(f"\n[{i}/{len(cands)}] proxy-eval ({args.epoch} ep): {rel}")
        code = open(path, encoding="utf-8").read()
        t0 = time.time()
        try:
            r = evaluate(code, args.epoch, args.batch, args.lr, save_to_db=False, num_workers=args.num_workers)
            r.update({"path": path, "rel": rel, "status": "ok", "secs": round(time.time() - t0, 1)})
            print(f"    -> PSNR {r['psnr_db']:.2f} dB (acc {r['accuracy']:.4f}) in {r['secs']}s")
        except Exception as e:
            r = {"path": path, "rel": rel, "status": "fail",
                 "error": f"{type(e).__name__}: {e}", "psnr_db": None,
                 "secs": round(time.time() - t0, 1)}
            print(f"    -> FAILED: {r['error']}")
            if args.verbose:
                traceback.print_exc()
        results.append(r)

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: r["psnr_db"], reverse=True)
    json.dump({"mode": "proxy", "epoch": args.epoch, "results": results, "ranked_ok": ok},
              open(args.results, "w"), indent=2)
    print(f"\n{'='*60}\nProxy screen: {len(ok)}/{len(results)} trained OK. Results -> {args.results}")
    _print_ranking(ok)


def run_full(args):
    data = json.load(open(args.results))
    ok = data.get("ranked_ok") or [r for r in data["results"] if r["status"] == "ok"]
    ok = sorted(ok, key=lambda r: r["psnr_db"], reverse=True)[:args.topk]
    print(f"Full-train top-{len(ok)} at {args.epoch} epochs (save_to_db=True)")
    out = []
    for i, r in enumerate(ok, 1):
        print(f"\n[{i}/{len(ok)}] full-train: {r['rel']}  (proxy {r['psnr_db']:.2f} dB)")
        code = open(r["path"], encoding="utf-8").read()
        try:
            full = evaluate(code, args.epoch, args.batch, args.lr, save_to_db=True, num_workers=args.num_workers)
            full.update({"rel": r["rel"], "proxy_psnr_db": r["psnr_db"], "status": "ok"})
            print(f"    -> FULL PSNR {full['psnr_db']:.2f} dB (saved as {full['db_name']})")
        except Exception as e:
            full = {"rel": r["rel"], "status": "fail", "error": f"{type(e).__name__}: {e}"}
            print(f"    -> FAILED: {full['error']}")
        out.append(full)
    json.dump({"mode": "full", "epoch": args.epoch, "results": out},
              open(args.full_results, "w"), indent=2)
    print(f"\n{'='*60}\nFull-train done. Results -> {args.full_results}")
    _print_ranking([r for r in out if r.get("status") == "ok"])


def _print_ranking(ok):
    if not ok:
        print("(no successful runs to rank)")
        return
    print(f"\n{'rank':<5}{'PSNR dB':<10}{'candidate'}")
    for i, r in enumerate(ok, 1):
        print(f"{i:<5}{r['psnr_db']:<10.2f}{r['rel']}")
    print("\nBaselines:")
    for name, db in BASELINES.items():
        print(f"  {db:.2f} dB  {name}")
    best = ok[0]["psnr_db"]
    tag = "BEATS" if best > BASELINES["DenoiseUnet7F (best F)"] else "below"
    print(f"\nBest discovered {best:.2f} dB {tag} best manual F (35.56 dB).")


def main():
    p = argparse.ArgumentParser(description="Denoise NAS search driver")
    sub = p.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("proxy", help="short-epoch screen of all candidates")
    pp.add_argument("--candidates-dir", default=config.MUTATED_MODELS_OUTPUT_ROOT)
    pp.add_argument("--epoch", type=int, default=4)
    pp.add_argument("--batch", type=int, default=16)
    pp.add_argument("--lr", type=float, default=1e-4)
    pp.add_argument("--limit", type=int, default=None)
    pp.add_argument("--results", default=DEFAULT_RESULTS)
    pp.add_argument("--num-workers", type=int, default=8)
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(func=run_proxy)

    pf = sub.add_parser("full", help="full-train top-K from a results JSON")
    pf.add_argument("--results", default=DEFAULT_RESULTS)
    pf.add_argument("--full-results", default=os.path.join(os.path.dirname(__file__), "denoise_full_results.json"))
    pf.add_argument("--topk", type=int, default=3)
    pf.add_argument("--epoch", type=int, default=50)
    pf.add_argument("--batch", type=int, default=16)
    pf.add_argument("--lr", type=float, default=1e-4)
    pf.add_argument("--num-workers", type=int, default=8)
    pf.set_defaults(func=run_full)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

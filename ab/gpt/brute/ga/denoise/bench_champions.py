"""Standard-benchmark head-to-head for the top MAI/Lemur champions (professor directive
2026-07-25): full-train each champion FROM SCRATCH on a standard denoising benchmark and
report PSNR + params + MACs, so the pipeline's models can be placed against the published
efficient-NAS-denoising frontier.

Benchmarks (dataset name = ab/nn/loader/<name>.py):
    sidd        real smartphone noise (SIDD-Medium, internal 288/32 split)
    awgn_color  synthetic AWGN sigma25, BSD400 clean train -> CBSD68 color test

One process per benchmark so the AWGN loader's module-level env (AWGN_TRAIN/TEST/SIGMA)
is fixed once before import. Pin the GPU with CUDA_VISIBLE_DEVICES outside.

Usage (from nn-dataset root, so sidd/ and data/awgn resolve):
    cd /a/mm/nn-dataset
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python <this> --benchmark sidd --epoch 50
    CUDA_VISIBLE_DEVICES=1 AWGN_TRAIN=bsd400_train AWGN_TEST=cbsd68_color AWGN_SIGMA=25 \
        PYTHONPATH=. .venv/bin/python <this> --benchmark awgn_color --epoch 50
"""
import argparse, json, os, time

# AWGN loader reads these at import; default to the verified canonical color config.
if os.environ.get("BENCH") != "no-awgn-defaults":
    os.environ.setdefault("AWGN_TRAIN", "bsd400_train")
    os.environ.setdefault("AWGN_TEST", "cbsd68_color")
    os.environ.setdefault("AWGN_SIGMA", "25")

import torch
import ab.nn.api as api

CHAMP_DIR = "/tmp/claude-224207/champions"
CHAMPIONS = ["DenoiseUnet7", "LiteDenoiseNet", "DenoiseUnet7F", "DnSearch-129935", "DnSkip-61f5c5"]
MAI_PSNR = {"DenoiseUnet7": 37.52, "LiteDenoiseNet": 36.51, "DenoiseUnet7F": 35.85,
            "DnSearch-129935": 35.35, "DnSkip-61f5c5": 35.09}
SCALE = 48.0
HERE = os.path.dirname(os.path.abspath(__file__))


def cost(src):
    """params (M) and conv/linear MACs (MMac) for a 3x256x256 input, static — no training."""
    try:
        from ab.gpt.brute.ga.denoise import edge_cost as EC
        r = EC.measure_cost(src)
        if not r.get("ok"):
            return None, None
        return round(r["params"] / 1e6, 3), round(r["macs"] / 1e6, 1)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    # "awgn-color" matches the loader file ab/nn/loader/awgn-color.py; the dataset
    # name must stay free of '_' or the stat dir it produces cannot be parsed.
    ap.add_argument("--benchmark", required=True, choices=["sidd", "awgn-color"])
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--champions", nargs="*", default=CHAMPIONS)
    ap.add_argument("--force", action="store_true", help="re-run champions that already have a result")
    args = ap.parse_args()

    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.num_workers}
    suffix = os.environ.get("BENCH_SUFFIX", "")   # per-process output so parallel runs don't race
    resf = os.path.join(HERE, f"bench_{args.benchmark}{suffix}_results.json")
    out = {}
    if os.path.exists(resf):
        try: out = json.load(open(resf))
        except Exception: out = {}

    cfg = (f"train={os.environ.get('AWGN_TRAIN')} test={os.environ.get('AWGN_TEST')} "
           f"sigma={os.environ.get('AWGN_SIGMA')}") if args.benchmark == "awgn-color" else "SIDD 288/32"
    print(f"=== {args.benchmark} head-to-head | {cfg} | {args.epoch}ep | "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} ===", flush=True)

    prefix = "Bench" + ("S" if args.benchmark == "sidd" else "A")  # distinct tmp prefix per benchmark
    for name in args.champions:
        if not args.force and out.get(name, {}).get("psnr_db") is not None:
            print(f"  {name:18} already done ({out[name]['psnr_db']} dB) -> skip", flush=True)
            continue
        src = open(os.path.join(CHAMP_DIR, name + ".py")).read()
        params, macs = cost(src)
        t0 = time.time()
        rec = {"mai_psnr": MAI_PSNR.get(name), "params_M": params, "macs_MMac": macs,
               "epoch": args.epoch}
        try:
            _, acc, _, _ = api.check_nn(src, "img-denoising", args.benchmark, "psnr", prm,
                                        save_to_db=False, prefix=prefix)
            rec["psnr_db"] = round(acc * SCALE, 3)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["sec"] = round(time.time() - t0)
        out[name] = rec
        json.dump(out, open(resf, "w"), indent=2)
        mai = rec['mai_psnr'] if rec['mai_psnr'] is not None else "-"
        print(f"  {name:18} MAI {mai} -> {args.benchmark} "
              f"{rec.get('psnr_db', rec.get('error'))}  "
              f"({params}M, {macs} MMac, {rec['sec']}s)", flush=True)
    print(f"DONE {args.benchmark} -> {resf}", flush=True)


if __name__ == "__main__":
    main()

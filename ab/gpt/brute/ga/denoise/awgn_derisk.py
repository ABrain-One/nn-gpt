"""SOTA de-risk: train champion architectures on AWGN (color, sigma from env) and
evaluate on a standard test set (BSD100-color), to see if our models are anywhere near
the published efficient-denoising frontier. Trains via the SAME check_nn path as the GA.

Env: AWGN_TRAIN, AWGN_TEST, AWGN_SIGMA set here before importing the loader.
Usage: awgn_derisk.py --epoch 40 --models LiteDenoiseNet DenoiseUnet7 DenoiseNAFNet DenoiseRestormerLite
"""
import argparse, json, os, time

os.environ.setdefault("AWGN_TRAIN", "train_clean/denoised")
os.environ.setdefault("AWGN_TEST", "bsd100_color/BSD100_HR")
os.environ.setdefault("AWGN_SIGMA", "25")

import ab.nn.api as api

NN = "/a/mm/nn-dataset/ab/nn/nn"
SCALE = 48.0
RESULTS = "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise/awgn_derisk_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["LiteDenoiseNet", "DenoiseUnet7", "DenoiseNAFNet", "DenoiseRestormerLite"])
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.num_workers}
    out = {}
    if os.path.exists(RESULTS):
        try: out = json.load(open(RESULTS))
        except Exception: out = {}
    print(f"AWGN de-risk | sigma={os.environ['AWGN_SIGMA']} test={os.environ['AWGN_TEST']} "
          f"epoch={args.epoch}", flush=True)
    for name in args.models:
        code = open(os.path.join(NN, name + ".py")).read()
        t0 = time.time()
        try:
            _, acc, _, _ = api.check_nn(code, "img-denoising", "awgn-color", "psnr", prm,
                                        save_to_db=False, prefix="DnAWGN")
            out[name] = {"psnr_db": round(acc * SCALE, 3), "epoch": args.epoch,
                         "sigma": int(os.environ["AWGN_SIGMA"]), "sec": round(time.time() - t0)}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"  {name:24} -> {out[name].get('psnr_db', out[name].get('error'))}", flush=True)
        json.dump(out, open(RESULTS, "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

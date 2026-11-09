"""Train ONE architecture on a dataset via check_nn and report PSNR + cost. For the
single-model beat-SplitterNet attempt (not a search). Usage:
  cd /a/mm/nn-dataset
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:<nn-gpt>:<ast> python train_one.py \
      --src <path.py> --name DenoiseSplitterNetL --dataset denoise --epoch 50
"""
import argparse, json, os, time
import ab.nn.api as api

SCALE = 48.0
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="denoise")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--metric", default="psnr", help="psnr (Y-channel) or psnr_rgb (paper-comparable)")
    ap.add_argument("--out", default=os.path.join(HERE, "beat_attempt_results.json"))
    args = ap.parse_args()

    src = open(args.src).read()
    params = macs = None
    try:
        from ab.gpt.brute.ga.denoise import edge_cost as EC
        c = EC.measure_cost(src)
        if c["ok"]:
            params, macs = c["params"] / 1e6, c["macs"] / 1e6
    except Exception:
        pass
    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.num_workers}
    print(f"=== train {args.name} on '{args.dataset}' {args.epoch}ep | "
          f"{params}M {macs} MMac | gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} ===", flush=True)
    t0 = time.time()
    rec = {"name": args.name, "dataset": args.dataset, "epoch": args.epoch,
           "metric": args.metric, "params_M": params, "macs_MMac": macs}
    try:
        _, acc, _, _ = api.check_nn(src, "img-denoising", args.dataset, args.metric, prm,
                                    save_to_db=False, prefix="Beat")
        rec["psnr_db"] = round(acc * SCALE, 3)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    rec["sec"] = round(time.time() - t0)
    out = {}
    if os.path.exists(args.out):
        try: out = json.load(open(args.out))
        except Exception: out = {}
    out[args.name] = rec
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"RESULT {args.name}: {rec.get('psnr_db', rec.get('error'))} dB "
          f"({params}M, {macs} MMac, {rec['sec']}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

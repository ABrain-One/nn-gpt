"""Do our models actually DENOISE, and do they do it on a whole image?

TWO QUESTIONS THE STORED METRIC CANNOT ANSWER, both raised by measurement rather than
suspicion.

(1) IS 33.7 dB GOOD? Nothing in the corpus records what the NOISY INPUT ITSELF scores. A
denoiser that returns its input unchanged still gets a PSNR, and if that no-op score is
close to 33.7 then a corpus median of 33.7 means the models are barely denoising. This came
up because the error-correlation control found that a do-nothing identity baseline
correlates with our TRAINED models at 0.930 -- higher than trained models correlate with
each other (0.923). That is what it looks like when residuals are dominated by noise the
models did not remove. The gain over no-op is the number that settles it, and it has never
been computed.

(2) DOES IT HOLD ON A FULL IMAGE? Validation images are 3200x2432 (7.8 MP). The loader
evaluates a single 256x256 CENTRE CROP -- 65,536 px, 0.84% of one image. Every stored PSNR
in the corpus is that crop. Centre crops of photographs are systematically flatter than
frame edges (subjects are centred, vignetting and detail live at the border), so a model
tuned on centre crops can score well and still fail on the frame it would actually be given.

WHAT IS MEASURED HERE, per model:
    psnr_crop   the stored metric, recomputed -- proves this harness agrees with the DB
    psnr_full   the same model over the ENTIRE image, tiled
    psnr_noop   the noisy input scored against clean, on the same region
    gain        psnr_model - psnr_noop, the only number that says "it denoised"

TILING, and why overlap is not optional. A 3200x2432 forward is ~120x the pixels the model
was built for; tiling is required for memory. But butt-jointed tiles produce seams, because
convolutions at a tile border see zero-padding instead of neighbouring pixels, and the seam
lands in the PSNR as the model's fault. Tiles overlap by `--overlap` and only the interior
is kept, so every scored pixel had real context on all sides.

  python fullimage_eval.py --top 12 --tile 256 --overlap 32
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "img-denoising"
IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def y_mse(a, b):
    """Mean squared error on the ITU-R BT.601 LUMINANCE channel, matching ab/nn/metric/psnr.py.

    NOT RGB. The stored metric converts to Y = 0.299R + 0.587G + 0.114B and takes the MSE
    there. Noise that is independent across channels partly cancels in that weighted sum, so
    Y-PSNR runs several dB above RGB-PSNR on the same image. Scoring RGB here and comparing
    against a Y-channel number in the database is not a measurement, it is two different
    quantities in one column -- it made these models look ~7 dB worse than their stored
    score and nearly produced a false "the models barely denoise" conclusion.

    float64: at 35 dB the MSE is ~3e-4, and float32 accumulation over 7.8 M pixels loses
    enough precision to move the last digit.
    """
    a, b = a.double().clamp(0, 1), b.double()
    if a.shape[-3] == 3:
        w = (0.299, 0.587, 0.114)
        ya = w[0] * a[..., 0, :, :] + w[1] * a[..., 1, :, :] + w[2] * a[..., 2, :, :]
        yb = w[0] * b[..., 0, :, :] + w[1] * b[..., 1, :, :] + w[2] * b[..., 2, :, :]
    else:
        ya, yb = a[..., 0, :, :], b[..., 0, :, :]
    return float(((ya - yb) ** 2).mean())


def psnr_from_mses(mses):
    """dB of the MEAN MSE -- the metric averages MSE over samples and converts ONCE.

    Averaging per-image dB instead is a different statistic: -log is concave, so the mean of
    the dB values sits above the dB of the mean MSE. Small next to the RGB/Y error, but it
    is the difference between reproducing the stored number and merely being near it.
    """
    import math
    m = sum(mses) / len(mses)
    return float("inf") if m <= 0 else 10.0 * math.log10(1.0 / m)


def tiled_forward(net, x, tile, overlap, device):
    """Run `net` over a large image in overlapping tiles, keeping only tile interiors."""
    import torch
    _b, _c, H, W = x.shape
    step = tile - 2 * overlap
    out = torch.zeros_like(x)
    cnt = torch.zeros((1, 1, H, W), device=x.device)
    for top in range(0, H, step):
        for left in range(0, W, step):
            t0, l0 = min(top, max(0, H - tile)), min(left, max(0, W - tile))
            patch = x[:, :, t0:t0 + tile, l0:l0 + tile]
            if patch.shape[-2] < tile or patch.shape[-1] < tile:
                continue
            with torch.no_grad():
                p = net(patch)
            # Keep the interior only: border pixels saw zero-padding, not real neighbours.
            oy = overlap if t0 > 0 else 0
            ox = overlap if l0 > 0 else 0
            ey = tile - (overlap if t0 + tile < H else 0)
            ex = tile - (overlap if l0 + tile < W else 0)
            out[:, :, t0 + oy:t0 + ey, l0 + ox:l0 + ex] += p[:, :, oy:ey, ox:ex]
            cnt[:, :, t0 + oy:t0 + ey, l0 + ox:l0 + ex] += 1
    return out / cnt.clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--ckpt", default="/a/mm/sweep_out/ckpt")
    ap.add_argument("--val", default="/a/mm/nn-dataset/validation")
    ap.add_argument("--top", type=int, default=12, help="models, by stored PSNR")
    ap.add_argument("--names-file", default=None,
                    help="evaluate exactly these models (one name per line) instead of --top")
    ap.add_argument("--stratify", type=int, default=0,
                    help="with --names-file, keep this many models spread evenly across the "
                         "stored-PSNR range rather than the first N")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/fullimage_eval.json")
    a = ap.parse_args()

    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== full-image evaluation | device={dev} ===", flush=True)

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    best = {n: (v or 0) * 48.0 for n, v in con.execute(
        "SELECT nn, MAX(accuracy) FROM stat WHERE task=? AND accuracy IS NOT NULL "
        "GROUP BY nn", (TASK,))}
    code = dict(con.execute("SELECT name, code FROM nn"))
    con.close()

    have = [n for n in code
            if os.path.exists(os.path.join(a.ckpt, n, "best_model.pth"))]
    if a.names_file:
        want = [l.strip() for l in open(a.names_file) if l.strip()]
        have = [n for n in want if n in set(have)]
        have.sort(key=lambda n: -best.get(n, 0))
        # STRATIFIED, not the top N. Validating a threshold means sampling ACROSS it: taking
        # the strongest models would put every sample on one side of the boundary and could
        # only ever confirm agreement.
        if a.stratify and len(have) > a.stratify:
            step = len(have) / a.stratify
            have = [have[int(i * step)] for i in range(a.stratify)]
        print(f"  {len(have)} models from {a.names_file}"
              + (f" (stratified across the PSNR range)" if a.stratify else ""))
    else:
        have.sort(key=lambda n: -best.get(n, 0))
        have = have[:a.top]
        print(f"  {len(have)} models (top by stored PSNR, weights present)")

    noisy_d = os.path.join(a.val, "original")
    clean_d = os.path.join(a.val, "denoised")
    nf = sorted(f for f in os.listdir(noisy_d) if f.lower().endswith((".png", ".jpg")))
    cf = sorted(f for f in os.listdir(clean_d) if f.lower().endswith((".png", ".jpg")))
    nf, cf = nf[:a.images], cf[:a.images]

    pairs = []
    for n, c in zip(nf, cf):
        x = TF.to_tensor(Image.open(os.path.join(noisy_d, n)).convert("RGB"))
        y = TF.to_tensor(Image.open(os.path.join(clean_d, c)).convert("RGB"))
        pairs.append((x, y))
    H, W = pairs[0][0].shape[-2:]
    print(f"  {len(pairs)} validation images at {W}x{H} "
          f"({H*W/1e6:.1f} MP each)", flush=True)

    def centre(t, s=256):
        i, j = (t.shape[-2] - s) // 2, (t.shape[-1] - s) // 2
        return t[..., i:i + s, j:j + s]

    # ---- THE NO-OP BASELINE. This is the number the corpus never recorded.
    noop_crop = psnr_from_mses([y_mse(centre(x), centre(y)) for x, y in pairs])
    noop_full = psnr_from_mses([y_mse(x, y) for x, y in pairs])
    print(f"\n=== NO-OP BASELINE (return the noisy input unchanged) ===")
    print(f"  256x256 centre crop : {noop_crop:6.2f} dB   <- what the stored metric scores")
    print(f"  full {W}x{H}   : {noop_full:6.2f} dB")
    print(f"  (a model must beat these to have denoised anything at all)")

    print(f"\n=== per model ===")
    print(f"  {'model':<44}{'stored':>7}{'crop':>7}{'full':>7}"
          f"{'gain_c':>8}{'gain_f':>8}")
    res, t0 = [], time.time()
    for name in have:
        try:
            ns = {}
            exec(compile(code[name], "<m>", "exec"), ns)
            net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), dev).to(dev)
            sd = torch.load(os.path.join(a.ckpt, name, "best_model.pth"),
                            map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            net.load_state_dict(sd, strict=False)
            net.eval()
            pc, pf = [], []
            for x, y in pairs:
                xc = centre(x).unsqueeze(0).to(dev)
                with torch.no_grad():
                    pc.append(y_mse(net(xc).cpu()[0], centre(y)))
                xf = x.unsqueeze(0).to(dev)
                o = tiled_forward(net, xf, a.tile, a.overlap, dev)
                pf.append(y_mse(o.cpu()[0], y))
            c_, f_ = psnr_from_mses(pc), psnr_from_mses(pf)
            res.append({"name": name, "stored": best.get(name, 0), "crop": c_, "full": f_,
                        "gain_crop": c_ - noop_crop, "gain_full": f_ - noop_full})
            print(f"  {name[:44]:<44}{best.get(name,0):7.2f}{c_:7.2f}{f_:7.2f}"
                  f"{c_-noop_crop:+8.2f}{f_-noop_full:+8.2f}", flush=True)
        except Exception as e:
            print(f"  {name[:44]:<44}  FAILED {type(e).__name__}: {str(e)[:50]}")
        finally:
            try:
                del net
            except Exception:
                pass
            if dev == "cuda":
                torch.cuda.empty_cache()

    if res:
        gc = sum(r["gain_crop"] for r in res) / len(res)
        gf = sum(r["gain_full"] for r in res) / len(res)
        dc = sum(r["crop"] - r["stored"] for r in res) / len(res)
        print(f"\n=== VERDICT ({len(res)} models, {time.time()-t0:.0f}s) ===")
        print(f"  mean gain over no-op, centre crop : {gc:+.2f} dB")
        print(f"  mean gain over no-op, FULL image  : {gf:+.2f} dB")
        print(f"  mean |recomputed crop - stored|   : {dc:+.2f} dB "
              f"(should be ~0 if this harness matches the DB)")
        print()
        if gf < 0.5:
            print("  => ON FULL IMAGES THESE MODELS ARE NOT MEANINGFULLY DENOISING.")
        elif gf < gc - 1.0:
            print("  => models denoise, but MUCH worse on the full frame than on the "
                  "centre crop the metric scores.")
        else:
            print("  => models denoise, and the centre-crop metric is representative.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"noop_crop": noop_crop, "noop_full": noop_full,
                   "images": len(pairs), "size": [W, H], "models": res}, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

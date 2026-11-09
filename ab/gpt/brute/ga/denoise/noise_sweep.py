"""Is the small denoising gain a property of our MODELS, or of the DATASET we gave them?

THE FINDING THIS INTERROGATES. Measured with a harness verified to reproduce stored DB
values exactly (mean |recomputed - stored| = -0.00 dB over 12 models): returning the noisy
MAI validation image UNCHANGED scores 33.02 dB, the best of 1227 searched models scores
34.54, and the corpus median is 33.71. So the median model improves the image by 0.69 dB and
the best by 1.5 dB. The whole search has been competing inside a couple of dB.

TWO EXPLANATIONS, and they demand different responses:
  (a) the models are weak -- then more search, better operators, longer training.
  (b) THE DATA HAS ALMOST NO NOISE TO REMOVE -- then no amount of search helps, because the
      metric cannot express the difference between a good and a bad denoiser, and the right
      move is to change the benchmark.

This distinguishes them by sweeping the noise level and watching how the gain over no-op
responds. A model that learned genuine denoising removes MORE noise when there is more to
remove, so its gain should GROW with sigma. A model that learned a near-identity mapping
tuned to one noise level has nothing to contribute as sigma rises, and its gain flattens or
goes negative.

THIS IS A GENERALISATION TEST, NOT A CAPABILITY TEST, and the distinction is not a caveat to
skip. Every model here was trained on MAI at its native noise level. Evaluating at sigma=25
asks how they behave OUT OF DISTRIBUTION. A model that degrades there is not thereby a bad
architecture -- it is a model being asked a question it was never trained on. What the sweep
licenses is a statement about the BENCHMARK: if gains stay tiny across every noise level, the
ceiling is the data; if they grow, the MAI noise floor was the binding constraint all along.

Synthetic AWGN is added to the CLEAN validation images (the DnCNN protocol, matching
ab/nn/loader/awgn_color.py: noisy = clean + N(0, sigma/255)), seeded so every model sees the
identical noise realisation. SIDD is included when present as a REAL-noise reference point,
because synthetic Gaussian noise is the easy case and real sensor noise is spatially
correlated.

  python noise_sweep.py --sigmas 0 5 10 15 25 50 --top 6
"""
import argparse
import json
import math
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
    """MSE on BT.601 luminance -- the quantity ab/nn/metric/psnr.py scores. Not RGB:
    channel-independent noise partly cancels in Y, so the two differ by several dB and
    mixing them silently invalidates every comparison against the database."""
    a, b = a.double().clamp(0, 1), b.double()
    if a.shape[-3] == 3:
        ya = 0.299 * a[..., 0, :, :] + 0.587 * a[..., 1, :, :] + 0.114 * a[..., 2, :, :]
        yb = 0.299 * b[..., 0, :, :] + 0.587 * b[..., 1, :, :] + 0.114 * b[..., 2, :, :]
    else:
        ya, yb = a[..., 0, :, :], b[..., 0, :, :]
    return float(((ya - yb) ** 2).mean())


def db(mses):
    """dB of the MEAN MSE. The metric averages MSE over samples then converts once;
    averaging per-image dB is a different statistic (-log is concave)."""
    m = sum(mses) / len(mses)
    return float("inf") if m <= 0 else 10.0 * math.log10(1.0 / m)


def centre(t, s=256):
    i, j = (t.shape[-2] - s) // 2, (t.shape[-1] - s) // 2
    return t[..., i:i + s, j:j + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--ckpt", default="/a/mm/sweep_out/ckpt")
    ap.add_argument("--val", default="/a/mm/nn-dataset/validation")
    ap.add_argument("--sidd", default="/a/mm/nn-dataset/sidd")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0, 5, 10, 15, 25, 50])
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--images", type=int, default=50)
    ap.add_argument("--sidd-images", type=int, default=32)
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/noise_sweep.json")
    a = ap.parse_args()

    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== noise sweep | device={dev} ===", flush=True)

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    best = {n: (v or 0) * 48.0 for n, v in con.execute(
        "SELECT nn, MAX(accuracy)*48 FROM stat WHERE task=? AND accuracy IS NOT NULL "
        "GROUP BY nn", (TASK,))} if False else {
        n: v for n, v in con.execute(
            "SELECT nn, MAX(accuracy)*48 FROM stat WHERE task=? AND accuracy IS NOT NULL "
            "GROUP BY nn", (TASK,))}
    code = dict(con.execute("SELECT name, code FROM nn"))
    con.close()

    names = [n for n in code if os.path.exists(os.path.join(a.ckpt, n, "best_model.pth"))]
    names.sort(key=lambda n: -best.get(n, 0))
    names = names[:a.top]
    print(f"  {len(names)} models (top by stored PSNR)")

    # CLEAN validation images. The MAI 'denoised' directory IS the ground truth, so adding
    # synthetic noise to it gives a controlled sweep on the same content the models were
    # evaluated on -- content is held fixed, only the noise level moves.
    cd = os.path.join(a.val, "denoised")
    nd = os.path.join(a.val, "original")
    cf = sorted(f for f in os.listdir(cd) if f.lower().endswith((".png", ".jpg")))[:a.images]
    nf = sorted(f for f in os.listdir(nd) if f.lower().endswith((".png", ".jpg")))[:a.images]
    clean = [centre(TF.to_tensor(Image.open(os.path.join(cd, f)).convert("RGB")))
             for f in cf]
    native = [centre(TF.to_tensor(Image.open(os.path.join(nd, f)).convert("RGB")))
              for f in nf]
    print(f"  {len(clean)} clean 256x256 centre crops", flush=True)

    def load(name):
        ns = {}
        exec(compile(code[name], "<m>", "exec"), ns)
        net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), dev).to(dev)
        sd = torch.load(os.path.join(a.ckpt, name, "best_model.pth"),
                        map_location=dev, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        r = net.load_state_dict(sd, strict=False)
        if len(r.unexpected_keys) == len(sd):
            raise ValueError("no checkpoint tensor matched")
        net.eval()
        return net

    results = {}

    # ---- the MAI native noise, as a control that must reproduce the known numbers
    print(f"\n=== MAI native noise (control -- must reproduce stored values) ===")
    noop = db([y_mse(n, c) for n, c in zip(native, clean)])
    print(f"  no-op {noop:.2f} dB")
    print(f"  {'model':<46}{'psnr':>7}{'stored':>8}{'gain':>7}")
    for nm in names:
        try:
            net = load(nm)
            ms = []
            for n, c in zip(native, clean):
                with torch.no_grad():
                    ms.append(y_mse(net(n.unsqueeze(0).to(dev)).cpu()[0], c))
            p = db(ms)
            print(f"  {nm[:46]:<46}{p:7.2f}{best.get(nm,0):8.2f}{p-noop:+7.2f}", flush=True)
            results.setdefault("native", {})[nm] = {"psnr": p, "gain": p - noop}
            del net
            torch.cuda.empty_cache() if dev == "cuda" else None
        except Exception as e:
            print(f"  {nm[:46]:<46}  FAILED {type(e).__name__}: {str(e)[:40]}")
    results["native_noop"] = noop

    # ---- synthetic AWGN sweep
    print(f"\n=== synthetic AWGN on the CLEAN images (same content, rising noise) ===")
    print(f"  {'sigma':>6}{'no-op':>9}{'best model':>12}{'gain':>8}   verdict")
    g = torch.Generator().manual_seed(1234)
    sweep = {}
    for s in a.sigmas:
        if s <= 0:
            continue
        noisy = [(c + torch.randn(c.shape, generator=g) * (s / 255.0)).clamp(0, 1)
                 for c in clean]
        nz = db([y_mse(n, c) for n, c in zip(noisy, clean)])
        rows = {}
        for nm in names:
            try:
                net = load(nm)
                ms = []
                for n, c in zip(noisy, clean):
                    with torch.no_grad():
                        ms.append(y_mse(net(n.unsqueeze(0).to(dev)).cpu()[0], c))
                rows[nm] = db(ms)
                del net
                torch.cuda.empty_cache() if dev == "cuda" else None
            except Exception:
                continue
        if not rows:
            continue
        bn, bp = max(rows.items(), key=lambda kv: kv[1])
        gain = bp - nz
        verdict = ("denoising" if gain > 0.5 else
                   "marginal" if gain > 0 else "WORSE THAN DOING NOTHING")
        print(f"  {s:6.0f}{nz:9.2f}{bp:12.2f}{gain:+8.2f}   {verdict}", flush=True)
        sweep[s] = {"noop": nz, "best": bp, "best_model": bn, "gain": gain,
                    "all": rows}
    results["awgn"] = sweep

    # ---- SIDD: real sensor noise, spatially correlated, the hard case
    if os.path.isdir(os.path.join(a.sidd, "original")):
        print(f"\n=== SIDD (REAL smartphone sensor noise) ===")
        sn = sorted(os.listdir(os.path.join(a.sidd, "original")))[:a.sidd_images]
        sc = sorted(os.listdir(os.path.join(a.sidd, "denoised")))[:a.sidd_images]
        sx = [centre(TF.to_tensor(Image.open(
            os.path.join(a.sidd, "original", f)).convert("RGB"))) for f in sn]
        sy = [centre(TF.to_tensor(Image.open(
            os.path.join(a.sidd, "denoised", f)).convert("RGB"))) for f in sc]
        snoop = db([y_mse(x, y) for x, y in zip(sx, sy)])
        print(f"  {len(sx)} pairs | no-op {snoop:.2f} dB "
              f"({'NOISIER' if snoop < noop else 'cleaner'} than MAI's {noop:.2f})")
        srows = {}
        for nm in names:
            try:
                net = load(nm)
                ms = []
                for x, y in zip(sx, sy):
                    with torch.no_grad():
                        ms.append(y_mse(net(x.unsqueeze(0).to(dev)).cpu()[0], y))
                srows[nm] = db(ms)
                print(f"  {nm[:46]:<46}{db(ms):7.2f}{db(ms)-snoop:+8.2f}", flush=True)
                del net
                torch.cuda.empty_cache() if dev == "cuda" else None
            except Exception as e:
                print(f"  {nm[:46]:<46}  FAILED {type(e).__name__}")
        if srows:
            bp = max(srows.values())
            print(f"  BEST gain over no-op: {bp - snoop:+.2f} dB")
            results["sidd"] = {"noop": snoop, "best": bp, "gain": bp - snoop,
                               "all": srows}

    print(f"\n=== READING THIS ===")
    print("  gain GROWS with sigma  -> the models learned real denoising; MAI's noise")
    print("                            floor was the binding constraint on the search.")
    print("  gain FLAT or NEGATIVE  -> they learned a near-identity map tuned to MAI, and")
    print("                            a noisier benchmark needs RETRAINING, not just")
    print("                            re-evaluation. (Expected: these are out of")
    print("                            distribution here -- this is a generalisation test.)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

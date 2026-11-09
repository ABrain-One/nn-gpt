"""Per-frame full-resolution RGB PSNR, for diagnosing WHICH frames a model fails on.

WHY THIS IS SEPARATE FROM fullres_eval.py. That script is the one that produced the published
numbers and it should not be touched to answer a diagnostic question. It also imports
torchvision, which cannot be installed on this host right now (the root filesystem has ~2 GB
free and a torchvision wheel needs more), and the K8s memory quota is at 558 of 560 Gi so a pod
is not available either. Everything torchvision was used for here is one call, `TF.to_tensor`,
which is `asarray(img)/255` transposed to CHW -- so this probe reimplements exactly that and
depends only on torch + PIL + numpy.

IT MUST AGREE WITH THE REAL SCRIPT OR IT IS USELESS. `psnr_rgb` and `load_net` are copied
verbatim from fullres_eval.py rather than reimplemented, and the probe prints its own whole-frame
mean so it can be checked against the stored `psnr_rgb_fullres` for the same model. If those two
disagree, the probe is wrong and its per-frame numbers mean nothing.

THE QUESTION IT EXISTS TO ANSWER. LiteDenoiseNet -- the published-architecture anchor -- scores
33.247 dB against the paper's 37.66, and reexport_report.py's own rule says a gap "beyond a few
dB" invalidates the protocol. Its worst frame is 19.76 dB while every generated model's worst is
25.6-27.2. Either one frame is degenerate for this model specifically, or that frame is hard for
everything and the anchor is merely worse everywhere. Only per-frame numbers separate those, and
only the second reading leaves the protocol usable.

  python perframe_probe.py --only LiteDenoiseNet
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

H = "/shared/ssd/home/b-b-ahmed/thesis"
RX = f"{H}/sweep_out/reexport"


def to_tensor(img):
    """torchvision.transforms.functional.to_tensor, without torchvision.

    HWC uint8 -> CHW float32 in [0,1]. That is the whole contract to_tensor has for a PIL RGB
    image, and matching it exactly is what makes this probe comparable to fullres_eval.py.
    """
    a = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(a.transpose(2, 0, 1).copy()).float().div_(255.0)


def psnr_rgb(out, ref, shave=0):
    """Copied verbatim from fullres_eval.py -- do not 'improve' it."""
    if shave:
        out = out[..., shave:-shave, shave:-shave]
        ref = ref[..., shave:-shave, shave:-shave]
    mse = torch.mean((out.clamp(0, 1) - ref) ** 2).item()
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def load_net(src_path, device):
    """Copied verbatim from fullres_eval.py."""
    code = open(src_path).read()
    ns = {"__name__": "reexport_model"}
    exec(compile(code, os.path.basename(src_path), "exec"), ns)
    Net = ns.get("Net")
    if Net is None:
        raise RuntimeError("no Net class in source")
    prm = {"lr": 1e-4, "momentum": 0.9, "dropout": 0.0, "batch": 1, "transform": "denoise"}
    net = Net((1, 3, 256, 256), (3,), prm, device)
    return net.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", required=True, help="model name (repeatable)")
    ap.add_argument("--subset", default=f"{RX}/subset.json")
    ap.add_argument("--val", default=f"{H}/nn-dataset/validation")
    ap.add_argument("--out", default=f"{RX}/perframe.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cpu")           # GPUs are held by the GA arms; 50 frames is affordable here
    sub = json.load(open(a.subset))
    src_dir = os.path.join(os.path.dirname(a.subset), "src")
    nd, cd = os.path.join(a.val, "original"), os.path.join(a.val, "denoised")
    nf = sorted(f for f in os.listdir(nd) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    cf = sorted(f for f in os.listdir(cd) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if len(nf) != len(cf):
        sys.exit(f"FATAL {len(nf)} noisy vs {len(cf)} clean")
    if a.limit:
        nf, cf = nf[:a.limit], cf[:a.limit]
    print(f"cpu | {len(nf)} frames | targets: {a.only}")

    out = {}
    for row in sub["models"]:
        name = row["model"]
        if name not in a.only:
            continue
        ck = None
        for store in (f"{RX}/ckpt", f"{H}/sweep_out/ckpt", f"{H}/nn-dataset/out/ckpt"):
            p = os.path.join(store, name, "best_model.pth")
            if os.path.isfile(p):
                ck = p
                break
        if not ck:
            print(f"  SKIP {name}: no checkpoint")
            continue
        net = load_net(os.path.join(src_dir, row["src"]), dev)
        sd = torch.load(ck, map_location=dev)
        sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if len(missing) > len(list(net.state_dict())) * 0.5:
            print(f"  SKIP {name}: checkpoint mismatch, {len(missing)} missing")
            continue
        net.eval()
        per = {}
        with torch.no_grad():
            for i, (fn, fc) in enumerate(zip(nf, cf), 1):
                x = to_tensor(Image.open(os.path.join(nd, fn)).convert("RGB")).unsqueeze(0)
                y = to_tensor(Image.open(os.path.join(cd, fc)).convert("RGB")).unsqueeze(0)
                per[fn] = round(psnr_rgb(net(x), y), 3)
                del x, y
                if i % 10 == 0:
                    print(f"    {name[:30]} {i}/{len(nf)}", flush=True)
        mean = sum(per.values()) / len(per)
        out[name] = {"mean": round(mean, 3), "missing_keys": len(missing), "per_frame": per}
        lo = sorted(per.items(), key=lambda kv: kv[1])[:5]
        print(f"  {name[:40]:<42} mean {mean:7.3f}  worst5 {[(k[:14], v) for k, v in lo]}")

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("CHECK: compare each 'mean' against psnr_rgb_fullres in fullres.jsonl for the same "
          "model. A mismatch means this probe is wrong, not that the model changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

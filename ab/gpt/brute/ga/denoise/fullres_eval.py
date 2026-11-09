"""Full-resolution RGB PSNR on the 50 MAI validation frames — the published protocol.

WHAT THIS REPRODUCES. The MAI 2026 paper evaluates "on full-resolution validation images
(2432 x 3200)" with "PSNR and SSIM computed on RGB channels after applying a 1-pixel boundary
crop", and reports 37.66 dB for LiteDenoiseNet. This corpus stores something different: Y-channel
PSNR on ONE 256x256 centre crop, which is 0.84% of a 7.8 MP frame. Those two numbers are not
comparable, and every cross-paper claim in this project has had to be hedged because of it.

PAIRING, WHICH IS EASY TO GET BACKWARDS. ab/nn/loader/denoise.py sets noisy = `original/` and
clean = `denoised/`, paired positionally over sorted filenames with a length check. Note that
denoise_awgn.py uses the SAME `denoised/` directory as its clean source but SYNTHESISES noise and
deliberately ignores `original/` -- so reusing that loader here would silently evaluate against
synthetic noise instead of MAI's real capture. The pairing is therefore done directly, not by
importing either loader.

WHOLE IMAGE, NOT TILES. The paper evaluates whole frames "to reflect global structural coherence
and the absence of tiling artifacts", so that is the default. A 4-level encoder needs H and W
divisible by 16; 2432 and 3200 both are, so no padding is required at native size. If a model OOMs
at full resolution the fallback is overlapped tiling, and `mode` records which path produced each
number -- a tiled row is not silently mixed in with whole-image rows.

BOTH BORDER CONVENTIONS ARE REPORTED. The paper shaves 1 pixel; ab/nn/metric/psnrrgb.py documents
"no border shave". On 3200x2432 that is 0.06% of the pixels and the difference should be
negligible -- printing both proves it rather than assuming it.

  python fullres_eval.py --subset /a/mm/sweep_out/reexport/subset.json
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# TORCHVISION IS OPTIONAL, AND THE FALLBACK IS EXACT -- NOT AN APPROXIMATION.
# The only thing torchvision was ever used for here is TF.to_tensor, whose entire contract for a
# PIL RGB image is "HWC uint8 -> CHW float32 in [0,1]". The numpy path below reproduces that
# bit-for-bit, which is what perframe_probe.py already relies on. Making the import optional
# matters because this module is now imported by the GA fitness path (fullres_score.py), and the
# GA pods are not guaranteed to have torchvision even though they have torch -- a hard import
# there would turn "score this candidate on the real metric" into an ImportError mid-run.
try:
    import torchvision.transforms.functional as TF

    def to_tensor(img):
        return TF.to_tensor(img)
except ImportError:  # pragma: no cover - depends on the image the pod was built from
    import numpy as _np

    def to_tensor(img):
        a = _np.asarray(img, dtype=_np.uint8)
        return torch.from_numpy(a.transpose(2, 0, 1).copy()).float().div_(255.0)


CKPT_DIRS = ["/a/mm/sweep_out/reexport/ckpt",     # retrained here, searched first
             "/a/mm/sweep_out/ckpt",              # 1,660 -- DenoiseGA, LLM_Pg*, SMOKEPAR
             "/a/mm/nn-dataset/out/ckpt"]         # 19 hand-written families incl. LiteDenoiseNet


def find_ckpt(name, declared=None):
    """Resolve a checkpoint across all three stores.

    ALL 16 MODELS MUST USE THE SAME SELECTION RULE, which is why this takes best_model.pth and not
    the best epoch under full-resolution PSNR. Util.py:203 documents that best_model.pth is chosen
    by 256x256 centre-crop PSNR and that this can INVERSE-RANK against the full-resolution protocol
    (two DenoiseSplitterNetX seeds: 37.908/37.920 on crops, 37.927/37.562 at full res). Selecting
    retrained models under the full-res metric while LiteDenoiseNet's pre-existing checkpoint was
    selected under the crop metric would hand the retrained models an advantage the anchor never
    had, which would corrupt the one comparison this whole run exists to make. LEMUR_SAVE_EVERY_EPOCH
    keeps the per-epoch snapshots so the "what if we selected on full res" question stays answerable
    separately, as a sensitivity check rather than as the headline.
    """
    if declared and os.path.exists(declared):
        return declared
    for base in CKPT_DIRS:
        p = os.path.join(base, name, "best_model.pth")
        if os.path.exists(p):
            return p
    return None


def load_net(src_path, device):
    """Instantiate Net from a staged source file without importing it as a module."""
    code = open(src_path).read()
    ns = {"__name__": "reexport_model"}
    exec(compile(code, os.path.basename(src_path), "exec"), ns)
    Net = ns.get("Net")
    if Net is None:
        raise RuntimeError("no Net class in source")
    prm = {"lr": 1e-4, "momentum": 0.9, "dropout": 0.0, "batch": 1, "transform": "denoise"}
    # in_shape is BATCHED (N,C,H,W) -- Train.py:210. Passing (3,256,256) made every model that
    # reads in_shape[1] build a 256-channel stem, which is why 5 of 16 rows died on
    # 'size mismatch for conv1.weight: [64,3,3,3] vs [64,256,3,3]'.
    net = Net((1, 3, 256, 256), (3,), prm, device)
    return net


def psnr_rgb(out, ref, shave=0):
    """Per-image RGB PSNR on the 0-1 scale, matching ab/nn/metric/psnrrgb.py."""
    if shave:
        out = out[..., shave:-shave, shave:-shave]
        ref = ref[..., shave:-shave, shave:-shave]
    mse = torch.mean((out.clamp(0, 1) - ref) ** 2).item()
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def forward_tiled_ladder(net, x, tiles=(1024, 512, 256), overlap=64):
    """Try progressively smaller tiles until one fits.

    A single 1024 tile is not always small enough. Some models expand to very wide intermediate
    channel counts, and on a 24 GiB card a 1024x1024 patch through those still asks for ~16 GiB --
    so the fallback OOM'd exactly like the whole-frame attempt it was supposed to rescue, and the
    row was written as FAIL with no number (14 of 198 models in shard 0). Stepping down to 512 and
    then 256 costs wall-clock on those few models and nothing on the rest. Each attempt clears the
    cache first, because the previous failed attempt leaves its partial allocations behind.
    """
    last = None
    for t in tiles:
        try:
            if x.is_cuda:
                torch.cuda.empty_cache()
            return forward_tiled(net, x, tile=t, overlap=overlap), f"tiled-{t}/{overlap}"
        except torch.cuda.OutOfMemoryError as e:
            last = e
    raise last


def forward_tiled(net, x, tile=1024, overlap=64):
    """Overlapped tiling, used only when the whole frame will not fit."""
    _, _, H, W = x.shape
    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    step = tile - overlap
    for top in range(0, H, step):
        for left in range(0, W, step):
            b, r = min(top + tile, H), min(left + tile, W)
            t, l = max(0, b - tile), max(0, r - tile)
            # snap the patch to a multiple of 16 so strided encoders keep exact shapes
            hh, ww = ((b - t) // 16) * 16, ((r - l) // 16) * 16
            patch = x[:, :, t:t + hh, l:l + ww]
            acc[:, :, t:t + hh, l:l + ww] += net(patch)
            wsum[:, :, t:t + hh, l:l + ww] += 1
            if r >= W:
                break
        if b >= H:
            break
    return acc / wsum.clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="/a/mm/sweep_out/reexport/subset.json")
    ap.add_argument("--val", default="/a/mm/nn-dataset/validation")
    ap.add_argument("--out", default="/a/mm/sweep_out/reexport/fullres.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 50 frames")
    ap.add_argument("--only", default=None, help="evaluate a single model by name")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = json.load(open(a.subset))
    src_dir = os.path.join(os.path.dirname(a.subset), "src")

    noisy_d, clean_d = os.path.join(a.val, "original"), os.path.join(a.val, "denoised")
    nf = sorted(f for f in os.listdir(noisy_d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    cf = sorted(f for f in os.listdir(clean_d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if len(nf) != len(cf):
        sys.exit(f"FATAL {len(nf)} noisy vs {len(cf)} clean -- positional pairing would be wrong")
    if a.limit:
        nf, cf = nf[:a.limit], cf[:a.limit]
    print(f"device {dev} | {len(nf)} validation frames | {len(sub['models'])} models")

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    print(f"already evaluated: {len(done)}")

    for row in sub["models"]:
        name = row["model"]
        if a.only and name != a.only:
            continue
        if name in done:
            continue
        ckpt = find_ckpt(name, row.get("ckpt"))
        if not ckpt:
            print(f"SKIP {name[:44]} -- no checkpoint in any store yet")
            continue
        rec = {"model": name, "n_frames": len(nf), "ckpt": ckpt, "status": "FAIL"}
        t0 = time.time()
        # CLEAR BEFORE, NOT ONLY AFTER. The post-loop cleanup runs inside a bare try and does not
        # run at all on the exception path, so a model that dies mid-frame leaves its net and its
        # full-resolution tensors resident. The next model then starts with ~18 of 23.5 GiB already
        # taken and OOMs after 1s on an allocation it would have satisfied from a clean card --
        # which is how one genuinely-large model turned into a run of consecutive FAILs.
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        try:
            net = load_net(os.path.join(src_dir, row["src"]), dev)
            sd = torch.load(ckpt, map_location=dev)
            sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
            missing, unexpected = net.load_state_dict(sd, strict=False)
            # A checkpoint that matches almost nothing loads "successfully" under strict=False and
            # then scores like an untrained net. Refuse rather than publish that.
            if len(missing) > len(list(net.state_dict())) * 0.5:
                raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing keys")
            rec["missing_keys"], rec["unexpected_keys"] = len(missing), len(unexpected)
            net.eval()
            p0, p1, mode = [], [], "whole"
            with torch.no_grad():
                for fn, fc in zip(nf, cf):
                    x = to_tensor(Image.open(os.path.join(noisy_d, fn)).convert("RGB")).unsqueeze(0).to(dev)
                    y = to_tensor(Image.open(os.path.join(clean_d, fc)).convert("RGB")).unsqueeze(0).to(dev)
                    try:
                        o = net(x)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        o, mode = forward_tiled_ladder(net, x)
                    p0.append(psnr_rgb(o, y, 0))
                    p1.append(psnr_rgb(o, y, 1))
                    del x, y, o
            rec.update(status="OK", mode=mode,
                       psnr_rgb_fullres=round(sum(p0) / len(p0), 3),
                       psnr_rgb_fullres_shave1=round(sum(p1) / len(p1), 3),
                       psnr_min=round(min(p0), 3), psnr_max=round(max(p0), 3),
                       # PER-FRAME, keyed by filename. The mean/min/max alone cannot distinguish
                       # "this model fails on one degenerate frame" from "this frame is hard for
                       # every model", and that distinction decides whether the anchor's -4.4 dB
                       # gap is a defect or a property of the validation set. 50 floats is a
                       # trivial addition to a row that already carries a checkpoint path.
                       per_frame={fn: round(v, 3) for fn, v in zip(nf, p0)})
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            # Free this model's frame tensors on the failure path too. They are locals of main(),
            # so they stay alive until rebound on the next iteration -- long enough to starve the
            # next model of the card. NameError is expected when the failure happened before the
            # frame loop bound them.
            for v in ("x", "y", "o", "sd"):
                try:
                    exec(f"del {v}")
                except Exception:
                    pass
        rec["seconds"] = round(time.time() - t0)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{rec['status']:<5} {name[:44]:<46} "
              f"{rec.get('psnr_rgb_fullres', '—')} dB ({rec.get('mode', '')}) {rec['seconds']}s"
              + (f"  {rec.get('error', '')}" if rec["status"] != "OK" else ""))
        try:
            del net
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

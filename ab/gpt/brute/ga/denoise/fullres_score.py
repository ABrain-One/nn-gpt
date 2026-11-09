"""Score ONE freshly-trained candidate on the published protocol, for use as GA fitness.

WHY THE GA NEEDS THIS. Four arms ran for a day, produced 1,287 distinct new architectures, and not
one beat a champion that existed before they started: best new model 34.886 dB against an incumbent
36.869 dB, both at 5 epochs, both from the same trainer. The arms were not under-trained -- a
50-epoch run on the champions moved the mean by -0.077 dB -- they were optimising the wrong signal.

Fitness has been 256x256 Y-channel crop PSNR, and measured over 1,578 models that metric has
Spearman +0.724 against full-resolution RGB overall but -0.025 among the top 100. It separates bad
models from good ones and then goes blind exactly where the search spends its time. So the GA
faithfully maximised a number that stops meaning anything near the frontier, and wandered.

WHAT IT COSTS. A full-resolution pass over the 50 MAI frames is ~35 s against ~210 s of 5-epoch
training: about 17% overhead per candidate, no extra GPU, and it reuses the checkpoint the trainer
already wrote. Against an arm that spent a day to produce nothing better, that is cheap.

WHY IT REUSES fullres_eval RATHER THAN REIMPLEMENTING. The number this returns has to be directly
comparable to the 1,589 full-resolution numbers already measured, and to the published protocol.
psnr_rgb, load_net, the tile ladder and to_tensor are all imported from that module, so the GA
cannot drift away from the corpus's own definition of the metric. fullres_eval's torchvision
import was made optional for this reason -- the GA pods have torch but are not guaranteed to have
torchvision, and a hard import would turn scoring into an ImportError mid-run.

FAIL-SOFT, AND LOUD. If anything goes wrong -- no checkpoint, OOM at every tile size, a shape
error -- this returns None rather than raising, and the caller keeps the crop score with a logged
warning. A scoring outage must not look like a population-wide quality collapse, which is the
failure mode the predictor's fail-soft path was written for after the same mistake.
"""
import os
import sys

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fullres_eval import forward_tiled_ladder, psnr_rgb, to_tensor  # noqa: E402

_FRAMES = {}


def _frames(val_dir, limit):
    """Cache the sorted (noisy, clean) filename pairs; the directory never changes mid-run."""
    ck = (val_dir, limit)
    if ck in _FRAMES:
        return _FRAMES[ck]
    nd, cd = os.path.join(val_dir, "original"), os.path.join(val_dir, "denoised")
    ext = (".png", ".jpg", ".jpeg")
    nf = sorted(f for f in os.listdir(nd) if f.lower().endswith(ext))
    cf = sorted(f for f in os.listdir(cd) if f.lower().endswith(ext))
    if len(nf) != len(cf):
        raise RuntimeError(f"{len(nf)} noisy vs {len(cf)} clean -- positional pairing would be wrong")
    if limit:
        nf, cf = nf[:limit], cf[:limit]
    _FRAMES[ck] = (nd, cd, nf, cf)
    return _FRAMES[ck]


def score_source(code, ckpt_path, val_dir, device=None, limit=0, log=None):
    """Full-resolution RGB PSNR in dB for `code` loaded with `ckpt_path`, or None on any failure.

    `limit` truncates the frame set. 0 = all 50, which is what every stored number used; a smaller
    value is faster but produces a figure that is NOT comparable to the corpus, so the caller must
    keep it at 0 for anything that will be reported.
    """
    say = log or (lambda m: None)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        say(f"        FULLRES: no checkpoint at {ckpt_path}; keeping crop score")
        return None
    net = None
    try:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nd, cd, nf, cf = _frames(val_dir, limit)

        ns = {"__name__": "ga_candidate"}
        exec(compile(code, "candidate.py", "exec"), ns)
        Net = ns.get("Net")
        if Net is None:
            say("        FULLRES: no Net class; keeping crop score")
            return None
        prm = {"lr": 1e-4, "momentum": 0.9, "dropout": 0.0, "batch": 1, "transform": "denoise"}
        net = Net((1, 3, 256, 256), (3,), prm, dev).to(dev)

        sd = torch.load(ckpt_path, map_location=dev)
        if isinstance(sd, dict):
            sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        missing, _ = net.load_state_dict(sd, strict=False)
        # A checkpoint matching almost nothing loads fine under strict=False and then scores like
        # an untrained net -- which would look like a terrible architecture rather than a bad load.
        if len(missing) > len(list(net.state_dict())) * 0.5:
            say(f"        FULLRES: checkpoint mismatch ({len(missing)} missing); keeping crop score")
            return None
        net.eval()

        tot = 0.0
        with torch.no_grad():
            for fn, fc in zip(nf, cf):
                x = to_tensor(Image.open(os.path.join(nd, fn)).convert("RGB")).unsqueeze(0).to(dev)
                y = to_tensor(Image.open(os.path.join(cd, fc)).convert("RGB")).unsqueeze(0).to(dev)
                try:
                    o = net(x)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    o, _mode = forward_tiled_ladder(net, x)
                tot += psnr_rgb(o, y, 0)
                del x, y, o
        return tot / len(nf)
    except Exception as e:
        say(f"        FULLRES: {type(e).__name__}: {str(e)[:120]}; keeping crop score")
        return None
    finally:
        try:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

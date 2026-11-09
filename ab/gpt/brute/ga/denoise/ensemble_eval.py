"""Is the CORPUS a better teacher than any single model in it?

THE QUESTION. Matching the published 37.66 dB appears to need knowledge distillation, which that
paper performs from a 41.6M-parameter teacher we do not have. We do have 3,952 trained
checkpoints. Averaging their outputs costs no training, so before building a distillation
pipeline we measure whether such an ensemble actually exceeds our best single model (37.042 dB
full-resolution RGB). If it does not, the idea is dead for the price of an afternoon.

WHY THIS CAN FAIL, AND WHY WE TEST INSTEAD OF ASSUMING. Denoising is regression, so an ensemble
averages pixels. Our models span 32-37 dB; averaging a 37 dB model with a 33 dB one is worse than
the 37 dB model alone. Any gain therefore has to come from a SMALL set of strong, decorrelated
members, and whether that beats the single best is purely empirical. We sweep k so the curve is
visible rather than reporting one hand-picked ensemble size.

A KNOWN WEAKNESS OF THE MEMBERS. `best_model.pth` was selected by the 256-crop metric, which
Util.save_if_best documents can inverse-rank against full resolution. Members are therefore not
necessarily each model's best full-res epoch, which lowers the ceiling this test can reach. A
positive result here is thus a lower bound on what a properly selected ensemble would give.

  python3 ensemble_eval.py --topk 2,4,8,16 --val /a/mm/nn-dataset/validation
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from fullres_eval import load_net, psnr_rgb, forward_tiled_ladder   # noqa: E402

T = "/a/mm" if os.path.isdir("/a/mm") else "/shared/ssd/home/b-b-ahmed/thesis"


def to_tensor(p):
    import numpy as np
    a = np.asarray(Image.open(p).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=T + "/nn-dataset/validation")
    ap.add_argument("--topk", default="1,2,4,8,16")
    ap.add_argument("--out", default=T + "/sweep_out/ensemble/result.json")
    ap.add_argument("--limit-frames", type=int, default=0)
    # --subset: take members from an explicit {model,src,ckpt} list (the same format
    # fullres_eval.py consumes) instead of discovering them from fullres_union.json +
    # sweep_out/ckpt. Needed for checkpoints that live outside the corpus tree, e.g. the
    # retrained champions under sweep_out/champion/ckpt/Champ-<hash>/epoch_NN.pth.
    ap.add_argument("--subset", default=None)
    ap.add_argument("--scores", default=None,
                    help="json {model: psnr} used to rank subset members")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.subset:
        sub = json.load(open(a.subset))["models"]
        src_dir = os.path.join(os.path.dirname(a.subset), "src")
        sc = json.load(open(a.scores)) if a.scores else {}
        cands = []
        for row in sub:
            nm = row["model"]
            base = nm.split("@")[0]
            cands.append((nm, float(sc.get(base, sc.get(nm, 0.0))), row["ckpt"],
                          open(os.path.join(src_dir, row["src"])).read()))
        cands.sort(key=lambda c: -c[1])
    else:
        cands = None
    fr = json.load(open(T + "/sweep_out/fullres_union.json")) if cands is None else {}
    src_db, canon = {}, {}
    if cands is None:
        # only needed for DB-driven discovery; subset mode already carries its sources
        import sqlite3
        con = sqlite3.connect("file:%s/nn-dataset/db/ab.nn.db?mode=ro" % T, uri=True)
        for n, c in con.execute("select name, code from nn"):
            src_db[n] = c
        canon = {n.replace("_", "-"): n for n in src_db if "_" in n}

    # rank candidates by MEASURED full-res quality, and require a checkpoint
    if cands is None:
      cands = []
      for n, p in sorted(fr.items(), key=lambda kv: -kv[1]):
        ck = os.path.join(T, "sweep_out/ckpt", n, "best_model.pth")
        if not os.path.exists(ck):
            continue
        code = src_db.get(n) or src_db.get(canon.get(n, ""))
        if code:
            cands.append((n, p, ck, code))
    print("candidates with full-res score AND a checkpoint: %d" % len(cands), flush=True)
    print("best single model: %s at %.3f dB" % (cands[0][0][:44], cands[0][1]), flush=True)

    ks = sorted({int(x) for x in a.topk.split(",")})
    need = max(ks)
    members = []
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(a.out), "src")
    os.makedirs(tmp, exist_ok=True)
    for n, p, ck, code in cands[:need]:
        f = os.path.join(tmp, n + ".py")
        open(f, "w").write(code)
        try:
            net = load_net(f, dev)
            sd = torch.load(ck, map_location=dev)
            net.load_state_dict(sd if not isinstance(sd, dict) or "state_dict" not in sd
                                else sd["state_dict"])
            net.eval()
            members.append((n, p, net))
            print("  loaded %-44s %.3f dB" % (n[:44], p), flush=True)
        except Exception as e:
            print("  SKIP %-44s %s: %s" % (n[:44], type(e).__name__, str(e)[:70]), flush=True)
        if len(members) >= need:
            break
    if not members:
        print("no members loaded"); return

    noisy = sorted(glob.glob(os.path.join(a.val, "original", "*")))
    clean = sorted(glob.glob(os.path.join(a.val, "denoised", "*")))
    assert len(noisy) == len(clean), "frame count mismatch"
    if a.limit_frames:
        noisy, clean = noisy[:a.limit_frames], clean[:a.limit_frames]
    print("\nframes: %d   members: %d   sweeping k=%s\n" % (len(noisy), len(members), ks),
          flush=True)

    # accumulate per-frame running means so each k is read off one forward pass per member
    sums = {k: 0.0 for k in ks}
    for i, (np_, cp) in enumerate(zip(noisy, clean), 1):
        x = to_tensor(np_).to(dev)
        ref = to_tensor(cp).to(dev)
        acc = torch.zeros_like(x)
        got = 0
        per_k = {}
        with torch.no_grad():
            for n, p, net in members:
                try:
                    y = net(x)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    y = forward_tiled_ladder(net, x)
                acc = acc + y.clamp(0, 1)
                got += 1
                if got in sums:
                    per_k[got] = psnr_rgb((acc / got).clamp(0, 1), ref, shave=1)
        for k, v in per_k.items():
            sums[k] += v
        del x, ref, acc
        torch.cuda.empty_cache()
        print("  frame %2d/%d  " % (i, len(noisy))
              + "  ".join("k=%d %.3f" % (k, per_k[k]) for k in sorted(per_k)), flush=True)

    print("\n=== ENSEMBLE RESULT (full-res RGB, 1px shave) ===")
    best_single = members[0][1]
    res = {}
    for k in sorted(sums):
        v = sums[k] / len(noisy)
        res[k] = v
        print("  k=%-3d  %.3f dB   (%+.3f vs best single, %+.3f vs published 37.660)"
              % (k, v, v - best_single, v - 37.660))
    json.dump({"best_single": best_single, "by_k": res,
               "members": [m[0] for m in members], "frames": len(noisy)},
              open(a.out, "w"), indent=1)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()

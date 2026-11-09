"""Prove that save_pth_weights=True actually lands a .pth on PERSISTENT storage.

Three separate things have to be true and none of them is checked anywhere else:
  1. check_nn accepts save_pth_weights and threads it to save_if_best (train_new hardcoded
     False until now, so no model trained through the API has ever written weights).
  2. LEMUR_CKPT_DIR redirects the destination. Our sweep pods root LEMUR at a pod-local
     emptyDir to keep concurrent writers off the NFS SQLite file; weights written under
     that root die with the pod, which would look exactly like success until you went to
     load one.
  3. The file that appears is a real state_dict that reloads into the same architecture.

save_to_db=False on purpose: this must not add a smoke-test row to the production DB.
"""
import os
import sys

NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"
MODEL = os.environ.get("SMOKE_MODEL", "DenoiseThunder")


def main():
    ckpt = os.environ.get("LEMUR_CKPT_DIR")
    print(f"LEMUR_CKPT_DIR = {ckpt or '(unset -> <root>/out/ckpt)'}")
    if not ckpt:
        print("FAIL: LEMUR_CKPT_DIR unset; weights would land on the pod-local root")
        return 1

    from ab.nn import api
    from ab.nn.util.Const import ckpt_dir
    print(f"resolved ckpt_dir = {ckpt_dir}")
    if str(ckpt_dir) != ckpt:
        print(f"FAIL: Const.ckpt_dir did not honour the env var")
        return 1

    code = open(os.path.join(NN_DIR, MODEL + ".py")).read()
    prm = {"lr": 1e-4, "batch": 16, "epoch": 1, "epoch_max": 1, "transform": "denoise"}
    name, acc, _a2t, _q = api.check_nn(
        code, "img-denoising", "denoise", "psnr", prm,
        save_to_db=False, prefix="SMOKEW", save_pth_weights=True)
    print(f"trained {name}  acc={acc:.4f}  ({acc * 48:.3f} dB)")

    hits = []
    for root, _d, files in os.walk(ckpt):
        for f in files:
            if f.endswith(".pth"):
                p = os.path.join(root, f)
                hits.append((p, os.path.getsize(p)))
    print(f"\n.pth files under {ckpt}: {len(hits)}")
    for p, s in sorted(hits)[-5:]:
        print(f"  {s / 1e6:8.2f} MB  {p}")
    mine = [h for h in hits if "SMOKEW" in h[0]]
    if not mine:
        print("FAIL: no checkpoint written for this run")
        return 1

    # A file of the right size is not proof it is loadable.
    import torch
    p = mine[0][0]
    sd = torch.load(p, map_location="cpu", weights_only=False)
    ns = {}
    exec(compile(code, MODEL, "exec"), ns)
    net = ns["Net"]((1, 3, 256, 256), (3,), dict(prm), "cpu")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real = [k for k in missing if not k.endswith("num_batches_tracked")]
    print(f"\nreload into fresh {MODEL}: {len(sd)} keys, "
          f"{len(real)} real-missing, {len(unexpected)} unexpected")
    if real or unexpected:
        print("FAIL: checkpoint does not match the architecture it came from")
        return 1
    print("\nPASS: weights saved to persistent storage and reload cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

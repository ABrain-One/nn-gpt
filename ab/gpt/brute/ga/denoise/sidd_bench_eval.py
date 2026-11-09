"""PHASE 2: evaluate trained checkpoints on the OFFICIAL SIDD sRGB validation blocks.

Why this run exists: none of our PSNR numbers are currently comparable to any published result.
`ab/nn/loader/sidd.py` scores on a split we invented (last 10% of SIDD-Medium by filename,
random 256 crops). Every SIDD number in the literature is measured on the 1280 fixed blocks in
ValidationNoisyBlocksSrgb.mat, so that is what this evaluates on -- see
`ab/nn/loader/sidd_bench.py` for the loader and the contamination argument.

EVALUATION ONLY. Nothing is trained here. Models are loaded from the checkpoints they already
have and run forward over the fixed blocks. Training on these blocks would be training on the
test set; the loader refuses to hand out a train split for exactly that reason.

WHAT TO EXPECT, so the numbers are not misread. The no-op (return the input untouched) scores
about 22.7 dB on these blocks, against 33.0 dB on our synthetic AWGN validation set: real
smartphone noise is far harder than the synthetic recipe our corpus was searched under. Published
SIDD sRGB SOTA is around 40 dB, from models three orders of magnitude larger than our 9,053 MMac
budget allows. Our models will land well below that and that is not a defect -- but it does mean
the only honest comparison is against other models in the same budget, which is Phase 3.

A SECOND, LARGER CAVEAT. Every model here was TRAINED on synthetic AWGN noise and is being tested
on real sensor noise. This measures cross-domain transfer, not SIDD performance -- a model trained
on SIDD-Medium would do far better. So these numbers are a lower bound on what the architectures
can do, and they must never be presented as "our method's SIDD result" without that sentence
attached. The cross-dataset study (0.27 dB selection cost) is the evidence that AWGN-trained
ranking transfers; absolute PSNR does not.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch


def find_ckpt(name, stores):
    for base in stores:
        p = os.path.join(base, name, "best_model.pth")
        if os.path.isfile(p):
            return p
    # prefix-renamed checkpoint dirs (check_nn(prefix=...) renames the model, so the directory is
    # <prefix>-<hash> rather than <family>-<hash>); match on the trailing hash.
    suffix = name.rsplit("-", 1)[-1]
    if len(suffix) >= 8:
        for base in stores:
            if not os.path.isdir(base):
                continue
            for d in os.listdir(base):
                if d.endswith(suffix):
                    p = os.path.join(base, d, "best_model.pth")
                    if os.path.isfile(p):
                        return p
    return None


def load_net(src, dev):
    ns = {}
    exec(compile(src, "<model>", "exec"), ns)
    Net = ns.get("Net")
    if Net is None:
        raise RuntimeError("no Net in source")
    # in_shape is BATCHED (N,C,H,W) -- Train.py:210. Passing (3,256,256) makes every model that
    # reads in_shape[1] build a 256-channel stem; that bug cost 5 of 16 rows in the re-export.
    return Net((1, 3, 256, 256), (3,), {"lr": 1e-4, "batch": 2, "dropout": 0.0,
                                        "momentum": 0.9}, dev).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit model names; default = --top-n best edge-legal from the DB")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--models-file", default=None,
                    help="newline-separated model names. PREFER THIS over --top-n: --top-n ranks "
                         "by the DB's max(accuracy), and the DB holds 50-epoch rows only for the "
                         "hand-written families, so it surfaces baselines and skips every searched "
                         "champion (whose converged results live in fulltrain50/beatbar).")
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--out", default="/a/mm/sweep_out/sidd_bench/results.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 1280 blocks")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sys.path.insert(0, "/a/mm/nn-dataset")
    # __import__ with a string, not `from ab.nn.loader.sidd-bench import ...`: the
    # loader is named sidd-bench.py because '_' is LEMUR's stat-dir field separator
    # and a dataset named `sidd_bench` produces a 5-field stat dir that Util.py:61
    # cannot unpack. A hyphen is not a Python identifier, so the import-statement
    # form is a SyntaxError -- but the import *system* accepts the name fine, which
    # is exactly how upstream loads cifar-10.py and celeba-gender.py. See naming.py.
    SiddBenchBlocks = getattr(
        __import__("ab.nn.loader.sidd-bench", fromlist=["SiddBenchBlocks"]),
        "SiddBenchBlocks")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SiddBenchBlocks()
    n_blocks = len(ds) if not a.limit else min(a.limit, len(ds))
    print(f"device {dev} | official SIDD blocks: {len(ds)} (using {n_blocks})", flush=True)

    import sqlite3
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    names = a.models
    if not names and a.models_file:
        names = [l.strip() for l in open(a.models_file) if l.strip()]
    if not names:
        # top-N by best recorded PSNR, any producer; edge-legality is a property of the model and
        # was already enforced when it was generated, so no re-filter here.
        names = [n for (n,) in con.execute(
            """select nn from stat where task='img-denoising' and metric='psnr'
               group by nn order by max(accuracy) desc limit ?""", (a.top_n,))]
    print(f"models to evaluate: {len(names)}", flush=True)

    STORES = ["/a/mm/sweep_out/reexport/ckpt", "/a/mm/sweep_out/ckpt",
              "/a/mm/nn-dataset/out/ckpt", "/a/mm/sweep_out/baselines/ckpt"]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    print(f"already evaluated: {len(done)}", flush=True)

    for name in names:
        if name in done:
            continue
        rec = {"model": name, "n_blocks": n_blocks, "protocol": "SIDD-official-validation-srgb",
               "status": "FAIL"}
        t0 = time.time()
        try:
            row = con.execute("select code from nn where name=?", (name,)).fetchone()
            if not row:
                raise RuntimeError("not in DB")
            ck = find_ckpt(name, STORES)
            if not ck:
                rec["error"] = "no checkpoint in any store"
                print(f"SKIP  {name[:44]:44} no checkpoint", flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                continue
            rec["ckpt"] = ck
            net = load_net(row[0], dev)
            sd = torch.load(ck, map_location=dev)
            if isinstance(sd, dict):
                sd = sd.get("model_state_dict", sd.get("state_dict", sd))
            missing, unexpected = net.load_state_dict(sd, strict=False)
            # A checkpoint matching almost nothing loads fine under strict=False and then scores
            # like an untrained net. Refuse rather than publish that number.
            # ANY missing key means this checkpoint was not saved from this architecture, and the
            # unset tensors are random init. Three rows in the first run passed a 50%-of-keys
            # threshold with 48/78/108 missing keys and scored 24.3/24.1/20.6 dB against a 22.7 dB
            # no-op -- one of them literally worse than returning the input. There is no safe
            # nonzero tolerance here, so require an exact match.
            if missing:
                raise RuntimeError(
                    f"checkpoint mismatch: {len(missing)} missing keys "
                    f"(e.g. {list(missing)[:3]}) -- refusing to score partially-loaded weights")
            rec["missing_keys"], rec["unexpected_keys"] = len(missing), len(unexpected)
            net.eval()

            # Per-block PSNR then averaged, which is what the SIDD protocol reports -- NOT the
            # PSNR of the pooled MSE. The two differ, and mixing them is a silent ~0.1-0.3 dB.
            tot, worst, best = 0.0, 1e9, -1e9
            with torch.no_grad():
                for i in range(0, n_blocks, a.batch):
                    xs, ys = [], []
                    for j in range(i, min(i + a.batch, n_blocks)):
                        x, y = ds[j]
                        xs.append(x)
                        ys.append(y)
                    x = torch.stack(xs).to(dev)
                    y = torch.stack(ys).to(dev)
                    p = net(x).clamp(0, 1)
                    mse = torch.mean((p - y) ** 2, dim=(1, 2, 3))
                    ps = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
                    tot += float(ps.sum())
                    worst = min(worst, float(ps.min()))
                    best = max(best, float(ps.max()))
            rec.update(status="OK", psnr_srgb=round(tot / n_blocks, 4),
                       psnr_min=round(worst, 3), psnr_max=round(best, 3))
            print(f"OK    {name[:44]:44} {rec['psnr_srgb']:>8.3f} dB "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL  {name[:44]:44} {rec['error'][:70]}", flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

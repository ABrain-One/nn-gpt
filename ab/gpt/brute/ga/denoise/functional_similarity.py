"""Functional-similarity (behavioural) index for the denoise models.

Structural fingerprints (scratchpad/seed_similarity.py) show the seeds *look*
different. This layer proves they *behave* differently: feed ONE fixed batch of
real noisy patches through every model and compare the denoised outputs. Two
architectures that map noisy->clean identically are the same model for our
purposes, no matter how their source differs — so this is the strongest
"not near-duplicate" evidence, and it needs no training (one forward each).

Two complementary pairwise measures:
  * linear CKA (Kornblith et al. 2019) on the output feature maps -- 1.0 = the
    two models' output representations are identical up to orthogonal transform;
    invariant to channel permutation / isotropic scaling. Minibatch Gram form.
  * output-agreement PSNR (dB) -- how close two models' denoised images are to
    EACH OTHER (not to ground truth). High dB = they produce the same picture.

Run with the nn-dataset venv python and both repos importable, e.g.:
    export PYTHONPATH=/a/mm/nn-gpt/ab/gpt/brute/ast:/a/mm/nn-dataset
    python functional_similarity.py \
        --seeds DenoiseUnet1F ... --models-dir seed_ga_models --n 32
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")

NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"


def log(m):
    print(m, flush=True)


def get_noisy_batch(n, seed=0):
    """Fixed batch of `n` real noisy patches (noisy_img, ...) from the denoise
    test set. Deterministic selection so every run/model sees identical input."""
    import torch
    # Import loader + transform directly from disk (the DB-backed load_dataset
    # regenerates them from a `loader` table that isn't populated in this env).
    from ab.nn.loader.denoise import loader
    from ab.nn.transform.denoise import transform
    _, _, _train, test = loader(transform, "img-denoising")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test), generator=g)[:n].tolist()
    noisy = torch.stack([test[i][0] for i in idx])  # (n,3,H,W)
    clean = torch.stack([test[i][1] for i in idx])
    return noisy, clean


def _disable_db_prm_lookup():
    """load_lemur_model calls ab.nn.api.data(only_best_accuracy=False) on EVERY
    model load to fetch that code's stored prm (~2 min full-DB scan each). For a
    forward-only pass we don't need it — default prm instantiates the denoise
    models fine — so stub it to an empty frame once and save ~2 min per model."""
    import pandas as pd
    import mutator.lemur_io as lio
    empty = pd.DataFrame(columns=["nn", "nn_code", "accuracy", "prm"])
    lio.nn_dataset.data = lambda *a, **k: empty


def model_outputs(source, noisy):
    """Load a model from source, forward the fixed noisy batch, return its
    denoised output as a CPU float tensor (n, 3, H, W). None on failure."""
    import torch
    from mutator.lemur_io import load_lemur_model, cleanup_temp_module
    model = None
    try:
        model = load_lemur_model(source)
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            out = model(noisy.to(dev))
        return out.detach().float().cpu()
    except Exception as e:
        log(f"    (load/forward failed: {type(e).__name__}: {str(e)[:80]})")
        return None
    finally:
        if model is not None:
            info = getattr(model, "_temp_module_info", None)
            if info:
                try:
                    cleanup_temp_module(*info)
                except Exception:
                    pass


def gram_centered(x):
    """Centered linear Gram matrix K_c = H (X X^T) H for X of shape (n, p)."""
    import torch
    x = x.reshape(x.shape[0], -1)
    k = x @ x.t()                       # (n, n) linear Gram
    n = k.shape[0]
    h = torch.eye(n) - torch.ones(n, n) / n
    return h @ k @ h


def linear_cka(kc, lc):
    """CKA from two pre-centered Gram matrices = <Kc,Lc>_F / (||Kc||*||Lc||)."""
    import torch
    hsic = (kc * lc).sum()
    denom = kc.norm() * lc.norm()
    return float(hsic / denom) if denom > 0 else 0.0


def out_psnr(a, b):
    """Agreement PSNR (dB) between two models' outputs, treating one as ref.
    Inputs assumed roughly in [0,1]; uses observed dynamic range as the peak."""
    import torch
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    peak = float((a.max() - a.min()).clamp(min=1e-3))
    return 10.0 * (torch.log10(torch.tensor(peak * peak / mse))).item()


def resolve_specs(seeds, models_dir):
    """Return list of (label, source). Seeds resolved from nn dir; models_dir
    globs *.py (checksum-named GA mutants), label from their header name= line."""
    specs = []
    for nm in seeds:
        p = os.path.join(NN_DIR, nm + ".py")
        if os.path.exists(p):
            specs.append((nm, open(p).read()))
        else:
            log(f"  (seed {nm} not found — skipped)")
    if models_dir and os.path.isdir(models_dir):
        for p in sorted(glob.glob(os.path.join(models_dir, "*.py"))):
            src = open(p).read()
            label = os.path.basename(p)[:-3]
            first = src.splitlines()[0] if src else ""
            if first.startswith("# name="):
                label = first.split("name=", 1)[1].split()[0] + "/" + label[:6]
            specs.append((label, src))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*",
                    default=[f"DenoiseUnet{i}F" for i in range(1, 11)] + ["LiteDenoiseNet"])
    ap.add_argument("--models-dir", default=None,
                    help="also include every *.py in this dir (GA mutant sources)")
    ap.add_argument("--n", type=int, default=32, help="noisy patches in the fixed batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "functional_similarity.json"))
    args = ap.parse_args()

    specs = resolve_specs(args.seeds, args.models_dir)
    log(f"=== Functional similarity (CKA + agreement-PSNR) | {len(specs)} models, "
        f"n={args.n} patches ===")

    noisy, clean = get_noisy_batch(args.n, args.seed)
    log(f"fixed batch: {tuple(noisy.shape)}")
    _disable_db_prm_lookup()  # skip the per-model full-DB prm scan (forward-only)

    # Two representations per model:
    #   output   = denoised image (o)          -> dominated by the shared clean
    #              signal, so output-CKA saturates near 1.0 for all good models.
    #   residual = what the model REMOVED (o - noisy) -> subtracts the common
    #              clean target, leaving the model-specific action. This is the
    #              DISCRIMINATIVE representation for "do these behave the same".
    labels, grams_o, grams_r, outs = [], [], [], []
    for label, src in specs:
        log(f"  forward {label}")
        o = model_outputs(src, noisy)
        if o is None:
            continue
        labels.append(label)
        outs.append(o)
        grams_o.append(gram_centered(o))
        grams_r.append(gram_centered(o - noisy))

    m = len(labels)
    if m < 2:
        log("need >=2 loadable models — abort.")
        return

    cka = [[0.0] * m for _ in range(m)]        # residual-CKA (discriminative)
    cka_o = [[0.0] * m for _ in range(m)]      # output-CKA (saturates ~1.0)
    psnr = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            cka[i][j] = linear_cka(grams_r[i], grams_r[j])
            cka_o[i][j] = linear_cka(grams_o[i], grams_o[j])
            psnr[i][j] = out_psnr(outs[i], outs[j])

    def short(l):
        return l.replace("DenoiseUnet", "").replace("LiteDenoiseNet", "Lite")[:7]

    log("\n=== RESIDUAL CKA (out-noisy; discriminative: 1.00 = same denoising action) ===")
    log("".ljust(16) + "".join(short(l).rjust(8) for l in labels))
    for i, li in enumerate(labels):
        log(li[:16].ljust(16) + "".join(f"{cka[i][j]:8.2f}" for j in range(m)))

    log("\n=== output CKA (saturates ~1.0 — shared clean signal dominates; shown for contrast) ===")
    log("".ljust(16) + "".join(short(l).rjust(8) for l in labels))
    for i, li in enumerate(labels):
        log(li[:16].ljust(16) + "".join(f"{cka_o[i][j]:8.2f}" for j in range(m)))

    log("\n=== output-agreement PSNR dB (higher = more identical images; ~99 = same) ===")
    log("".ljust(16) + "".join(short(l).rjust(8) for l in labels))
    for i, li in enumerate(labels):
        log(li[:16].ljust(16) + "".join(f"{psnr[i][j]:8.1f}" for j in range(m)))

    # diversity summary (on the discriminative residual-CKA)
    offc = [cka[i][j] for i in range(m) for j in range(m) if i != j]
    offco = [cka_o[i][j] for i in range(m) for j in range(m) if i != j]
    offp = [psnr[i][j] for i in range(m) for j in range(m) if i != j]
    log("\n=== nearest functional neighbour (highest off-diagonal RESIDUAL CKA) ===")
    for i, li in enumerate(labels):
        j = max((k for k in range(m) if k != i), key=lambda k: cka[i][k])
        log(f"  {li[:18].ljust(18)} ~ {labels[j][:18].ljust(18)} "
            f"resCKA={cka[i][j]:.3f}  agree={psnr[i][j]:.1f} dB")
    log(f"\nmean off-diagonal RESIDUAL CKA = {sum(offc)/len(offc):.3f} "
        f"(discriminative; 1.0 = same denoising action)")
    log(f"max  off-diagonal RESIDUAL CKA = {max(offc):.3f}")
    log(f"mean off-diagonal output   CKA = {sum(offco)/len(offco):.3f} "
        f"(saturated by shared clean signal — not discriminative)")
    log(f"mean output-agreement = {sum(offp)/len(offp):.1f} dB "
        f"(low dB = models genuinely produce different images)")

    json.dump({"labels": labels, "residual_cka": cka, "output_cka": cka_o,
               "agree_psnr_db": psnr, "n": args.n,
               "mean_offdiag_residual_cka": sum(offc)/len(offc),
               "max_offdiag_residual_cka": max(offc),
               "mean_offdiag_output_cka": sum(offco)/len(offco)},
              open(args.out, "w"), indent=1)
    log(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

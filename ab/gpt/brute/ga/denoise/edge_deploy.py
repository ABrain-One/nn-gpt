"""Fast STATIC edge-deployability check — gate models at GENERATION time.

Thesis target = resource-constrained edge devices (mobile SoC, TFLite/LiteRT on the
GPU delegate). Deployability must be a HARD FILTER during search, not a post-hoc check
on winners: an undeployable model is not a result. Full conversion (to_tflite.py via
LiteRT) is ground-truth but seconds-per-model + heavy deps — too slow to gate every
candidate. This module is the cheap proxy: instantiate on CPU (~1s, no training) and
scan the module tree for the ops known to break / stall the GPU delegate.

Ruleset (from the seeds' own notes + observed conversion failures):
  HARD (conversion FAILS or is unusable on-device):
    * nn.ConvTranspose2d with output_padding != 0   -> LiteRT conversion failure
      (this is the exact op that kills 4/11 seeds; use PixelShuffle/Upsample instead)
  SOFT (converts but forces CPU fallback = defeats the GPU delegate → slow):
    * nn.LayerNorm / nn.GroupNorm / nn.InstanceNorm  (BatchNorm2d is the clean choice;
      it folds into the preceding conv — see DenoiseUnet9F._Norm2d comment)
  Plus an optional hard MAC cap tied to the target device (passed in from the caller,
  which already computes MACs — we don't recompute).

Strictness: `strict=True` treats SOFT ops as hard fails too (pure-GPU-delegate target);
default treats them as warnings. Our skip operators emit only element-wise adds
(residuals) → TFLite-clean, so they never trip this.

API: deployability(source, strict=False) -> dict(deployable, hard, soft)
     macs_ok(macs, mac_cap_mmac) -> bool
"""
import torch
import torch.nn as nn


def _instantiate(source):
    """exec the source and build Net on CPU. Returns the nn.Module or None."""
    ns = {}
    try:
        exec(compile(source, "<edge>", "exec"), ns)
        if "Net" not in ns:
            return None
        return ns["Net"](in_shape=(1, 3, 256, 256), prm={"lr": 1e-4}, device="cpu")
    except Exception:
        return None


def deployability(source, strict=False):
    """Scan the module tree for GPU-delegate-hostile ops.
    Returns {deployable: bool, hard: [str], soft: [str]}.
    deployable = no hard fails (and, if strict, no soft ops either)."""
    model = _instantiate(source)
    if model is None:
        # cannot even build -> not deployable (and the trainer would reject it anyway)
        return {"deployable": False, "hard": ["instantiation-failed"], "soft": []}

    hard, soft = [], []
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            op = m.output_padding
            if any(p != 0 for p in (op if isinstance(op, (tuple, list)) else (op,))):
                hard.append(f"ConvTranspose2d(output_padding={op})")
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                            nn.InstanceNorm2d, nn.InstanceNorm3d)):
            soft.append(type(m).__name__)

    deployable = not hard and (not strict or not soft)
    return {"deployable": deployable, "hard": hard, "soft": soft}


def macs_ok(macs, mac_cap_mmac):
    """True if macs (raw MACs) is within the device cap (in MMac). cap<=0 disables."""
    if not mac_cap_mmac or mac_cap_mmac <= 0:
        return True
    if macs is None:
        return True  # unknown cost -> don't block on it here
    return (macs / 1e6) <= mac_cap_mmac


if __name__ == "__main__":
    import os
    import sys
    NN = "/a/mm/nn-dataset/ab/nn/nn"
    sys.path.insert(0, "/a/mm/nn-dataset")
    seeds = sys.argv[1:] or [f"{i}F" for i in range(1, 11)]
    print(f"{'seed':7}{'deployable':12}{'hard fails':28}{'soft (CPU-fallback)':20}")
    print("-" * 67)
    for s in seeds:
        p = os.path.join(NN, f"DenoiseUnet{s}.py")
        if not os.path.exists(p):
            continue
        r = deployability(open(p).read())
        print(f"{s:7}{('YES' if r['deployable'] else 'NO'):12}"
              f"{(', '.join(r['hard']) or '-'):28}{(', '.join(r['soft']) or '-'):20}")

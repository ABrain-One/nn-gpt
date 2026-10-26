"""Architecture descriptors of a model's code, computed the way nn-dataset fills
the nn_stat table (ab/nn/util/NNAnalysis.py: total_params is the sum of parameter
sizes, total_layers is len(list(model.modules()))). The model is built on CPU
with the input shape its transform produces (batch 1, 3 channels)."""
import re

import torch

_DEFAULT_PRM = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2, "batch": 64}


def image_size(transform, default: int = 32) -> int:
    """Final side length a transform name produces: norm_32_flip -> 32,
    norm_256_224_flip -> 224 (last number is the crop). Unknown -> default."""
    nums = re.findall(r"\d+", str(transform or ""))
    return int(nums[-1]) if nums else default


def nn_descriptors(code: str, prm=None, transform=None, n_classes: int = 10):
    """Return (total_params, total_layers) for the Net defined in code."""
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(code, "<nn>", "exec"), scope, scope)
    full_prm = dict(_DEFAULT_PRM)
    full_prm.update(prm or {})
    side = image_size(transform or full_prm.get("transform"))
    with torch.device("cpu"):
        model = scope["Net"]((1, 3, side, side), (n_classes,), full_prm, torch.device("cpu"))
    return int(sum(p.numel() for p in model.parameters())), len(list(model.modules()))

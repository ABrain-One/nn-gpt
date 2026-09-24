"""
Compute parameter count and FLOPs for a generated NN file.

Usage (in container with torch):
    from ab.gpt.util.eval.nn_cost import get_nn_cost
    params, flops = get_nn_cost("/path/to/new_nn.py", dataset="cifar-10")
    # Returns (None, None) on any failure.
"""

import importlib.util
import os
import sys
import types
from typing import Optional, Tuple

# Native input resolutions per dataset (C, H, W)
DATASET_INPUT_SHAPE = {
    "mnist":          (1,  28,  28),
    "svhn":           (3,  32,  32),
    "cifar-10":       (3,  32,  32),
    "cifar10":        (3,  32,  32),
    "cifar-100":      (3,  32,  32),
    "cifar100":       (3,  32,  32),
    "celeba-gender":  (3, 128, 128),
    "celeba_gender":  (3, 128, 128),
    "imagenette":     (3, 224, 224),
    "places365":      (3, 256, 256),
}

# Number of output classes per dataset
DATASET_NUM_CLASSES = {
    "mnist":          10,
    "svhn":           10,
    "cifar-10":       10,
    "cifar10":        10,
    "cifar-100":      100,
    "cifar100":       100,
    "celeba-gender":  2,
    "celeba_gender":  2,
    "imagenette":     10,
    "places365":      365,
}


def _load_net_class(nn_file: str):
    """Dynamically load the Net class from a generated NN file."""
    spec = importlib.util.spec_from_file_location("_nn_module_tmp", nn_file)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = nn_file
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Net"):
        raise AttributeError(f"No 'Net' class found in {nn_file}")
    return mod.Net


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_flops_fvcore(model, input_tensor) -> Optional[int]:
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, input_tensor)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total())
    except Exception:
        return None


def _count_flops_ptflops(model, input_shape_chw) -> Optional[int]:
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, input_shape_chw, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        return int(macs) if macs is not None else None
    except Exception:
        return None


def _count_flops_thop(model, input_tensor) -> Optional[int]:
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return int(macs) if macs is not None else None
    except Exception:
        return None


def get_nn_cost(
    nn_file: str,
    dataset: str,
    batch_size: int = 1,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (param_count, flops_macs).
    Both are None if instantiation fails or torch is unavailable.

    flops_macs = MACs (multiply-accumulate ops) at dataset native resolution.
    Tries fvcore → ptflops → thop in order; None if all unavailable.
    """
    if not os.path.isfile(nn_file):
        return None, None

    try:
        import torch
    except ImportError:
        return None, None

    dataset_key = dataset.replace("-", "_").lower() if dataset else ""
    shape_chw = DATASET_INPUT_SHAPE.get(dataset, DATASET_INPUT_SHAPE.get(dataset_key))
    num_classes = DATASET_NUM_CLASSES.get(dataset, DATASET_NUM_CLASSES.get(dataset_key, 10))

    if shape_chw is None:
        return None, None

    try:
        Net = _load_net_class(nn_file)
    except Exception:
        return None, None

    try:
        prm = {"num_classes": num_classes}
        model = Net(prm)
        model.eval()
    except Exception:
        try:
            model = Net()
            model.eval()
        except Exception:
            return None, None

    param_count = _count_params(model)

    try:
        dummy = torch.zeros(batch_size, *shape_chw)
        flops = (
            _count_flops_fvcore(model, dummy)
            or _count_flops_ptflops(model, shape_chw)
            or _count_flops_thop(model, dummy)
        )
    except Exception:
        flops = None

    return param_count, flops


def batch_cost(nn_file: str, dataset: str) -> dict:
    """Return a dict with param_count and flop_count (JSON-serialisable)."""
    params, flops = get_nn_cost(nn_file, dataset)
    return {"param_count": params, "flop_count": flops}

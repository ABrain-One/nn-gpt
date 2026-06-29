"""Shared forward-pass structural validation for generated LEMUR Net models."""

from __future__ import annotations

from typing import Dict, Tuple


def quick_structural_check(
    code: str,
    channels: int = 3,
    hw: int = 32,
    num_classes: int = 10,
) -> Tuple[bool, str]:
    """
    Pre-flight a generated model the way nn-dataset's evaluator does:
    instantiate Net(in_shape, out_shape, prm, device) with in_shape =
    (batch=1, channels, H, W), then forward a real batch.

    Returns (ok, message).
    """
    import torch as _torch

    ns: Dict = {}
    try:
        exec(
            "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
            "from torch import Tensor\n",
            ns,
        )
        exec(code, ns)
    except Exception as e:
        return False, f"exec_error: {type(e).__name__}: {e}"

    Net = ns.get("Net")
    if Net is None:
        return False, "no_Net_class"

    device = _torch.device("cpu")
    prm = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2}
    eval_in_shape = (1, channels, hw, hw)
    try:
        model = Net(eval_in_shape, (num_classes,), prm, device)
    except Exception as e:
        return False, f"init_error: {type(e).__name__}: {e}"

    try:
        model.eval()
        x = _torch.randn(2, channels, hw, hw)
        with _torch.no_grad():
            y = model(x)
    except Exception as e:
        return False, f"forward_error: {type(e).__name__}: {e}"

    shape = tuple(getattr(y, "shape", ()))
    if shape != (2, num_classes):
        return False, f"bad_output_shape: got {shape}, want (2, {num_classes})"
    return True, "ok"

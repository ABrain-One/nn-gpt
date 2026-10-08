#!/usr/bin/env python3
"""
Zero-cost (training-free) architecture proxies for AccPredictor.py.

Given only nn_code (raw PyTorch source), a dataset name, and a hyperparameter
dict, this module instantiates the network at random initialization and
computes a handful of scalar scores that are known to correlate with a
network's eventual trained accuracy -- without running any training.

Standalone by design: does not import from AccPredictor.py and does not
modify the existing pipeline. Intended to be tested independently first
(see __main__ below), then wired into AccPredictor.py's prompt construction
as a separate, additive step.

Proxies implemented:
  - synflow   : Tanaka et al. 2020, data-free saliency (all-ones input)
  - nwot      : Mellor et al. 2021 ("NAS without training"), activation-pattern
                kernel log-determinant on a batch of random inputs
  - grad_norm : L2 norm of gradients from one real forward/backward pass
  - log_params, depth : cheap static architecture fingerprints

Usage:
    python zero_cost_proxies.py
    (reads a few real records from out/acc_predict/llm_finetuning_data.jsonl
     and prints proxy scores, to sanity-check that scores vary across
     architectures before wiring this into the training pipeline)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ab.nn.util.Util import torch_device as _ab_torch_device
except ImportError:
    _ab_torch_device = None


# --- dataset -> (in_shape, out_shape) lookup -----------------------------
# Real image sizes per the LEMUR dataset loaders (nn-dataset/ab/nn/loader/).
# Batch dim is a placeholder; it is overridden per-proxy call.
# wikitext/coco are non-image tasks (language modeling / detection) --
# proxies will typically fail to instantiate cleanly for these, which is
# expected and handled as a best-effort failure (see compute_proxies).
_DATASET_SHAPES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "cifar-10": ((1, 3, 32, 32), (10,)),
    "cifar-100": ((1, 3, 32, 32), (100,)),
    "mnist": ((1, 1, 28, 28), (10,)),
    "svhn": ((1, 3, 32, 32), (10,)),
    "imagenette": ((1, 3, 224, 224), (10,)),
    "celeba-gender": ((1, 3, 224, 224), (2,)),
    "places365": ((1, 3, 224, 224), (365,)),
}
_DEFAULT_SHAPE = ((1, 3, 224, 224), (10,))

_DEFAULT_PRM = {"lr": 0.01, "momentum": 0.9, "dropout": 0.5, "batch": 8}


def _get_shapes(dataset: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _DATASET_SHAPES.get((dataset or "").lower(), _DEFAULT_SHAPE)


def _shape_candidates(dataset: str) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Increasing-size fallback shapes for one dataset. Needed because the real
    training input size for a given dataset depends on transform_code (which
    can resize), not just the dataset's native size -- some architectures are
    deep enough that a native 32x32 input (e.g. cifar/svhn) collapses to a
    spatial size smaller than a later kernel before the forward pass
    completes. Rather than assume one fixed shape per dataset, retry once or
    twice at larger sizes before giving up on an architecture.
    """
    in_shape, out_shape = _get_shapes(dataset)
    _, c, h, w = in_shape
    sizes = sorted({h, 96, 224})
    return [((1, c, s, s), out_shape) for s in sizes]


def torch_device() -> torch.device:
    if _ab_torch_device is not None:
        return _ab_torch_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_UNRECOVERABLE_ERROR_TYPES = (SyntaxError, NameError, ImportError, ModuleNotFoundError, AttributeError, TypeError)


def _instantiate_model(
    nn_code: str,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    prm: Optional[dict],
    device: torch.device,
) -> nn.Module:
    namespace: dict[str, Any] = {}
    exec(compile(nn_code, "<nn_code>", "exec"), namespace)
    if "Net" not in namespace:
        raise ValueError("nn_code does not define a class named 'Net'")
    ModelClass = namespace["Net"]

    safe_prm = {**_DEFAULT_PRM, **(prm or {})}
    model = ModelClass(in_shape, out_shape, safe_prm, device)
    model.to(device)
    return model


# --- static architecture fingerprint --------------------------------------

def _count_depth(model: nn.Module) -> int:
    depths = [len(list(model.modules()))]
    return max(depths) if depths else 0


def _static_features(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "log_params": math.log(total_params + 1),
        "depth": _count_depth(model),
    }


# --- SynFlow ----------------------------------------------------------------
# Tanaka et al. 2020. Linearizes the network (abs of all trainable params),
# forward-passes an all-ones input, backprops the summed output, and scores
# each parameter by |weight * grad|. Purely data-free -- no labels, no real
# images -- which is exactly why it transfers across architecture families.

def _linearize(model: nn.Module) -> dict[str, torch.Tensor]:
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data = param.data.abs()
    return signs


def _nonlinearize(model: nn.Module, signs: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in signs:
            param.data = param.data * signs[name]


def _synflow_forward_backward(model: nn.Module, in_shape: tuple[int, ...], device: torch.device, dtype) -> float:
    x = torch.ones((1, *in_shape[1:]), dtype=dtype, device=device)
    model.zero_grad()
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    torch.sum(output).backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p * p.grad)).item()
    return score


def _synflow_score(model: nn.Module, in_shape: tuple[int, ...], device: torch.device) -> float:
    # Linearization (all trainable weights -> abs value) amplifies forward
    # activations layer over layer; for deep, concatenation-heavy networks
    # (e.g. DenseNet, RegNet -- confirmed empirically on 32/1550 cached
    # architectures) this overflows float32 to inf, and inf-inf in backprop
    # then yields nan. Computing in double precision first avoids most of
    # this. Some architectures (e.g. RNN/LSTM via cuDNN) do not support
    # double precision on GPU, so fall back to the original dtype if the
    # double-precision attempt fails or still produces a non-finite score.
    model.eval()
    orig_dtype = next(model.parameters()).dtype
    signs = _linearize(model)
    try:
        try:
            model.to(torch.double)
            score = _synflow_forward_backward(model, in_shape, device, torch.double)
            if not math.isfinite(score):
                raise ValueError(f"non-finite synflow score in double precision: {score}")
        except Exception:
            model.to(orig_dtype)
            score = _synflow_forward_backward(model, in_shape, device, orig_dtype)
            if not math.isfinite(score):
                raise ValueError(f"non-finite synflow score: {score}")
        return score
    finally:
        model.to(orig_dtype)
        _nonlinearize(model, signs)
        model.zero_grad()


# --- NWOT / NASWOT -----------------------------------------------------------
# Mellor et al. 2021. Runs a batch of random inputs through the untrained
# network, records the binary (active/inactive) activation pattern per sample,
# and scores the network by how distinctly it separates different inputs --
# log-determinant of the resulting binary-activation kernel matrix.
#
# Coverage note: many LEMUR architectures call activations functionally
# (F.relu(x) inside forward()) instead of using an nn.ReLU module, so a
# module-hook-only approach misses them entirely (verified empirically:
# 3/8 sampled architectures used only F.relu). F.relu/F.relu_ are therefore
# monkey-patched for the duration of the forward pass to capture those too,
# in addition to hooking ReLU-family activation modules.

_ACTIVATION_MODULE_TYPES = (
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.ELU,
    nn.SiLU, nn.Hardswish, nn.Mish, nn.PReLU,
)


def _nwot_score(
    model: nn.Module, in_shape: tuple[int, ...], device: torch.device, batch_size: int = 8
) -> float:
    codes: list[torch.Tensor] = []

    def record(t: torch.Tensor) -> None:
        if isinstance(t, torch.Tensor) and t.dim() >= 1:
            codes.append((t > 0).float().reshape(t.size(0), -1))

    def module_hook(_module, _inp, out):
        record(out)

    handles = [
        m.register_forward_hook(module_hook)
        for m in model.modules()
        if isinstance(m, _ACTIVATION_MODULE_TYPES)
    ]

    orig_relu, orig_relu_ = F.relu, F.relu_

    def patched_relu(input, inplace=False):
        out = orig_relu(input, inplace=inplace)
        record(out)
        return out

    def patched_relu_(input):
        out = orig_relu_(input)
        record(out)
        return out

    F.relu = patched_relu
    F.relu_ = patched_relu_
    try:
        x = torch.randn((batch_size, *in_shape[1:]), device=device)
        model.eval()
        with torch.no_grad():
            model(x)

        if not codes:
            raise ValueError(
                "no activations captured (no ReLU-family modules or F.relu calls found)"
            )

        full = torch.cat(codes, dim=1).double()
        n = full.size(0)
        same = full @ full.t()
        diff = (1 - full) @ (1 - full).t()
        kernel = same + diff  # kernel[i,j] = # of matching activation bits between i,j

        sign, logdet = torch.linalg.slogdet(kernel + 1e-6 * torch.eye(n, device=device).double())
        if sign.item() == 0:
            raise ValueError("singular activation kernel")
        return logdet.item()
    finally:
        F.relu, F.relu_ = orig_relu, orig_relu_
        for h in handles:
            h.remove()


# --- grad_norm ---------------------------------------------------------------
# One real forward/backward pass on random data with random labels (no actual
# dataset needed, only the correct tensor shapes). Sum of L2 norms of the
# resulting gradients -- a cheap proxy for how much useful signal a single
# batch produces at initialization.

def _grad_norm_score(
    model: nn.Module,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int = 8,
) -> float:
    model.train()
    x = torch.randn((batch_size, *in_shape[1:]), device=device)
    num_classes = out_shape[0] if out_shape else 10
    y = torch.randint(0, max(num_classes, 2), (batch_size,), device=device)

    model.zero_grad()
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = F.cross_entropy(out, y)
    loss.backward()

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    model.zero_grad()
    return total ** 0.5


# --- SNIP ---------------------------------------------------------------------
# Lee et al. 2019 (connection sensitivity). Like grad_norm, one real labeled
# batch and one backward pass -- but scores each parameter by |weight *
# gradient| (a saliency, same formula as SynFlow) rather than the gradient
# magnitude alone. Unlike SynFlow, this uses real random data + labels, not
# an all-ones synthetic input.

def _snip_score(
    model: nn.Module, in_shape: tuple[int, ...], out_shape: tuple[int, ...],
    device: torch.device, batch_size: int = 8,
) -> float:
    model.train()
    x = torch.randn((batch_size, *in_shape[1:]), device=device)
    num_classes = out_shape[0] if out_shape else 10
    y = torch.randint(0, max(num_classes, 2), (batch_size,), device=device)

    model.zero_grad()
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = F.cross_entropy(out, y)
    loss.backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p * p.grad)).item()
    model.zero_grad()
    return score


# --- Fisher -------------------------------------------------------------------
# Theis et al. 2018 / Turner et al. 2020 style: approximates each layer's
# Fisher information from the activations themselves, not the raw weights --
# hooks ReLU-family outputs (same activation-detection machinery as NWOT),
# retains their gradients, and scores each layer by how much a unit's
# activation-times-its-gradient varies, summed over all captured layers.

def _fisher_score(
    model: nn.Module, in_shape: tuple[int, ...], out_shape: tuple[int, ...],
    device: torch.device, batch_size: int = 8,
) -> float:
    # Same F.relu monkey-patching as NWOT (_nwot_score) -- many architectures
    # in this dataset call F.relu(x) functionally inside forward() rather
    # than using an nn.ReLU module, so a module-hook-only approach misses
    # them (confirmed empirically: this exact gap first found for NWOT).
    model.train()
    acts: list[torch.Tensor] = []

    def record(t) -> None:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            t.retain_grad()
            acts.append(t)

    def module_hook(_module, _inp, out):
        record(out)

    handles = [
        m.register_forward_hook(module_hook)
        for m in model.modules()
        if isinstance(m, _ACTIVATION_MODULE_TYPES)
    ]

    orig_relu, orig_relu_ = F.relu, F.relu_

    def patched_relu(input, inplace=False):
        out = orig_relu(input, inplace=inplace)
        record(out)
        return out

    def patched_relu_(input):
        out = orig_relu_(input)
        record(out)
        return out

    F.relu, F.relu_ = patched_relu, patched_relu_
    try:
        x = torch.randn((batch_size, *in_shape[1:]), device=device)
        num_classes = out_shape[0] if out_shape else 10
        y = torch.randint(0, max(num_classes, 2), (batch_size,), device=device)

        model.zero_grad()
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = F.cross_entropy(out, y)
        loss.backward()

        if not acts:
            raise ValueError(
                "no activations captured for Fisher (no ReLU-family modules or F.relu calls found)"
            )

        score = 0.0
        for act in acts:
            if act.grad is not None:
                contrib = (act.grad * act).sum(dim=tuple(range(1, act.dim())))
                score += contrib.pow(2).sum().item()
        return score
    finally:
        F.relu, F.relu_ = orig_relu, orig_relu_
        for h in handles:
            h.remove()
        model.zero_grad()


# --- GraSP ----------------------------------------------------------------
# Wang et al. 2020 (gradient signal preservation). Uses a Hessian-vector
# product (via double backprop) to score how much pruning each connection
# would preserve gradient flow through the network -- a second-order
# variant of SNIP. More fragile than the other proxies (some architectures'
# forward passes break the double-backward graph), so failures here are
# expected and handled the same best-effort way as everything else.

def _grasp_score(
    model: nn.Module, in_shape: tuple[int, ...], out_shape: tuple[int, ...],
    device: torch.device, batch_size: int = 8,
) -> float:
    model.train()
    x = torch.randn((batch_size, *in_shape[1:]), device=device)
    num_classes = out_shape[0] if out_shape else 10
    y = torch.randint(0, max(num_classes, 2), (batch_size,), device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters for GraSP")

    model.zero_grad()
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    valid = [(p, g) for p, g in zip(params, grads) if g is not None]
    if not valid:
        raise ValueError("no gradients for GraSP")

    gnorm = sum((g * g).sum() for _, g in valid)
    hv = torch.autograd.grad(gnorm, [p for p, _ in valid], retain_graph=False, allow_unused=True)

    score = 0.0
    for (p, _), h in zip(valid, hv):
        if h is not None:
            score += (-p.detach() * h.detach()).sum().item()
    model.zero_grad()
    return score


# --- public entry point -----------------------------------------------------

def _compute_proxies_at_shape(
    nn_code: str,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    prm: Optional[dict],
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[str], bool]:
    """
    One attempt at one shape. Returns (result_fields, failure_messages,
    instantiate_ok). instantiate_ok=False means retrying a larger shape is
    pointless (the failure is in the code itself, not the input size).
    """
    result: dict[str, Any] = {
        "synflow": None, "nwot": None, "grad_norm": None,
        "log_params": None, "depth": None,
        "snip": None, "fisher": None, "grasp": None,
    }
    failures: list[str] = []

    try:
        model = _instantiate_model(nn_code, in_shape, out_shape, prm, device)
    except _UNRECOVERABLE_ERROR_TYPES as e:
        return result, [f"instantiate_failed (unrecoverable): {e}"], False
    except Exception as e:
        return result, [f"instantiate_failed: {e}"], True  # might be a shape issue; worth retrying larger

    try:
        result.update(_static_features(model))
    except Exception as e:
        failures.append(f"static_failed: {e}")

    try:
        result["synflow"] = _synflow_score(model, in_shape, device)
    except Exception as e:
        failures.append(f"synflow_failed: {e}")

    try:
        result["nwot"] = _nwot_score(model, in_shape, device, batch_size)
    except Exception as e:
        failures.append(f"nwot_failed: {e}")

    try:
        result["grad_norm"] = _grad_norm_score(model, in_shape, out_shape, device, batch_size)
    except Exception as e:
        failures.append(f"grad_norm_failed: {e}")

    try:
        result["snip"] = _snip_score(model, in_shape, out_shape, device, batch_size)
    except Exception as e:
        failures.append(f"snip_failed: {e}")

    try:
        result["fisher"] = _fisher_score(model, in_shape, out_shape, device, batch_size)
    except Exception as e:
        failures.append(f"fisher_failed: {e}")

    try:
        result["grasp"] = _grasp_score(model, in_shape, out_shape, device, batch_size)
    except Exception as e:
        failures.append(f"grasp_failed: {e}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, failures, True


def compute_proxies(
    nn_code: str,
    dataset: str,
    prm: Optional[dict] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    seed: int = 0,
) -> dict:
    """
    Best-effort computation of zero-cost proxies for one architecture.
    Never raises: on any failure, the corresponding field is None and
    proxy_status records what failed, so the record can still flow
    through the pipeline (see IMPROVEMENT_PLAN_Zero_Cost_Proxies.md).

    Tries increasing input shapes (see _shape_candidates) because the
    dataset's native size is not always the size an architecture was
    actually trained at (transform_code can resize) -- some deep nets
    fail at a small native shape purely because of that mismatch, not
    because anything is wrong with the architecture itself.
    """
    device = device or torch_device()
    torch.manual_seed(seed)

    last_result: dict[str, Any] = {
        "proxy_status": "ok", "synflow": None, "nwot": None,
        "grad_norm": None, "log_params": None, "depth": None,
        "snip": None, "fisher": None, "grasp": None,
    }
    all_attempts: list[str] = []

    for in_shape, out_shape in _shape_candidates(dataset):
        result, failures, worth_retrying = _compute_proxies_at_shape(
            nn_code, in_shape, out_shape, prm, device, batch_size
        )
        last_result = {**last_result, **result, "shape_used": in_shape}
        any_success = any(result[k] is not None for k in ("synflow", "nwot", "grad_norm", "snip", "fisher", "grasp"))

        if any_success:
            last_result["proxy_status"] = "ok" if not failures else "; ".join(failures)
            return last_result

        all_attempts.extend(f"[shape={in_shape}] {msg}" for msg in failures)
        if not worth_retrying:
            break

    last_result["proxy_status"] = "; ".join(all_attempts) if all_attempts else "unknown_failure"
    return last_result


# --- normalization (fit on train split only; see fit_proxy_normalization.py) ---
# Raw proxy scales are wildly different (synflow can be ~1e4 to ~1e12; nwot is
# a log-determinant, roughly -inf..a few hundred; grad_norm spans orders of
# magnitude like synflow). Feeding raw numbers like "933105726221.995" to an
# LLM as text is both unreadable and dominates the prompt numerically. Each
# proxy is log1p-transformed if it's a non-negative, wide-range quantity,
# then z-scored using statistics fit ONLY on the training-family split (never
# on val/test -- fitting on val/test would leak information about those
# architectures' relative scale into the model).

PROXY_NAMES = ("synflow", "nwot", "grad_norm", "log_params", "depth", "snip", "fisher", "grasp")
PROXY_TRANSFORMS = {
    "synflow": "log1p",
    "grad_norm": "log1p",
    "nwot": "none",
    "log_params": "none",  # already log-scaled by _static_features
    # depth is heavily right-skewed (median 24, p90 162, max 686 across the
    # LEMUR sample -- a handful of large MoE architectures with hundreds of
    # submodules dominate a raw z-score), so log-transform it too.
    "depth": "log1p",
    # snip/fisher: same |weight*grad| or sum-of-squares structure as
    # synflow/grad_norm, always non-negative, likely wide-range -> log1p.
    "snip": "log1p",
    "fisher": "log1p",
    # grasp is a SIGNED quantity (-theta * Hv, can be negative) -- log1p here
    # uses _apply_transform's signed-log branch (log1p(v) for v>=0, else
    # -log1p(-v)), which handles the sign safely rather than assuming
    # non-negativity like the other proxies. Revisit once real value ranges
    # are observed (see IMPROVEMENT_PLAN_Additional_Proxies.md).
    "grasp": "log1p",
}


def _apply_transform(name: str, value) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        # Defensive: a lone nan/inf (e.g. from stale cache entries computed
        # before the double-precision SynFlow fix) must never silently
        # poison the whole column's mean/std -- treat as missing instead.
        return None
    if PROXY_TRANSFORMS.get(name) == "log1p":
        return math.log1p(v) if v >= 0 else -math.log1p(-v)
    return v


def fit_normalization_stats(rows: list[dict], proxy_names: tuple[str, ...] = PROXY_NAMES) -> dict:
    """
    rows: proxy_cache records (dicts with proxy_names as keys, None allowed).
    Returns {proxy_name: {"transform", "mean", "std", "n"}}. Column-wise:
    each proxy uses only its own non-null values, independent of whether
    other proxies succeeded for that same architecture (see the "partial
    success" finding in PROGRESS_Zero_Cost_Proxies.md -- most non-'ok' rows
    still have usable values in other columns).
    """
    stats: dict[str, dict] = {}
    for name in proxy_names:
        vals = [_apply_transform(name, r.get(name)) for r in rows]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        if n < 2:
            stats[name] = {"transform": PROXY_TRANSFORMS.get(name, "none"), "mean": 0.0, "std": 1.0, "n": n}
            continue
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var ** 0.5
        stats[name] = {
            "transform": PROXY_TRANSFORMS.get(name, "none"),
            "mean": mean,
            "std": std if std > 1e-12 else 1.0,
            "n": n,
        }
    return stats


def normalize_proxy_value(name: str, raw_value, stats: dict) -> Optional[float]:
    """Apply the same transform + z-score used to fit `stats` to one raw value."""
    if raw_value is None:
        return None
    v = _apply_transform(name, raw_value)
    s = stats.get(name, {"mean": 0.0, "std": 1.0})
    return (v - s["mean"]) / (s["std"] or 1.0)


# Normalization stats the published ABrain/Accuracy-Prediction model was trained
# with. Single source of truth for anything that scales live-computed proxies
# (AccPredictor and acc_client), so inference matches training without needing a
# proxy_norm_stats.json file.
DEFAULT_PROXY_NORM_STATS: dict = {
    "synflow":    {"transform": "log1p", "mean": 28.010968242141725, "std": 28.76250416924666,  "n": 1377},
    "nwot":       {"transform": "none",  "mean": 91.97849879832741,  "std": 11.345025242202839, "n": 1373},
    "grad_norm":  {"transform": "log1p", "mean": 1.2624715355820684, "std": 1.5277296957245465, "n": 1359},
    "log_params": {"transform": "none",  "mean": 14.027992648327869, "std": 3.6699369017614827, "n": 1392},
    "depth":      {"transform": "log1p", "mean": 3.4690361790519724, "std": 0.8966121575625369, "n": 1392},
    "snip":       {"transform": "log1p", "mean": 2.6654958255013574, "std": 2.1397100177914203, "n": 1359},
    "fisher":     {"transform": "log1p", "mean": 0.5598579965866282, "std": 1.789205532349267,  "n": 1354},
    "grasp":      {"transform": "log1p", "mean": -2.65979787337622,  "std": 3.613374620607079,  "n": 1351},
}


# --- manual sanity check -----------------------------------------------------

def _sanity_check(n: int = 8) -> None:
    # parents[4] = repo root (file lives at ab/gpt/util/method/zero_cost_proxies.py)
    data_path = Path(__file__).resolve().parents[4] / "out" / "acc_predict" / "llm_finetuning_data.jsonl"
    if not data_path.exists():
        print(f"No data found at {data_path}")
        return

    device = torch_device()
    print(f"Using device: {device}\n")

    rows = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    header = (
        f"{'dataset':<14}{'synflow':>12}{'nwot':>10}{'grad_norm':>10}"
        f"{'snip':>12}{'fisher':>12}{'grasp':>12}{'depth':>7}  status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        prm = row.get("prm")
        if isinstance(prm, str):
            try:
                prm = json.loads(prm)
            except json.JSONDecodeError:
                prm = {}
        scores = compute_proxies(row.get("nn_code", ""), row.get("dataset", ""), prm, device=device)

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "  -  "

        print(
            f"{row.get('dataset',''):<14}"
            f"{fmt(scores['synflow']):>12}"
            f"{fmt(scores['nwot']):>10}"
            f"{fmt(scores['grad_norm']):>10}"
            f"{fmt(scores['snip']):>12}"
            f"{fmt(scores['fisher']):>12}"
            f"{fmt(scores['grasp']):>12}"
            f"{str(scores['depth'] or '-'):>7}  "
            f"{scores['proxy_status']}"
        )


if __name__ == "__main__":
    _sanity_check()

"""Edge-cost measurement + PSNR/cost objective for the denoise GA.

The thesis ("Towards Automated Image Denoising Model Generation for
Resource-Constrained Edge Systems") makes efficiency a first-class objective, so
GA fitness must reward quality PER unit of edge cost, not PSNR alone.

Two hardware-independent cost proxies, measured at the denoise operating
resolution (256x256), no external profiler dependency:
  * params  -- parameter count  (~ on-device memory footprint)
  * macs    -- multiply-accumulates (~ latency / energy); the edge-relevant one.
MACs are counted with forward hooks on the ops that dominate these U-Nets
(Conv2d / ConvTranspose2d / Linear); everything else (norms, activations,
element-wise) is negligible compute and ignored.

`effective_db` folds cost into PSNR as a MnasNet-style log-ratio penalty
    eff_dB = psnr_dB - weight * log(cost / budget)
so a model under budget gets a bonus, over budget a penalty, smoothly. With
mode="soft" it is a one-sided budget constraint (no penalty while within budget).
"""
import math

# torch + mutator.lemur_io (which pulls in the whole nn-dataset DB stack) are
# imported lazily inside measure_cost, so the pure-math objective below
# (effective_db) can be used/tested without loading torch or the database.


def _cleanup(model):
    from mutator.lemur_io import cleanup_temp_module
    info = getattr(model, "_temp_module_info", None)
    if info:
        try:
            cleanup_temp_module(*info)
        except Exception:
            pass


def _mac_hooks(model):
    """Register MAC-counting hooks; return (handles, macs_accumulator)."""
    import torch.nn as nn
    macs = {"v": 0}
    handles = []

    def conv(m, inp, out):
        kh, kw = m.kernel_size
        macs["v"] += out.numel() * (m.in_channels // m.groups) * kh * kw

    def convt(m, inp, out):
        kh, kw = m.kernel_size
        macs["v"] += out.numel() * (m.in_channels // m.groups) * kh * kw

    def linear(m, inp, out):
        macs["v"] += out.numel() * m.in_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(convt))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear))
    return handles, macs


def measure_cost(source):
    """Instantiate a LEMUR denoise Net from source; return its edge cost.

    Returns {"ok": bool, "params": int, "macs": int}. On any failure ok=False
    and the caller should fall back to no cost penalty (never kill a model that
    otherwise trained fine just because profiling hiccuped)."""
    import torch
    from mutator.lemur_io import load_lemur_model, get_dataset_params
    h, w = get_dataset_params("denoise")["spatial_dims"]
    model = None
    try:
        model = load_lemur_model(source)
        params = sum(p.numel() for p in model.parameters())
        dev = next(model.parameters()).device
        handles, macs = _mac_hooks(model)
        model.eval()
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, h, w, device=dev))
        finally:
            for h in handles:
                h.remove()
        return {"ok": True, "params": int(params), "macs": int(macs["v"])}
    except Exception:
        return {"ok": False, "params": 0, "macs": 0}
    finally:
        if model is not None:
            _cleanup(model)


def effective_db(psnr_db, cost, budget, weight, mode="log"):
    """Combine PSNR (dB) and edge cost into a single fitness score (dB units).

    cost/budget in the SAME units (macs or params). Returns psnr_db unchanged if
    the cost term is disabled or unusable, so callers degrade gracefully."""
    if not cost or not budget or budget <= 0 or weight <= 0:
        return psnr_db
    ratio = max(cost, 1.0) / budget
    if mode == "soft":                       # one-sided: free while within budget
        return psnr_db - weight * max(0.0, math.log(ratio))
    return psnr_db - weight * math.log(ratio)  # log: bonus under, penalty over

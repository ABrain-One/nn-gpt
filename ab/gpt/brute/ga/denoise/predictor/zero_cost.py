"""Training-free (zero-cost) proxies for the denoise accuracy predictor.

The learned regressor's second input block: cheap scalars that need ONE forward
(+one backward) on a random batch, no data, no optimisation. These are the
NAS-bench zero-cost proxies adapted to an image->image UNet. They give the
surrogate a signal about the model's *trainability* that the pure-static AST
features (features.py) can't see (e.g. two nets with identical op-counts but
different init-time gradient flow).

Proxies:
  * grad_norm  — L2 norm of all param grads after one L1(out, noisy) backward.
                 Low ~ vanishing gradients / dead net.
  * snip       — sum |g * w| (Lee et al. 2019 connection sensitivity).
  * grasp      — -sum (H g . w) approximated as sum(g*w) sign-flipped growth
                 (we use the cheap snip-family surrogate sum(g^2*|w|)).
  * synflow    — data-agnostic: all-ones input, product-of-abs-weights signal
                 = sum over params of |g_synflow * w| (Tanaka et al. 2020).
  * jacob_cov  — log-det of the input-jacobian covariance (Mellor et al. 2021);
                 higher ~ the net distinguishes inputs well at init.
  * act_std    — std of the output activations (dead/saturated detector).

Input is a small random batch (2x3x64x64) so every proxy is ~tens of ms on CPU
and the UNet's downsampling stays valid. Everything is guarded: a model that
won't instantiate/forward returns all-NaN (HistGBR consumes NaN natively), so a
bad candidate never crashes the funnel — it just scores as missing.

API: proxies(source, device="cpu") -> dict[str,float]  (NaN on failure)
     PROXY_NAMES  (stable ordered list)
"""
import math

PROXY_NAMES = ["zc_grad_norm", "zc_snip", "zc_grasp", "zc_synflow",
               "zc_jacob_cov", "zc_act_std"]

_NAN = {k: float("nan") for k in PROXY_NAMES}


def _build(source, device):
    ns = {}
    exec(compile(source, "<zc>", "exec"), ns)
    if "Net" not in ns:
        return None
    net = ns["Net"](in_shape=(1, 3, 64, 64), prm={"lr": 1e-4}, device=device)
    return net.to(device)


def proxies(source, device="cpu"):
    """Compute all zero-cost proxies for one architecture source. NaN-safe."""
    try:
        import torch
    except Exception:
        return dict(_NAN)
    # DETERMINISTIC: seed torch from the source hash so the same architecture
    # always yields identical proxies (reproducible surrogate features). Without
    # this the random probe batch makes every re-run of the predictor drift.
    import hashlib
    seed = int(hashlib.md5(source.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)

    try:
        net = _build(source, device)
        if net is None:
            return dict(_NAN)
    except Exception:
        return dict(_NAN)

    out = dict(_NAN)
    try:
        net.train()
        for p in net.parameters():
            p.requires_grad_(True)
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(2, 3, 64, 64, generator=gen, device=device,
                        requires_grad=True)
        y = net(x)
        if not torch.is_tensor(y):
            return out
        # ---- jacob_cov (needs grad wrt input, pre-loss) ----
        try:
            g = torch.autograd.grad(y.sum(), x, retain_graph=True,
                                    create_graph=False)[0]
            j = g.reshape(g.shape[0], -1)
            j = j - j.mean(0, keepdim=True)
            cov = j @ j.t()
            ev = torch.linalg.eigvalsh(cov.double())
            ev = ev[ev > 1e-8]
            out["zc_jacob_cov"] = float(ev.log().sum()) if ev.numel() else float("nan")
        except Exception:
            pass
        out["zc_act_std"] = float(y.detach().std())

        # ---- gradient-based proxies (one L1 backward) ----
        target = torch.randn_like(y)
        loss = torch.nn.functional.l1_loss(y, target)
        net.zero_grad(set_to_none=True)
        loss.backward()
        gn2, snip, grasp = 0.0, 0.0, 0.0
        for p in net.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            gn2 += float(g.pow(2).sum())
            snip += float((g * p.detach()).abs().sum())
            grasp += float((g.pow(2) * p.detach().abs()).sum())
        out["zc_grad_norm"] = math.sqrt(gn2)
        out["zc_snip"] = snip
        out["zc_grasp"] = grasp

        # ---- synflow (data-agnostic, all-ones, abs weights) ----
        try:
            net.zero_grad(set_to_none=True)
            xs = torch.ones(1, 3, 64, 64, device=device)
            ys = net(xs)
            net.zero_grad(set_to_none=True)
            ys.sum().backward()
            syn = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    syn += float((p.grad.detach() * p.detach()).abs().sum())
            out["zc_synflow"] = math.log(syn + 1.0)
        except Exception:
            pass
    except Exception:
        return out
    finally:
        try:
            net.zero_grad(set_to_none=True)
        except Exception:
            pass
    return out


def vector(source, device="cpu"):
    p = proxies(source, device)
    return [p[k] for k in PROXY_NAMES]


if __name__ == "__main__":
    import sys
    print(proxies(open(sys.argv[1]).read()))

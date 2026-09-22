"""Pre-evaluation behavioral-descriptor extraction for generated NN code.

MAP-Elites needs each candidate's behavior *before* it is trained, so seeds and
newly generated models can be placed on the same grid the corpus populates. The
two descriptors are:

  * ``params`` = ``sum(p.numel() for p in model.parameters())``
  * ``depth``  = ``len(list(model.modules()))``

These are the EXACT definitions used by ``ab.nn.util.NNAnalysis`` (``total_params``
/ ``total_layers``) that produce the ``nn_stat`` values the archive reads for the
corpus -- see ``analyze_model_comprehensive``. Replicating the formulas (rather
than importing) keeps this module light, but they must stay in sync with
NNAnalysis; both are one-liners over the instantiated model, so drift is unlikely.

Extraction instantiates the model on CPU (no training, no forward pass required
for the counts) inside a fully-guarded ``exec``. Generated code may be raw LLM
output with syntax errors or runtime faults, so *every* failure returns ``None``
and the caller simply skips archiving that candidate.
"""

from __future__ import annotations

from typing import Optional


class _DefaultingPrm(dict):
    """A prm dict that yields a benign default for any hyperparameter the model
    reads but we did not pre-fill, so instantiation does not KeyError."""

    def __missing__(self, key):
        k = str(key).lower()
        if "dropout" in k:
            return 0.5
        if "momentum" in k:
            return 0.9
        return 0.01


def _build_prm(namespace: dict) -> dict:
    """Construct a plausible prm from the model's ``supported_hyperparameters``."""
    prm = _DefaultingPrm()
    fn = namespace.get("supported_hyperparameters")
    names = set()
    if callable(fn):
        try:
            names = set(fn()) or set()
        except Exception:
            names = set()
    for name in names:
        prm[name] = prm[name]  # materialise via __missing__ defaults
    # common extras some models read directly
    for extra in ("lr", "momentum", "dropout", "weight_decay"):
        if extra not in prm:
            prm[extra] = prm[extra]
    return prm


def extract_descriptors(
    nn_code: str,
    in_shape: tuple = (1, 3, 32, 32),
    out_shape: tuple = (10,),
    device: str = "cpu",
    do_forward: bool = False,
) -> Optional[dict]:
    """Return ``{'params': int, 'depth': int}`` for ``nn_code``, or ``None``.

    Parameters
    ----------
    in_shape / out_shape:
        Constructor probe shapes. Defaults match cifar-10 (NCHW, 10 classes),
        the pipeline's dataset. ``in_shape[1]`` is read as the channel count by
        the LEMUR ``Net`` contract.
    do_forward:
        If True, run one CPU forward pass so any lazy modules materialise their
        parameters before counting. Off by default because standard modules
        expose their params at construction and a forward pass adds risk/cost.
    """
    if not isinstance(nn_code, str) or "class Net" not in nn_code:
        return None
    try:
        import torch  # deferred: importing this module never needs torch

        namespace: dict = {}
        exec(compile(nn_code, "<generated_nn>", "exec"), namespace)  # noqa: S102
        Net = namespace.get("Net")
        if not isinstance(Net, type):
            return None

        prm = _build_prm(namespace)
        dev = torch.device(device)
        model = Net(tuple(in_shape), tuple(out_shape), prm, dev)

        if do_forward:
            try:
                model.eval()
                with torch.no_grad():
                    model(torch.zeros(tuple(in_shape), device=dev))
            except Exception:
                pass  # counts below still valid for non-lazy modules

        params = int(sum(p.numel() for p in model.parameters()))
        depth = int(len(list(model.modules())))
        if params <= 0 or depth <= 0:
            return None
        return {"params": params, "depth": depth}
    except Exception:
        return None

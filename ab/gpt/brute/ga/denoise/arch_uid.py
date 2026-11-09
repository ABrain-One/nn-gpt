"""THE canonical architectural id. One construction, used by every producer and consumer.

WHY THIS MODULE EXISTS. There were two ways to get the WL graph hash and they silently
disagreed:

  arch_embed.exact(net)        hashes a net the caller already built
  arch_embed.exact_source(src) builds via mutator.lemur_io.load_lemur_model, then hashes

They instantiate the model with DIFFERENT arguments, so the same architecture gets different
hashes. arch_uid_audit.py populated the dataset's nn.arch_uid with the second; preflight
computed the first. The cross-check then reported "0 architectures already in the dataset"
for candidates whose architectures were provably already there -- not a clean result, a
comparison of incomparable numbers. Switching preflight to exact_source did not fix it
either: load_lemur_model is not importable inside preflight's rlimit-capped child, so every
call raised and was swallowed into None by the surrounding `except`.

An identifier is only useful if everyone computes it the same way. So: one function, one
instantiation signature, and it REPORTS why it failed instead of returning None quietly.

The signature below (1x3x256x256, the frozen denoise prm, CPU) is the one preflight already
uses to build and forward every candidate, so hashing costs no extra model construction there
and stays reproducible everywhere else.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def build(src):
    """Instantiate a candidate the canonical way. Raises on failure."""
    ns = {}
    exec(compile(src, "<arch>", "exec"), ns)
    net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
    net.eval()
    return net


def of_net(net):
    """WL graph hash of an already-built net."""
    from arch_embed import exact
    return exact(net)


def of_source(src):
    """-> (uid, error). uid is None only when the error string says why.

    Returning the reason is the point: a silent None is indistinguishable from "no
    duplicates", which is exactly how the broken cross-check looked correct.
    """
    try:
        return of_net(build(src)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:160]

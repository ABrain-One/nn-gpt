"""Accuracy predictor: fine-tune and evaluate a Qwen3-8B model that predicts a
training run's final best_accuracy and best_epoch from early-epoch signals.

This package is the refactored form of the former single-file
``ab.gpt.act.AccPredictor`` module. Its public surface is unchanged: every name
that lived at module top level (functions, constants, and the private helpers
external scripts import) is re-exported here, so ``from ab.gpt.act.acc_predictor
import <name>`` and attribute access ``acc_predictor.<name>`` behave exactly as
the monolith did. Experiment scripts that flip toggles must write them on
``acc_predictor.config`` (the single source of truth), e.g.
``config.USE_SOURCE_CODE = True``.
"""
from __future__ import annotations

# Preserve the monolith's friendly dependency error: importing this package (the
# back-compat entry point) pulls in the fine-tuning stack, exactly as the old
# top-level imports did. Individual pure-logic submodules (config/common/features/
# preprocess) can still be imported without these.
try:
    import unsloth  # noqa: F401
    import trl  # noqa: F401
    import transformers  # noqa: F401
    import datasets  # noqa: F401
    import torch  # noqa: F401
    import tqdm  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Missing required package. Install: pip install unsloth[colab-new] trl datasets transformers"
    ) from e

# Import order respects the dependency DAG (config <- common <- features <- ...).
from . import config
from . import common
from . import features
from . import preprocess
from . import model
from . import dataset
from . import train
from . import infer
from . import eval  # noqa: A004  (module name mirrors the original test stage)
from .__main__ import main

# Faithfully mirror the monolith's flat namespace: re-export every non-dunder
# name from every submodule (public API + the underscore-prefixed helpers that
# external scripts reach into) without clobbering earlier bindings.
_SUBMODULES = (config, common, features, preprocess, model, dataset, train, infer, eval)
for _m in _SUBMODULES:
    for _k, _v in vars(_m).items():
        if not _k.startswith("__"):
            globals().setdefault(_k, _v)
globals().setdefault("main", main)
del _m, _k, _v

"""Backward-compatible shim. The implementation now lives in the ``acc_predictor`` package.

This module was historically a single ~1800-line file. It has been split into the
``ab.gpt.act.acc_predictor`` package (config / common / features / preprocess /
dataset / model / train / infer / eval / __main__). This shim re-exports the entire
former top-level namespace -- public functions and constants plus the underscore
helpers that external scripts import (``_load_model``, ``_extended_metrics``,
``_load_proxy_cache``, ...) -- so existing imports keep working unchanged.

New code should import from ``ab.gpt.act.acc_predictor`` directly. Experiment scripts
that flip toggles must write them on the config module, e.g.::

    from ab.gpt.act.acc_predictor import config
    config.USE_SOURCE_CODE = True

(assigning ``AccPredictor.USE_SOURCE_CODE`` here would only rebind this shim's copy,
which the reader functions do not consult).
"""
from __future__ import annotations

from ab.gpt.act import acc_predictor as _pkg
from ab.gpt.act.acc_predictor import main  # noqa: F401

# Mirror the package's full flat namespace (public + underscore helpers).
globals().update({k: v for k, v in vars(_pkg).items() if not k.startswith("__")})


if __name__ == "__main__":
    main()

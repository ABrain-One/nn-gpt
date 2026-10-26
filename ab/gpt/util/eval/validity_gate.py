"""Fill every training slot with a model that can actually train.

What this buys is **sample size, not saved compute**. ``Eval.evaluate()``
already rejects structurally broken code before any training starts, so an
invalid generation costs almost no GPU time. It does, however, produce no
paired observation: at the observed 40-50% valid rate, generating exactly
``test_nn`` prompts yields only about half that many usable baselines, and the
standard error of the mean paired delta goes from 0.019 to 0.027 at n=30.

So the pipeline oversamples generation and this gate decides what gets trained.
Without a cap the oversample would be self-defeating -- every survivor would be
trained and the cycle would run two to three times over budget.

``ab/gpt/util/eval/static_validity.py::check_static_validity`` is upstream and
already knows how to answer "will this file run" by AST alone; nothing in the
generate -> evaluate path called it. This module calls it at the one place
where it decides something.

Rejected generations are deleted rather than archived, so the epoch directory
holds only the models the experiment actually measures. Their names and
reasons stay in ``validity_report.json``, which is what the
valid-generation-rate metric needs -- and that rate is itself a headline
result: a fine-tune that raises accuracy while breaking the interface more
often has not improved the model. Set NNGPT_VALIDITY_KEEP_REJECTED=1 to keep
the files for debugging.

Usage
-----
    from ab.gpt.util.eval.validity_gate import gate_directory

    kept, rejected = gate_directory(synth_dir, keep=40)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

REJECT_DIRNAME = "_rejected"
REPORT_NAME = "validity_report.json"

#: Set to 1 to keep rejected generations on disk under ``_rejected/`` instead
#: of deleting them. Off by default: with 2x oversampling, half of what is
#: generated is never trained, and leaving it on disk doubles the directories
#: written per cycle for models that are not part of the experiment. The
#: validity report keeps the counts and the reasons, which is what the
#: valid-generation-rate metric actually needs.
KEEP_REJECTED_ENV = "NNGPT_VALIDITY_KEEP_REJECTED"


def _model_files(models_dir: Path) -> List[Path]:
    """Candidate model directories, in generation order."""
    out = []
    for d in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if d.name.startswith("_"):
            continue
        nn = d / "new_nn.py"
        if not nn.exists():
            cands = sorted(d.glob("*.py"))
            if not cands:
                continue
            nn = cands[0]
        out.append(nn)
    return out


def _keep_rejected() -> bool:
    return os.environ.get(KEEP_REJECTED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _discard(model_dir: Path, reject_dir: Path, keep_rejected: bool) -> None:
    """Remove a generation that will not be trained, or park it for inspection."""
    try:
        if keep_rejected:
            reject_dir.mkdir(exist_ok=True)
            shutil.move(str(model_dir), str(reject_dir / model_dir.name))
        else:
            shutil.rmtree(model_dir, ignore_errors=True)
    except Exception as exc:
        print(f"[validity-gate] could not discard {model_dir.name}: {exc}", flush=True)


def gate_directory(models_dir: str, keep: Optional[int] = None,
                   move_rejected: bool = None) -> Tuple[List[str], List[dict]]:
    """Statically validate generated models; keep the first ``keep`` that pass.

    Returns ``(kept_dirs, rejected)``. Rejected entries carry the reason, so the
    failure modes of a cycle can be read off directly rather than inferred from
    a missing accuracy.

    Validation never raises: a checker that itself errors is treated as a pass,
    because the cost of wrongly discarding a good model is a lost sample, while
    the cost of wrongly keeping a bad one is only the training time the
    pipeline would have spent anyway.
    """
    from ab.gpt.util.eval.static_validity import check_static_validity

    base = Path(models_dir)
    if not base.is_dir():
        return [], []

    keep_rejected = _keep_rejected() if move_rejected is None else bool(move_rejected)
    reject_dir = base / REJECT_DIRNAME
    kept: List[str] = []
    rejected: List[dict] = []

    # Two passes on purpose. Validating only until the budget fills would make
    # the valid-generation rate depend on how many slots were left, and that
    # rate is a headline result -- a fine-tune that raises accuracy while
    # breaking the interface more often has not improved the model. So every
    # generation is checked, then the first `keep` valid ones are trained.
    checked = []
    for nn_file in _model_files(base):
        try:
            result = check_static_validity(str(nn_file))
            ok = bool(getattr(result, "valid", True))
            reason = getattr(result, "reason", None) or getattr(result, "error", None)
        except Exception as exc:                      # checker failure -> pass
            ok, reason = True, f"validity check errored: {exc}"
        checked.append((nn_file.parent, ok, reason))

    n_invalid = sum(1 for _, ok, _ in checked if not ok)
    for model_dir, ok, reason in checked:
        if not ok:
            rejected.append({"model": model_dir.name, "reason": str(reason),
                             "valid": False})
            _discard(model_dir, reject_dir, keep_rejected)
        elif keep is not None and len(kept) >= keep:
            rejected.append({"model": model_dir.name, "reason": "over budget",
                             "valid": True})
            _discard(model_dir, reject_dir, keep_rejected)
        else:
            kept.append(str(model_dir))

    n_checked = len(checked)
    n_valid = n_checked - n_invalid
    report = {
        "generated": n_checked,
        "statically_valid": n_valid,
        "statically_invalid": n_invalid,
        # Over every generation, not only those checked before the budget
        # filled, so the rate is comparable across cycles.
        "valid_rate": round(n_valid / n_checked, 4) if n_checked else None,
        "trained": len(kept),
        "rejected": rejected,
    }
    (base / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[validity-gate] {n_valid} of {n_checked} generations are statically "
          f"valid; training {len(kept)} and discarding {len(rejected)} "
          f"({n_invalid} invalid, {len(rejected) - n_invalid} over budget)"
          + ("" if not keep_rejected else f" into {REJECT_DIRNAME}/") + ".",
          flush=True)
    return kept, rejected

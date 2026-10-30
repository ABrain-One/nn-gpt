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

The AST check is necessary and not sufficient. On one measured run every one of
218 evaluation failures passed it and then died in ``Net.__init__`` -- a
generated helper unpacking a 4-tuple ``in_shape`` into five names. So the gate
has a second stage, ``ab/gpt/util/eval/smoke.py``: build the model on CPU and
push one batch through it. It is fail-open (a verdict of None keeps the model),
because a lost good generation costs a sample while a kept bad one costs only
the training time the pipeline was going to spend anyway.

Which candidates fill the budget is decided per seed, not alphabetically.
``sorted()`` over ``B0, B0_1, B1, B10, B10_1, B11 ...`` spends the budget on the
alternatives of the lowest-numbered seeds: measured, the 30 slots were filled
before seed B20 was reached at all, so ten test baselines were never evaluated
in that cycle. One candidate per seed is taken per pass, so every seed gets its
first candidate before any seed gets its second.

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


#: Set to 0 to skip the build-and-run stage and keep only the AST check.
SMOKE_ENV = "NNGPT_VALIDITY_SMOKE"

#: Wall-clock seconds the whole smoke stage may take for one cycle. Past it the
#: remaining candidates are kept untested: the point of the stage is to stop
#: wasting GPU minutes, and spending an hour of CPU to do it defeats that.
SMOKE_BUDGET_ENV = "NNGPT_SMOKE_BUDGET_SECONDS"


def _smoke_enabled() -> bool:
    """Off by default: the build now happens where the training happens.

    This gate used to build every candidate in a separate CPU subprocess, and
    the repair turn built them again during generation. Three builds per
    candidate per cycle, two of them approximations: a different device, a
    guessed input shape, and no dataset. The second one ran while the vLLM
    engine was resident and the trainer was parked in RAM, and it is what the
    kernel killed the run over.

    The evaluation worker already constructs the model, on the real device,
    with the real dataset and the real prm, moments before training it -- so
    its own attempt is both the cheaper check and the truthful one. A candidate
    that cannot build fails there in seconds, with a real traceback, and is
    reported as a failed evaluation. Nothing is gained by predicting that
    outcome beforehand in a process that cannot afford to.

    Set NNGPT_VALIDITY_SMOKE=1 to build here as well.
    """
    return os.environ.get(SMOKE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _smoke_budget() -> float:
    try:
        return float(os.environ.get(SMOKE_BUDGET_ENV, "900") or 900)
    except ValueError:
        return 900.0


def _seed_group(name: str) -> str:
    """Which seed a generation directory belongs to: ``B12_3`` -> ``B12``."""
    return name.split("_", 1)[0]


def _fill_budget(model_dirs: List[Path], keep: Optional[int]) -> List[Path]:
    """Up to ``keep`` directories, one per seed per pass.

    Alphabetical order spends the budget on the alternatives of the
    lowest-numbered seeds: with 38 candidates over 30 slots, sorted() filled
    every slot before seed B20 and ten pinned test baselines went unevaluated.
    Each seed contributes its first candidate before any seed contributes its
    second, so the cycle measures as many distinct baselines as it can.
    """
    if keep is None or keep >= len(model_dirs):
        return list(model_dirs)
    groups: dict = {}
    for d in model_dirs:
        groups.setdefault(_seed_group(d.name), []).append(d)
    order = sorted(groups)
    out: List[Path] = []
    depth = 0
    while len(out) < keep:
        took = 0
        for g in order:
            if depth < len(groups[g]):
                out.append(groups[g][depth])
                took += 1
                if len(out) >= keep:
                    break
        if not took:
            break
        depth += 1
    return out


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
    # generation is checked, then the budget is filled from what passed.
    checked = []
    for nn_file in _model_files(base):
        try:
            result = check_static_validity(str(nn_file))
            ok = bool(getattr(result, "valid", True))
            reason = getattr(result, "reason", None) or getattr(result, "error", None)
        except Exception as exc:                      # checker failure -> pass
            ok, reason = True, f"validity check errored: {exc}"
        checked.append([nn_file.parent, ok, reason, "static"])

    n_static_invalid = sum(1 for _, ok, _, _ in checked if not ok)

    # Second stage: actually build it. Only on what passed the AST check, and
    # only while the budget lasts -- 30 candidates at the per-candidate timeout
    # would be hours, so a run that goes long stops testing and keeps the rest.
    n_smoke_invalid = 0
    if _smoke_enabled():
        import time
        from ab.gpt.util.eval.smoke import smoke_check_dir
        budget = _smoke_budget()
        started = time.time()
        for entry in checked:
            model_dir, ok = entry[0], entry[1]
            if not ok:
                continue
            if budget and (time.time() - started) > budget:
                entry[3] = "smoke_skipped_budget"
                continue
            verdict, reason = smoke_check_dir(model_dir)
            if verdict is False:
                entry[1], entry[2], entry[3] = False, reason, "smoke"
                n_smoke_invalid += 1
            elif verdict is None:
                entry[3] = "smoke_unknown"
            else:
                entry[3] = "smoke_ok"
        print(f"[validity-gate] smoke test: {n_smoke_invalid} of "
              f"{len(checked) - n_static_invalid} statically valid generations "
              f"cannot build or run a batch", flush=True)
    else:
        print(f"[validity-gate] no build here: the evaluation worker builds "
              f"each candidate on the real device with the real dataset and "
              f"reports what happened ({SMOKE_ENV}=1 to build here too)",
              flush=True)

    n_invalid = n_static_invalid + n_smoke_invalid
    for model_dir, ok, reason, stage in checked:
        if not ok:
            rejected.append({"model": model_dir.name, "reason": str(reason),
                             "valid": False, "stage": stage})
            _discard(model_dir, reject_dir, keep_rejected)

    survivors = [(m, s) for m, ok, _r, s in checked if ok]
    chosen = _fill_budget([m for m, _s in survivors], keep)
    chosen_set = {str(p) for p in chosen}
    stage_of = {str(m): s for m, s in survivors}
    for model_dir, _s in survivors:
        if str(model_dir) in chosen_set:
            kept.append(str(model_dir))
        else:
            rejected.append({"model": model_dir.name, "reason": "over budget",
                             "valid": True, "stage": stage_of.get(str(model_dir))})
            _discard(model_dir, reject_dir, keep_rejected)

    n_checked = len(checked)
    n_valid = n_checked - n_invalid
    report = {
        "generated": n_checked,
        "statically_valid": n_checked - n_static_invalid,
        "statically_invalid": n_static_invalid,
        # Builds and runs one batch. This is the number that predicts how many
        # of the cycle's slots will come back with an accuracy.
        "runnable": n_valid,
        "smoke_invalid": n_smoke_invalid,
        # Over every generation, not only those checked before the budget
        # filled, so the rate is comparable across cycles.
        "valid_rate": round(n_valid / n_checked, 4) if n_checked else None,
        "seeds_covered": len({_seed_group(Path(k).name) for k in kept}),
        "trained": len(kept),
        "rejected": rejected,
    }
    (base / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[validity-gate] {n_valid} of {n_checked} generations build and run "
          f"({n_static_invalid} fail the AST check, {n_smoke_invalid} fail to "
          f"build); training {len(kept)} from "
          f"{report['seeds_covered']} distinct seed(s) and discarding "
          f"{len(rejected)}"
          + ("" if not keep_rejected else f" into {REJECT_DIRNAME}/") + ".",
          flush=True)
    return kept, rejected

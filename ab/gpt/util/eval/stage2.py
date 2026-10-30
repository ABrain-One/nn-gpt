"""Re-measure the best candidates of a cycle before believing their accuracy.

Every candidate is trained once, for one epoch, and the largest of those single
numbers is reported as the cycle's result. The pipeline's own documentation
states the consequence -- noise alone clears the 0.05 pair threshold about one
time in six -- and the run shows it directly: the same architecture, trained
twice from the same seed row, scored 0.9562 and 0.9621. A 0.6-point spread on a
quantity whose cycle-to-cycle movement is 0.9708 -> 0.9621 means the headline
number is partly a draw.

So evaluation becomes a ladder:

* stage 1 -- what the pipeline already does. Every candidate once, identical
  protocol, as a screen. Comparable across candidates, and cheap.
* stage 2 -- the top few, trained again, several times. The mean of those
  repeats is what the cycle reports as its best, and their spread is what says
  whether a gain over the previous cycle is real.

Spending the budget this way costs nothing extra in the limit: the models that
are not in the top few do not need a second measurement, because nothing is
decided by their exact value.

Stage 2 repeats at the SAME number of epochs by default. Raising
``NNGPT_STAGE2_EPOCHS`` gives a better estimate of what the architecture can
do, but writes LEMUR rows at an epoch the corpus builder does not query, so
those measurements inform the report and not the next cycle's training pairs.

Off unless ``NNGPT_STAGE2_TOPK`` is set.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Optional

CARRY = ("new_nn.py", "hp.txt", "tr.py", "dataframe.df")
STAGE2_DIRNAME = "synth_nn_stage2"
REPORT = "stage2_results.json"


def _topk_env(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def enabled() -> bool:
    return _topk_env("NNGPT_STAGE2_TOPK", 0) > 0


def _measured(models_dir: Path) -> List[Dict]:
    """(dir, accuracy) for every stage-1 candidate that produced a number."""
    out = []
    for d in sorted(models_dir.glob("B*")):
        info = d / "eval_info.json"
        if not info.is_file():
            continue
        try:
            payload = json.loads(info.read_text(encoding="utf-8"))
        except Exception:
            continue
        acc = payload.get("accuracy")
        if acc is None:
            res = payload.get("eval_results")
            if isinstance(res, dict):
                acc = res.get("accuracy", res.get("acc"))
        try:
            acc = float(acc)
        except (TypeError, ValueError):
            continue
        out.append({"dir": d, "accuracy": acc})
    out.sort(key=lambda r: -r["accuracy"])
    return out


def _stage(models_dir: Path, picks: List[Dict], repeats: int) -> Optional[Path]:
    stage_dir = models_dir.parent / STAGE2_DIRNAME
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for pick in picks:
        src = pick["dir"]
        for r in range(repeats):
            # No eval_info.json in the copy: Eval skips a directory that
            # already holds one, so a repeat that carried it would be reported
            # as already done and the stage would measure nothing.
            dst = stage_dir / f"{src.name}r{r}"
            dst.mkdir(parents=True, exist_ok=True)
            for name in CARRY:
                f = src / name
                if f.is_file():
                    shutil.copyfile(f, dst / name)
            if not (dst / "new_nn.py").is_file():
                shutil.rmtree(dst, ignore_errors=True)
                continue
            made += 1
    return stage_dir if made else None


def run(models_dir, epoch, nn_name_prefix=None, nn_train_epochs=1) -> Optional[Dict]:
    """Re-measure the top candidates. Returns the stage-2 summary, or None."""
    if not enabled():
        return None
    models_dir = Path(models_dir)
    topk = _topk_env("NNGPT_STAGE2_TOPK", 0)
    repeats = max(2, _topk_env("NNGPT_STAGE2_REPEATS", 3))
    epochs = _topk_env("NNGPT_STAGE2_EPOCHS", int(nn_train_epochs or 1))

    measured = _measured(models_dir)
    if not measured:
        print("[stage2] nothing was measured in stage 1; skipped", flush=True)
        return None
    picks = measured[:topk]
    stage_dir = _stage(models_dir, picks, repeats)
    if stage_dir is None:
        print("[stage2] could not stage any candidate; skipped", flush=True)
        return None

    print(f"[stage2] re-measuring the top {len(picks)} of {len(measured)} "
          f"candidates {repeats}x at {epochs} epoch(s): "
          + ", ".join(f"{p['dir'].name}={p['accuracy']:.4f}" for p in picks),
          flush=True)

    # The per-epoch report is written from whichever directory was evaluated
    # last, so letting it run here would overwrite the cycle's record with one
    # computed over a handful of re-measured copies.
    previous = os.environ.get("NNGPT_SUPPRESS_EPOCH_REPORT")
    os.environ["NNGPT_SUPPRESS_EPOCH_REPORT"] = "1"
    # Screening runs are kept out of the database (NNGPT_EVAL_SAVE_TO_DB=0 in
    # the run script) precisely so this pass can be the only thing that writes
    # to it: a model enters the corpus because it was measured several times,
    # not because it was lucky once.
    previous_save = os.environ.get("NNGPT_EVAL_SAVE_TO_DB")
    os.environ["NNGPT_EVAL_SAVE_TO_DB"] = "1"
    try:
        from ab.gpt.act.eval import Eval as NNEval
        NNEval.main(nn_name_prefix=nn_name_prefix, nn_train_epochs=epochs,
                    only_epoch=epoch, custom_synth_dir=str(stage_dir))
    except Exception as exc:
        print(f"[stage2] evaluation failed: {exc}", flush=True)
        return None
    finally:
        if previous is None:
            os.environ.pop("NNGPT_SUPPRESS_EPOCH_REPORT", None)
        else:
            os.environ["NNGPT_SUPPRESS_EPOCH_REPORT"] = previous
        if previous_save is None:
            os.environ.pop("NNGPT_EVAL_SAVE_TO_DB", None)
        else:
            os.environ["NNGPT_EVAL_SAVE_TO_DB"] = previous_save

    by_candidate: Dict[str, List[float]] = {}
    for row in _measured(stage_dir):
        base = row["dir"].name.rsplit("r", 1)[0]
        by_candidate.setdefault(base, []).append(row["accuracy"])

    summary = {"epoch": epoch, "epochs_per_run": epochs, "repeats": repeats,
               "candidates": [], "stage1_max": picks[0]["accuracy"]}
    for pick in picks:
        name = pick["dir"].name
        runs = sorted(by_candidate.get(name, []))
        if not runs:
            continue
        summary["candidates"].append({
            "model": name,
            "stage1_accuracy": round(pick["accuracy"], 4),
            "runs": [round(a, 4) for a in runs],
            "mean": round(statistics.fmean(runs), 4),
            "max": round(max(runs), 4),
            "std": round(statistics.stdev(runs), 4) if len(runs) > 1 else 0.0,
        })
    if not summary["candidates"]:
        print("[stage2] no repeat produced a number", flush=True)
        return None

    best = max(summary["candidates"], key=lambda c: c["mean"])
    summary["best_model"] = best["model"]
    # The reportable result of the cycle: a mean over repeats, not the largest
    # single draw out of 38 attempts.
    summary["best_mean"] = best["mean"]
    summary["best_max"] = best["max"]
    noise = [c["std"] for c in summary["candidates"] if c["std"]]
    summary["repeat_std"] = round(statistics.fmean(noise), 4) if noise else 0.0
    (models_dir / REPORT).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[stage2] best {best['model']}: mean {best['mean']:.4f} over "
          f"{len(best['runs'])} runs (single-run stage 1 said "
          f"{best['stage1_accuracy']:.4f}); typical repeat spread "
          f"±{summary['repeat_std']:.4f}. A cycle-to-cycle gain smaller than "
          f"that is not evidence.", flush=True)
    return summary

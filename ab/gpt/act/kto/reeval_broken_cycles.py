#!/usr/bin/env python3
"""Re-evaluate broken cycles of a KTO/DivPO run — EVALUATION ONLY.

Some cycles finish generation but skip evaluation (the cycle crashed at/after eval,
or eval returned empty), leaving cycle_<n>/nneval/gen_*/ with new_nn.py but no
eval_info.json, no usable metrics.json, and zeros/gaps in the plots. This script
finds those cycles and runs ONLY evaluation on them, so plot_kto_cycles.py renders
them correctly:

  * NNEval (re)produces eval_info.json / eval_summary.json per generation — the exact
    files the plotter reads for per-model accuracy.
  * a plot-compatible metrics.json (the "bucketing" block) is reconstructed from the
    eval results + the on-disk generation state.

It NEVER touches checkpoints, training, the preference caches, or generation_records
(read-only), so it cannot change a run's resume state or the model chain. Evaluation
uses the SAME card/ab.nn protocol as the pipeline (defaults: norm_256_flip, lr 0.01,
momentum 0.9, batch 10, dropout 0.2, 1 epoch) — override to match a run that differed.

A cycle is "broken" (re-evaluated) iff it has evaluable generations (new_nn.py) and
NONE of them carry an eval_info.json. Already-evaluated cycles are skipped (use
--force to redo them). Idempotent: re-running only re-touches still-broken cycles.

  # in-container, GPU:
  python -m ab.gpt.act.kto.reeval_broken_cycles --run_dir out_<TAG>/nngpt
  python -m ab.gpt.act.kto.reeval_broken_cycles --run_dir out_<TAG>/nngpt --dry_run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_cycle_num(p: Path) -> int:
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _acc_from_eval_info(payload: Dict[str, Any]) -> Optional[float]:
    """Accuracy from a per-model eval_info.json (mirrors plot_kto_cycles.py)."""
    res = payload.get("eval_results")
    a = None
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        a = res[1]
    elif isinstance(res, dict):
        a = res.get("accuracy", res.get("acc"))
        if a is None:
            eps = res.get("epochs", [])
            if eps and isinstance(eps[0], dict):
                a = eps[0].get("accuracy", eps[0].get("acc"))
    try:
        return float(a) if a is not None else None
    except (TypeError, ValueError):
        return None


def _has_pipeline_cycles(d: Path) -> bool:
    """True if d holds real pipeline cycles (cycle_*/nneval), not stray cycle_* dirs
    (NNEval leaves its own cycle_* bookkeeping directly under nngpt/ without nneval/)."""
    return d.is_dir() and any((c / "nneval").is_dir() for c in d.glob("cycle_*"))


def find_subdirs(run_dir: Path) -> List[Path]:
    """Every output_subdir holding real cycle_*/nneval/ — the run dir itself AND its
    immediate children, so nngt/<subdir>/cycle_* is found even when stray cycle_* dirs
    sit directly under nngpt/. Filtered by nneval/ so non-pipeline cycle dirs are ignored."""
    out: List[Path] = []
    if _has_pipeline_cycles(run_dir):
        out.append(run_dir)
    if run_dir.is_dir():
        for d in sorted(run_dir.iterdir()):
            if d != run_dir and _has_pipeline_cycles(d):
                out.append(d)
    return out


def scan_cycle(cycle_dir: Path) -> Dict[str, Any]:
    """On-disk state of one cycle (ground truth, no reliance on records)."""
    nneval = cycle_dir / "nneval"
    evaluable: List[Path] = []     # gen dirs with new_nn.py (novel, to evaluate)
    nonnovel = 0                   # gen dirs with new_nn.notnovel.py (Jaccard/graph dup)
    unparseable = 0                # gen dirs with neither (generation didn't parse)
    already_eval = 0               # gen dirs already carrying eval_info.json
    not_attempted = 0              # new_nn.py never evaluated (no eval_info AND no error.txt)
    if nneval.is_dir():
        for md in sorted(nneval.iterdir(), key=lambda d: d.name):
            if not md.is_dir():
                continue
            nn = md / "new_nn.py"
            aside = md / "new_nn.notnovel.py"
            if nn.exists():
                evaluable.append(md)
                if (md / "eval_info.json").exists():
                    already_eval += 1
                elif not (md / "error.txt").exists():
                    not_attempted += 1     # eval never reached this one (interrupted)
            elif aside.exists():
                nonnovel += 1
            else:
                unparseable += 1
    return {
        "cycle": _parse_cycle_num(cycle_dir),
        "cycle_dir": cycle_dir,
        "nneval": nneval,
        "evaluable": evaluable,
        "n_evaluable": len(evaluable),
        "nonnovel": nonnovel,
        "unparseable": unparseable,
        "already_eval": already_eval,
        "not_attempted": not_attempted,
        "has_metrics": (cycle_dir / "metrics.json").exists(),
        # Broken = has models whose evaluation was never attempted (fully-skipped OR
        # interrupted part-way), so the cycle's metrics are incomplete. A cycle where
        # every model was attempted (eval_info or error.txt) is complete, even if some
        # failed — not re-evaluated.
        "broken": len(evaluable) > 0 and not_attempted > 0,
    }


def _generated_total(cycle_dir: Path, st: Dict[str, Any]) -> int:
    recs = cycle_dir / "generation_records.jsonl"
    if recs.exists():
        try:
            return sum(1 for line in recs.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:  # noqa: BLE001
            pass
    return st["n_evaluable"] + st["nonnovel"] + st["unparseable"]


def _desirable_total_estimate(subdir: Path, cycle: int) -> int:
    """Best-effort accumulated-data count for the CSV/printout (read-only)."""
    for fname in ("divpo_pairs_cache.jsonl", "kto_desirable_cache.jsonl"):
        f = subdir / fname
        if not f.exists():
            continue
        n = 0
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c = json.loads(line).get("_meta", {}).get("cycle")
                if c is None or int(c) <= cycle:
                    n += 1
        except Exception:  # noqa: BLE001
            return 0
        return n
    return 0


def run_eval_for_cycle(st: Dict[str, Any], *, dataset: str, transform: str, lr: float,
                       momentum: float, batch: int, dropout: float, train_epochs: int,
                       multi_gpu: bool) -> None:
    """Drive NNEval on this cycle's nneval/ exactly as the pipeline does."""
    from ab.gpt import NNEval
    NNEval.main(
        nn_name_prefix=None,
        nn_train_epochs=train_epochs,
        only_epoch=0,
        save_to_db=False,
        nn_alter_epochs=1,
        task="img-classification",
        dataset=dataset,
        metric="acc",
        lr=lr,
        batch=batch,
        dropout=dropout,
        momentum=momentum,
        transform=transform,
        custom_synth_dir=str(st["nneval"]),
        cycle=st["cycle"],
        use_sequential=not multi_gpu,
        use_all_visible_gpus=multi_gpu,
    )


def reconstruct_metrics(st: Dict[str, Any], subdir: Path, threshold: float) -> Dict[str, Any]:
    """Rebuild the plot-compatible metrics.json from the (now written) eval outputs.

    Mirrors the DivPO/KTO bucketing: novel-and-good → desirable; non-novel + low-acc +
    failed + unparseable → undesirable. Accuracies are read from the novel gens' freshly
    written eval_info.json (non-novel were never evaluated, exactly as in a live cycle).
    """
    cycle = st["cycle"]
    accuracies: List[float] = []
    n_desirable = n_low = n_failed = 0
    for md in st["evaluable"]:
        info = md / "eval_info.json"
        acc = None
        if info.exists():
            try:
                acc = _acc_from_eval_info(json.loads(info.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                acc = None
        if acc is None:
            n_failed += 1
            continue
        accuracies.append(acc)
        if acc >= threshold:
            n_desirable += 1
        else:
            n_low += 1

    n_nonnovel = st["nonnovel"]
    n_unparseable = st["unparseable"]
    n_undesirable = n_low + n_failed + n_unparseable + n_nonnovel
    best = max(accuracies) if accuracies else 0.0
    passed = [a for a in accuracies if a >= threshold]
    avg = (sum(passed) / len(passed)) if passed else 0.0
    avg_all = (sum(accuracies) / len(accuracies)) if accuracies else 0.0

    bucketing = {
        "new_desirable": n_desirable,
        "new_undesirable": n_undesirable,
        "not_novel_skipped": n_nonnovel,
        "undesirable_breakdown": {
            "non_novel": n_nonnovel, "low_accuracy": n_low,
            "failed": n_failed, "unparseable": n_unparseable,
        },
        "evaluated_accuracies": len(accuracies),
        "best_accuracy": best,
        "avg_accuracy": avg,
        "avg_accuracy_all": avg_all,
        "effective_threshold": threshold,
        "desirable_total": _desirable_total_estimate(subdir, cycle),
        "undesirable_total": 0,
        "reeval_reconstructed": True,
    }
    return {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "generated": _generated_total(st["cycle_dir"], st),
        "extractable": st["n_evaluable"],
        "bucketing": bucketing,
        "dataset": {"reeval": True},
        # Training genuinely did not run for a broken cycle — report it truthfully.
        "training": {"success": False, "skipped": True, "reason": "reeval_eval_only"},
        "reeval_reconstructed": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="out_<TAG>/nngpt (scans every subdir) or a single output_subdir")
    p.add_argument("--accuracy_threshold", type=float, default=0.40)
    p.add_argument("--dataset", type=str, default="cifar-10")
    p.add_argument("--eval_transform", type=str, default="norm_256_flip")
    p.add_argument("--eval_lr", type=float, default=0.01)
    p.add_argument("--eval_momentum", type=float, default=0.9)
    p.add_argument("--eval_batch", type=int, default=10)
    p.add_argument("--eval_dropout", type=float, default=0.2)
    p.add_argument("--eval_train_epochs", type=int, default=1)
    p.add_argument("--multi_gpu", action="store_true",
                   help="Distribute eval across all visible GPUs (NNEval worker pool)")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even cycles that already have eval_info.json")
    p.add_argument("--cycles", type=str, default="",
                   help="Comma-separated cycle numbers to target (default: auto-detect all broken)")
    p.add_argument("--dry_run", action="store_true",
                   help="List broken cycles and exit without evaluating")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")
    subdirs = find_subdirs(run_dir)
    if not subdirs:
        raise SystemExit(f"No output_subdir with cycle_*/ under {run_dir}")
    forced = {int(c) for c in args.cycles.split(",") if c.strip().isdigit()}

    targets: List[Tuple[Path, Dict[str, Any]]] = []
    print("=" * 72)
    for sub in subdirs:
        cyc_dirs = sorted(sub.glob("cycle_*"), key=_parse_cycle_num)
        print(f"[{sub.name}] {len(cyc_dirs)} cycles")
        for cd in cyc_dirs:
            st = scan_cycle(cd)
            if st["cycle"] < 0:
                continue
            auto = st["broken"] or (args.force and st["n_evaluable"] > 0)
            broken = (st["cycle"] in forced and auto) if forced else auto
            flag = ("BROKEN" if st["broken"] else
                    "force" if broken else
                    "ok" if st["already_eval"] else
                    "empty")
            print(f"  cycle {st['cycle']:>3}: evaluable={st['n_evaluable']:>3} "
                  f"nonnovel={st['nonnovel']:>3} unparseable={st['unparseable']:>3} "
                  f"already_eval={st['already_eval']:>3} not_attempted={st['not_attempted']:>3} "
                  f"metrics={'y' if st['has_metrics'] else 'n'}  -> {flag}")
            if broken:
                targets.append((sub, st))
    print("-" * 72)
    print(f"Broken cycles to re-evaluate: {len(targets)}"
          + (f"  ({', '.join(str(s['cycle']) for _, s in targets)})" if targets else ""))
    print("=" * 72)

    if args.dry_run or not targets:
        if args.dry_run:
            print("dry-run: nothing evaluated.")
        return

    for i, (sub, st) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] re-evaluating {sub.name} cycle {st['cycle']} "
              f"({st['n_evaluable']} models)...", flush=True)
        try:
            run_eval_for_cycle(
                st, dataset=args.dataset, transform=args.eval_transform, lr=args.eval_lr,
                momentum=args.eval_momentum, batch=args.eval_batch, dropout=args.eval_dropout,
                train_epochs=args.eval_train_epochs, multi_gpu=args.multi_gpu)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] NNEval failed for cycle {st['cycle']}: {type(e).__name__}: {e}")
            continue
        # Re-scan (eval_info.json now written) and rebuild metrics.json.
        st2 = scan_cycle(st["cycle_dir"])
        st2["cycle_dir"] = st["cycle_dir"]
        metrics = reconstruct_metrics(st2, sub, args.accuracy_threshold)
        out = st["cycle_dir"] / "metrics.json"
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        b = metrics["bucketing"]
        print(f"  wrote {out}  | desirable={b['new_desirable']} undesirable={b['new_undesirable']} "
              f"not_novel={b['not_novel_skipped']} best={b['best_accuracy']*100:.2f}% "
              f"avg(>=thr)={b['avg_accuracy']*100:.2f}% evaluated={b['evaluated_accuracies']}")

    print("\nDone. Re-plot with slurm_nngpt_kto_plot.sh to pick up the restored cycles.")


if __name__ == "__main__":
    main()

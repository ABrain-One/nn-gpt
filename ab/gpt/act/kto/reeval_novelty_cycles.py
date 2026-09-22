#!/usr/bin/env python3
"""Recompute novelty + re-evaluate cycles whose novelty filter mis-flagged (almost)
every generation as non-novel — e.g. a resume poisoned the similarity index, leaving
new_desirable=0 / evaluated=0 (metrics.json even records training as successful)
despite healthy neighbouring cycles.

For each affected cycle this rebuilds the CORRECT novelty reference from disk — the
LEMUR DB (ab.nn) + every extractable generation from EARLIER cycles of the SAME run
(not the poisoned live index, not the pair cache) — recomputes each generation's
nearest MinHash-Jaccard similarity, restores the ones that are genuinely novel
(new_nn.notnovel.py -> new_nn.py), evaluates them, and rewrites a corrected
metrics.json. Non-novel calls that the recompute confirms are left as-is; a cycle is
only "fixed" when the clean recompute actually disagrees with the recorded labels.

This is a SEPARATE tool from reeval_broken_cycles (whose helpers it reuses, unchanged
— that one handles cycles with no eval at all; this one handles wrongly-non-novel
cycles). It writes only nneval/ novelty labels + eval_info/eval_summary/metrics —
never checkpoints, the pair caches, or generation_records.

  # dry-run reports recorded-vs-recomputed novelty (no eval); needs the container (DB):
  python -m ab.gpt.act.kto.reeval_novelty_cycles --run_dir out_<TAG>/nngpt/divpo_core --dry_run
  # fix + evaluate the mis-flagged cycles:
  python -m ab.gpt.act.kto.reeval_novelty_cycles --run_dir out_<TAG>/nngpt/divpo_core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab.gpt.act.kto.reeval_broken_cycles import (
    find_subdirs,
    _parse_cycle_num,
    scan_cycle,
    run_eval_for_cycle,
    reconstruct_metrics,
)
from ab.gpt.act.kto.graph_novelty_audit import graph_hash_for_file


def _read_gen_code(gen_dir: Path) -> Optional[str]:
    for name in ("new_nn.py", "new_nn.notnovel.py"):
        f = gen_dir / name
        if f.exists():
            try:
                return f.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return None
    return None


def _gen_dirs(cycle_dir: Path) -> List[Path]:
    nneval = cycle_dir / "nneval"
    if not nneval.is_dir():
        return []
    return [d for d in sorted(nneval.iterdir(), key=lambda p: p.name) if d.is_dir()]


def _gen_code_file(gen_dir: Path) -> Optional[Path]:
    for name in ("new_nn.py", "new_nn.notnovel.py"):
        f = gen_dir / name
        if f.exists():
            return f
    return None


def _graph_hash_of(gen_dir: Path, gate: Dict[str, Any]) -> Optional[str]:
    """Canonical graph hash of a generation (None if untraceable → fail open)."""
    cf = _gen_code_file(gen_dir)
    if cf is None:
        return None
    try:
        h, _ = graph_hash_for_file(cf, gate["sizes"], gate["in_ch"], gate["n_cls"],
                                   "aten", gate["granularity"], gate["wl_rounds"])
        return h
    except Exception:  # noqa: BLE001
        return None


def add_cycle_to_index(idx, cycle_dir: Path, graph_seen=None, gate=None) -> int:
    """Add every extractable generation of a cycle to the reference (mirrors the live
    pipeline, which adds each generation regardless of its novelty label). With a graph
    gate active, also register each generation's canonical graph hash — the graph
    reference is generations only (no DB), exactly as the live gate builds it."""
    n = 0
    for gd in _gen_dirs(cycle_dir):
        code = _read_gen_code(gd)
        if code and code.strip():
            idx.add(code)
            n += 1
        if graph_seen is not None and gate is not None:
            h = _graph_hash_of(gd, gate)
            if h is not None:
                graph_seen.add(h)
    return n


def recompute_cycle_novelty(idx, cycle_dir: Path, threshold: float,
                            graph_seen=None, gate=None
                            ) -> List[Tuple[Path, bool, float]]:
    """Recompute novelty per generation, in gen order, accumulating into the reference
    as the live pre-filter does. Novel = Jaccard < threshold AND (when a graph gate is
    active) graph-distinct vs earlier generations. Untraceable archs fail open (Jaccard
    decides), matching the pipeline. Returns (gen_dir, is_novel, jaccard) per gen."""
    out: List[Tuple[Path, bool, float]] = []
    for gd in _gen_dirs(cycle_dir):
        code = _read_gen_code(gd)
        if not code or not code.strip():
            continue  # unparseable — not part of the reference or the novelty set
        j = idx.nearest_jaccard(code)
        idx.add(code)
        is_novel = j < threshold
        if graph_seen is not None and gate is not None:
            h = _graph_hash_of(gd, gate)
            if h is not None:
                if h in graph_seen:
                    is_novel = False  # graph-duplicate → non-novel even if Jaccard-novel
                graph_seen.add(h)
        out.append((gd, is_novel, j))
    return out


def restore_labels(results: List[Tuple[Path, bool, float]]) -> Tuple[int, int]:
    """Make the on-disk novelty labels match the recompute: novel -> new_nn.py (so
    NNEval evaluates it), non-novel -> new_nn.notnovel.py (so NNEval skips it)."""
    restored = demoted = 0
    for gd, is_novel, _ in results:
        nn = gd / "new_nn.py"
        aside = gd / "new_nn.notnovel.py"
        if is_novel:
            if aside.exists() and not nn.exists():
                try:
                    aside.rename(nn)
                    restored += 1
                except Exception:  # noqa: BLE001
                    pass
        else:
            if nn.exists():
                try:
                    nn.rename(aside)
                    demoted += 1
                except Exception:  # noqa: BLE001
                    pass
    return restored, demoted


def _recorded_stats(cycle_dir: Path) -> Dict[str, Any]:
    m = cycle_dir / "metrics.json"
    if not m.exists():
        return {}
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("bucketing", {})
    except Exception:  # noqa: BLE001
        return {}


def is_candidate(cycle_dir: Path, suspect_frac: float) -> Tuple[bool, Dict[str, Any]]:
    """A cycle is suspect if (almost) every extractable generation is labelled non-novel
    and none was evaluated — the signature of a poisoned novelty index."""
    st = scan_cycle(cycle_dir)
    extractable = st["n_evaluable"] + st["nonnovel"]
    st["extractable"] = extractable
    if extractable == 0:
        return False, st
    suspect = (st["already_eval"] == 0 and st["nonnovel"] >= suspect_frac * extractable)
    return suspect, st


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="out_<TAG>/nngpt/<subdir> (e.g. .../divpo_core) or the nngpt dir")
    p.add_argument("--cycles", type=str, default="",
                   help="Comma-separated cycle numbers to force (default: auto-detect suspects)")
    p.add_argument("--suspect_frac", type=float, default=0.9,
                   help="Auto-detect: fraction of extractable gens labelled non-novel to be suspect")
    p.add_argument("--sim_threshold", type=float, default=0.85)
    p.add_argument("--sim_db_task", type=str, default="img-classification")
    p.add_argument("--sim_db_dataset", type=str, default="cifar-10")
    p.add_argument("--accuracy_threshold", type=float, default=0.40)
    p.add_argument("--dataset", type=str, default="cifar-10")
    p.add_argument("--eval_transform", type=str, default="norm_256_flip")
    p.add_argument("--eval_lr", type=float, default=0.01)
    p.add_argument("--eval_momentum", type=float, default=0.9)
    p.add_argument("--eval_batch", type=int, default=10)
    p.add_argument("--eval_dropout", type=float, default=0.2)
    p.add_argument("--eval_train_epochs", type=int, default=1)
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Recompute + report recorded-vs-recomputed novelty; evaluate nothing")
    # Graph gate — pass for runs trained with --graph_gate so the recompute matches
    # their novelty (Jaccard-novel AND graph-distinct vs earlier generations).
    p.add_argument("--graph_gate", action="store_true")
    p.add_argument("--graph_gate_granularity", choices=["topology", "typed"], default="topology")
    p.add_argument("--graph_gate_sizes", type=str, default="32,64,96,224,256")
    p.add_argument("--graph_gate_wl_rounds", type=int, default=3)
    p.add_argument("--graph_gate_num_classes", type=int, default=10)
    p.add_argument("--graph_gate_in_channels", type=int, default=3)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    subdirs = find_subdirs(run_dir)
    if not subdirs:
        raise SystemExit(f"No output_subdir with cycle_*/nneval under {run_dir}")
    forced = {int(c) for c in args.cycles.split(",") if c.strip().isdigit()}

    gate = None
    if args.graph_gate:
        gate = {
            "sizes": [int(s) for s in args.graph_gate_sizes.split(",") if s.strip()],
            "in_ch": args.graph_gate_in_channels,
            "n_cls": args.graph_gate_num_classes,
            "granularity": args.graph_gate_granularity,
            "wl_rounds": args.graph_gate_wl_rounds,
        }

    from ab.gpt.act.kto.similarity_penalty import SimilarityIndex

    fixed = 0
    for sub in subdirs:
        cyc_dirs = sorted(sub.glob("cycle_*"), key=_parse_cycle_num)
        print("=" * 74)
        print(f"[{sub.name}] {len(cyc_dirs)} cycles — building novelty reference (DB + prior gens)")
        idx = SimilarityIndex(threshold=args.sim_threshold)
        n_db = idx.add_db(args.sim_db_task, args.sim_db_dataset)
        graph_seen = set() if gate else None
        print(f"  reference seeded with {n_db} LEMUR DB architectures "
              f"(threshold {args.sim_threshold})"
              + ("; graph gate ON — also require graph-distinct vs earlier gens" if gate else ""))

        for cd in cyc_dirs:
            cnum = _parse_cycle_num(cd)
            if cnum < 0:
                continue
            suspect, st = is_candidate(cd, args.suspect_frac)
            target = (cnum in forced) or (not forced and suspect)
            if not target:
                add_cycle_to_index(idx, cd, graph_seen, gate)  # reference only
                continue

            rec = _recorded_stats(cd)
            results = recompute_cycle_novelty(idx, cd, args.sim_threshold, graph_seen, gate)
            n_novel = sum(1 for _, nov, _ in results if nov)
            n_nonnovel = len(results) - n_novel
            rec_nonnovel = int(rec.get("not_novel_skipped", st["nonnovel"]))
            print(f"  cycle {cnum:>3}: extractable={len(results):>3} | recorded non-novel="
                  f"{rec_nonnovel:>3} -> recomputed novel={n_novel:>3} non-novel={n_nonnovel:>3}"
                  + ("   [would fix]" if n_novel > 0 else "   [confirmed non-novel — no change]"))

            if args.dry_run or n_novel == 0:
                continue

            restored, demoted = restore_labels(results)
            print(f"    restored {restored} gen(s) to new_nn.py, demoted {demoted}; evaluating...",
                  flush=True)
            try:
                st2 = scan_cycle(cd)
                run_eval_for_cycle(
                    st2, dataset=args.dataset, transform=args.eval_transform, lr=args.eval_lr,
                    momentum=args.eval_momentum, batch=args.eval_batch, dropout=args.eval_dropout,
                    train_epochs=args.eval_train_epochs, multi_gpu=args.multi_gpu)
            except Exception as e:  # noqa: BLE001
                print(f"    [ERROR] NNEval failed for cycle {cnum}: {type(e).__name__}: {e}")
                continue
            st3 = scan_cycle(cd)
            st3["cycle_dir"] = cd
            metrics = reconstruct_metrics(st3, sub, args.accuracy_threshold)
            metrics["reeval_novelty_recomputed"] = True
            (cd / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            b = metrics["bucketing"]
            fixed += 1
            print(f"    wrote metrics.json | desirable={b['new_desirable']} "
                  f"not_novel={b['not_novel_skipped']} best={b['best_accuracy']*100:.2f}% "
                  f"avg(>=thr)={b['avg_accuracy']*100:.2f}% evaluated={b['evaluated_accuracies']}")

    print("=" * 74)
    if args.dry_run:
        print("dry-run: nothing evaluated or modified.")
    else:
        print(f"Fixed {fixed} cycle(s). Re-plot with slurm_nngpt_kto_plot.sh.")


if __name__ == "__main__":
    main()

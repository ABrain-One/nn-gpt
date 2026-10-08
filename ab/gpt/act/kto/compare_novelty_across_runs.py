#!/usr/bin/env python3
"""Compare base (Jaccard) vs graph novelty across runs, cycle by cycle.

For every run and every cycle up to --max_cycle, this recomputes — FRESH and with the
SAME method for ALL runs (including the *_graphgate ones, re-checked from scratch) —
how many of the generated architectures are novel under two definitions:

  * base novelty : MinHash-Jaccard nearest similarity < threshold vs LEMUR DB + every
                   earlier generation of that run.
  * graph novelty: canonical graph (torch.fx make_fx aten trace + Weisfeiler-Lehman
                   hash) not seen in any earlier generation of that run (generations
                   only, no DB — as the live graph gate builds it). Untraceable archs
                   fail open (counted novel), matching the gate.

It produces two comparison plots — one per definition, one line per run — so the runs
are measured on ONE consistent metric each, plus a CSV. Read-only; nothing is modified.

  # in-container (CPU; graph tracing is the slow part — parallelised, cached):
  python -m ab.gpt.act.kto.compare_novelty_across_runs --runs_root . --out_dir novelty_compare
  python -m ab.gpt.act.kto.compare_novelty_across_runs \
      --run out_3062863/nngpt/divpo_core --run out_3062862/nngpt/divpo_anchors --out_dir novelty_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ab.gpt.act.kto.reeval_novelty_cycles import (
    _gen_dirs, _read_gen_code, _gen_code_file,
)


def _trace_worker(job: Tuple[str, dict]) -> Tuple[str, Optional[str]]:
    """Graph hash of one generation file (runs in a worker process)."""
    path_str, gate = job
    from ab.gpt.act.kto.graph_novelty_audit import graph_hash_for_file
    try:
        h, _ = graph_hash_for_file(Path(path_str), gate["sizes"], gate["in_ch"],
                                   gate["n_cls"], "aten", gate["granularity"],
                                   gate["wl_rounds"])
        return path_str, h
    except Exception:  # noqa: BLE001
        return path_str, None


def discover_runs(explicit: List[str], runs_root: Optional[str]) -> List[Tuple[str, Path]]:
    """(label, path) for each run output_subdir holding cycle_*/nneval/."""
    runs: List[Tuple[str, Path]] = []
    seen = set()

    def add(p: Path) -> None:
        if not any((c / "nneval").is_dir() for c in p.glob("cycle_*")):
            return
        label = p.name
        if label in seen:                      # disambiguate same subdir under two tags
            label = f"{p.parent.parent.name}:{label}"
        if label in seen:
            return
        seen.add(label)
        runs.append((label, p))

    for r in explicit or []:
        add(Path(r))
    if runs_root:
        for p in sorted(Path(runs_root).glob("out_*/nngpt/*")):
            if p.is_dir():
                add(p)
    return runs


def present_cycles(run_path: Path, max_cycle: int) -> List[int]:
    """Cycle numbers whose FOLDER exists (ground truth, not metrics), sorted. A
    genuinely-missing folder in between is simply absent, so downstream renumbering
    collapses the gap. Capped to the first max_cycle present folders (0 = no cap)."""
    nums = []
    for p in run_path.glob("cycle_*"):
        s = p.name.split("_")
        if p.is_dir() and len(s) > 1 and s[1].isdigit():
            nums.append(int(s[1]))
    nums.sort()
    return nums[:max_cycle] if max_cycle and max_cycle > 0 else nums


def _iter_gen_files(run_path: Path, max_cycle: int):
    for orig in present_cycles(run_path, max_cycle):
        for gd in _gen_dirs(run_path / f"cycle_{orig}"):
            cf = _gen_code_file(gd)
            if cf is not None:
                yield orig, cf


def compute_graph_hashes(all_files: List[str], gate: dict, cache_file: Path,
                         workers: int) -> Dict[str, Optional[str]]:
    """Graph hash for every generation file, cached to disk and computed in parallel."""
    cache: Dict[str, Optional[str]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
    todo = [f for f in all_files if f not in cache]
    print(f"[graph] {len(all_files)} generations · {len(cache)} cached · {len(todo)} to trace "
          f"({workers} worker(s))", flush=True)
    if todo:
        jobs = [(f, gate) for f in todo]
        done = 0
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as ex:
                for path_str, h in ex.map(_trace_worker, jobs, chunksize=8):
                    cache[path_str] = h
                    done += 1
                    if done % 200 == 0 or done == len(todo):
                        print(f"[graph] traced {done}/{len(todo)}", flush=True)
                        cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — fall back to sequential
            print(f"[graph] pool failed ({type(e).__name__}: {e}); tracing sequentially", flush=True)
            for f in todo:
                if f in cache:
                    continue
                cache[f] = _trace_worker((f, gate))[1]
                done += 1
                if done % 100 == 0:
                    print(f"[graph] traced {done}/{len(todo)}", flush=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    n_ok = sum(1 for v in cache.values() if v is not None)
    print(f"[graph] hashes ready ({n_ok}/{len(cache)} traceable)", flush=True)
    return cache


def novelty_per_cycle(run_path: Path, max_cycle: int, threshold: float,
                      sim_db_task: str, sim_db_dataset: str,
                      graph_hashes: Dict[str, Optional[str]]
                      ) -> Dict[int, Optional[Dict[str, int]]]:
    """Stream cycles in order; per cycle count base-novel and graph-novel generations."""
    from ab.gpt.act.kto.similarity_penalty import SimilarityIndex
    idx = SimilarityIndex(threshold=threshold)
    idx.add_db(sim_db_task, sim_db_dataset)
    graph_seen = set()
    out: Dict[int, Optional[Dict[str, int]]] = {}
    # Iterate only cycles whose FOLDER exists, in original order, but key the result by
    # a contiguous renumbering (1..N) so a missing folder in between collapses the gap.
    # The novelty values are still computed against the true prior generations.
    for new, orig in enumerate(present_cycles(run_path, max_cycle), start=1):
        base_novel = graph_novel = total = 0
        for gd in _gen_dirs(run_path / f"cycle_{orig}"):
            cf = _gen_code_file(gd)
            if cf is None:
                continue
            code = _read_gen_code(gd)
            if not code or not code.strip():
                continue
            total += 1
            if idx.nearest_jaccard(code) < threshold:
                base_novel += 1
            idx.add(code)
            h = graph_hashes.get(str(cf))
            if h is None:                       # untraceable → fail open (novel)
                graph_novel += 1
            else:
                if h not in graph_seen:
                    graph_novel += 1
                graph_seen.add(h)
        out[new] = {"base_novel": base_novel, "graph_novel": graph_novel,
                    "total": total, "orig": orig}
    return out


def _plot(results: Dict[str, Dict[int, Optional[Dict[str, int]]]], key: str,
          title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    xs = sorted({c for per in results.values() for c in per})  # contiguous renumbered
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (label, per) in enumerate(sorted(results.items())):
        ys = [per[c][key] if c in per else float("nan") for c in xs]
        ax.plot(xs, ys, marker=markers[i % len(markers)], markersize=4, linewidth=1.6,
                color=cmap(i % 10), label=label)
    ax.set_xlabel("Cycle (missing cycles collapsed)"); ax.set_ylabel("Novel architectures generated")
    ax.set_title(title); ax.set_xticks(xs); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[],
                   help="A run output_subdir (…/nngpt/<subdir>). Repeatable.")
    p.add_argument("--runs_root", type=str, default=None,
                   help="Auto-discover out_*/nngpt/*/ under this dir")
    p.add_argument("--out_dir", type=str, default="novelty_compare")
    p.add_argument("--max_cycle", type=int, default=20)
    p.add_argument("--sim_threshold", type=float, default=0.85)
    p.add_argument("--sim_db_task", type=str, default="img-classification")
    p.add_argument("--sim_db_dataset", type=str, default="cifar-10")
    p.add_argument("--graph_gate_granularity", choices=["topology", "typed"], default="topology")
    p.add_argument("--graph_gate_sizes", type=str, default="32,64,96,224,256")
    p.add_argument("--graph_gate_wl_rounds", type=int, default=3)
    p.add_argument("--graph_gate_num_classes", type=int, default=10)
    p.add_argument("--graph_gate_in_channels", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(12, (mp.cpu_count() or 4)))
    args = p.parse_args()

    runs = discover_runs(args.run, args.runs_root)
    if not runs:
        raise SystemExit("No runs found (use --run <path> and/or --runs_root <dir>).")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Runs:")
    for label, path in runs:
        print(f"  {label:40s} {path}")

    gate = {
        "sizes": [int(s) for s in args.graph_gate_sizes.split(",") if s.strip()],
        "in_ch": args.graph_gate_in_channels, "n_cls": args.graph_gate_num_classes,
        "granularity": args.graph_gate_granularity, "wl_rounds": args.graph_gate_wl_rounds,
    }

    all_files: List[str] = []
    for _, path in runs:
        all_files += [str(cf) for _, cf in _iter_gen_files(path, args.max_cycle)]
    all_files = sorted(set(all_files))
    graph_hashes = compute_graph_hashes(all_files, gate,
                                        out_dir / "graph_hash_cache.json", args.workers)

    results: Dict[str, Dict[int, Optional[Dict[str, int]]]] = {}
    for label, path in runs:
        print(f"[novelty] {label} …", flush=True)
        results[label] = novelty_per_cycle(
            path, args.max_cycle, args.sim_threshold,
            args.sim_db_task, args.sim_db_dataset, graph_hashes)

    cap = f" (first {args.max_cycle} present cycles)" if args.max_cycle and args.max_cycle > 0 else ""
    _plot(results, "base_novel",
          f"Base (Jaccard) novel architectures per cycle{cap}",
          out_dir / "base_novelty_comparison.png")
    _plot(results, "graph_novel",
          f"Graph-distinct novel architectures per cycle{cap}",
          out_dir / "graph_novelty_comparison.png")

    csv_path = out_dir / "novelty_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "cycle", "orig_cycle", "total", "base_novel", "graph_novel"])
        for label, per in sorted(results.items()):
            for cyc in sorted(per):
                d = per[cyc]
                w.writerow([label, cyc, d.get("orig", cyc), d["total"],
                            d["base_novel"], d["graph_novel"]])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

"""LLMatic MAP-Elites archive for the chromosome GA in ab/gpt/brute/ga/meta_evolution.

Opt-in: nothing here runs unless the environment variable ``LLMATIC_GA`` is set
(to anything other than ``0``/``false``/``no``/``off``). With it unset,
:func:`from_env` returns None and the GA runners skip every hook.

When enabled, the GA runner (run_fractal_evolution.py / run_fractal_frozen.py):

  * adds every evaluated individual to a :class:`MAPElitesArchive` whose cells are
    (param count, depth) of the generated model, read by :func:`extract_descriptors`;
  * saves the archive to a JSON file in ``PIPELINE_DIR`` at the end of each run,
    storing the chromosome of every elite;
  * on the next run (the next meta-iteration restart), replaces part of the random
    initial population with chromosomes of elites from distinct cells.

Selection, crossover, mutation, elitism and fitness are untouched, so the GA
operators the meta-LLM rewrites keep full control of the search; only the
starting population of each restart changes.

Environment:
  LLMATIC_GA                   enable (unset = off)
  LLMATIC_GA_RESEED_FRACTION   share of the initial population taken from the
                               archive (default 0.5, clamped to [0, 1])
  LLMATIC_GA_ARCHIVE           archive JSON path (default: PIPELINE_DIR/
                               llmatic_ga_archive_<dataset>_<MODEL_NAME>[_<RUN_TS>][_seed<FROZEN_SEED>].json)
  LLMATIC_GA_PARAMS_RANGE      "min,max" parameter count spanned by the grid's params axis
  LLMATIC_GA_DEPTH_RANGE       "min,max" module count spanned by the grid's depth axis
                               (both unset = MAPElitesArchive defaults, 1e5..1e8 and ~4..158)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Optional

from ab.gpt.util.method.llmatic.archive import MAPElitesArchive
from ab.gpt.util.method.llmatic.descriptors import extract_descriptors

ENV_FLAG = "LLMATIC_GA"
ENV_FRACTION = "LLMATIC_GA_RESEED_FRACTION"
ENV_ARCHIVE = "LLMATIC_GA_ARCHIVE"
ENV_PARAMS_RANGE = "LLMATIC_GA_PARAMS_RANGE"
ENV_DEPTH_RANGE = "LLMATIC_GA_DEPTH_RANGE"
DEFAULT_FRACTION = 0.5
_FORMAT_VERSION = 1


def enabled() -> bool:
    value = os.environ.get(ENV_FLAG, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def reseed_fraction() -> float:
    try:
        fraction = float(os.environ.get(ENV_FRACTION, DEFAULT_FRACTION))
    except ValueError:
        fraction = DEFAULT_FRACTION
    return min(max(fraction, 0.0), 1.0)


def _log_range(name: str) -> Optional[tuple]:
    """log10 (min, max) from a "min,max" env var; None when unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        lo, hi = (float(x) for x in raw.split(","))
        if 0 < lo < hi:
            return (math.log10(lo), math.log10(hi))
    except ValueError:
        pass
    print(f"[llmatic-ga] ignoring {name}={raw!r} (expected 'min,max' with 0 < min < max)", flush=True)
    return None


def default_archive_path(pipeline_dir: str, dataset: str) -> str:
    parts = [dataset, os.environ.get("MODEL_NAME") or "unknown"]
    if os.environ.get("RUN_TS"):
        parts.append(os.environ["RUN_TS"])
    if os.environ.get("FROZEN_SEED"):
        parts.append(f"seed{os.environ['FROZEN_SEED']}")
    return os.path.join(pipeline_dir, f"llmatic_ga_archive_{'_'.join(parts)}.json")


def from_env(pipeline_dir: str, dataset: str) -> Optional["GABridge"]:
    """Return a loaded bridge when ``LLMATIC_GA`` is set, otherwise None."""
    if not enabled():
        return None
    path = os.environ.get(ENV_ARCHIVE) or default_archive_path(pipeline_dir, dataset)
    num_classes = 100 if "100" in str(dataset) else 10
    return GABridge(path, num_classes=num_classes,
                    param_log_range=_log_range(ENV_PARAMS_RANGE), depth_log_range=_log_range(ENV_DEPTH_RANGE))


class GABridge:
    """Persistent MAP-Elites archive of GA chromosomes."""

    def __init__(self, path: str, num_classes: int = 10, rng: Optional[random.Random] = None,
                 param_log_range: Optional[tuple] = None, depth_log_range: Optional[tuple] = None):
        self.path = path
        self.num_classes = num_classes
        # Only pass ranges that were given, so MAPElitesArchive keeps its own defaults otherwise.
        self._grid = {k: v for k, v in (("param_log_range", param_log_range),
                                        ("depth_log_range", depth_log_range)) if v is not None}
        self.archive = MAPElitesArchive(rng=rng, **self._grid)
        self.iteration = 1
        self.load()

    # ------------------------------------------------------------------ #
    def add(self, chromosome: dict, code_str: str, fitness) -> bool:
        """Add one evaluated individual; returns True if it became a cell elite."""
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            return False
        if fitness <= 0:  # failed evaluations (0.0) and duplicate penalties (-1.0)
            return False
        descriptors = extract_descriptors(code_str, out_shape=(self.num_classes,))
        if descriptors is None:
            return False
        return self.archive.add(
            nn_name=hashlib.md5(code_str.encode()).hexdigest(),
            accuracy=fitness,
            descriptors=descriptors,
            source="generated",
            row={"chromosome": dict(chromosome)},
        )

    def reseed(self, pop_size: int, fraction: Optional[float] = None) -> list:
        """Chromosomes of up to ``fraction * pop_size`` elites, one per distinct cell."""
        if fraction is None:
            fraction = reseed_fraction()
        elites = list(self.archive.cells.values())
        k = min(int(round(pop_size * fraction)), len(elites))
        if k <= 0:
            return []
        return [dict(e.row["chromosome"]) for e in self.archive._rng.sample(elites, k)]

    def patch_initial_population(self, ga) -> None:
        """Make ``ga`` start from archive elites: the GA still builds its random
        population, then the first K individuals are replaced by reseeded chromosomes."""
        original = ga._initialize_population

        def _initialize_population():
            original()
            chromosomes = self.reseed(len(ga.population))
            for ind, chromosome in zip(ga.population, chromosomes):
                ind['chromosome'] = chromosome
                ind['fitness'] = None
            print(f"[llmatic-ga] reseeded {len(chromosomes)}/{len(ga.population)} from archive "
                  f"({len(self.archive.cells)} filled cells)", flush=True)

        ga._initialize_population = _initialize_population

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[llmatic-ga] could not read archive {self.path}: {e}", flush=True)
            return
        # Elites are re-binned from their stored params/depth on the current grid.
        self.archive = MAPElitesArchive(rng=self.archive._rng, **self._grid)
        for e in data.get("elites", []):
            self.archive.add(
                nn_name=e["nn_name"],
                accuracy=e["accuracy"],
                descriptors={"params": e["params"], "depth": e["depth"]},
                source="archive",
                row={"chromosome": e["chromosome"]},
            )
        self.iteration = int(data.get("iterations", 0)) + 1
        print(f"[llmatic-ga] loaded archive {self.path}: "
              f"{len(self.archive.cells)} filled cells from {self.iteration - 1} iteration(s)", flush=True)

    def save(self) -> None:
        a = self.archive
        data = {
            "version": _FORMAT_VERSION,
            "iterations": self.iteration,
            "config": {
                "param_bins": a.param_bins,
                "depth_bins": a.depth_bins,
                "param_log_range": [a.param_log_lo, a.param_log_hi],
                "depth_log_range": [a.depth_log_lo, a.depth_log_hi],
            },
            "stats": a.get_stats(),
            "elites": [
                {"cell": list(e.cell), "nn_name": e.nn_name, "accuracy": e.accuracy,
                 "params": e.params, "depth": e.depth, "chromosome": e.row["chromosome"]}
                for e in a.cells.values()
            ],
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, self.path)
        self.log_stats()

    def log_stats(self) -> None:
        s = self.archive.get_stats()
        print(f"[llmatic-ga] iter {self.iteration} coverage={s['coverage']:.2f} "
              f"filled={s['filled_cells']}/{s['total_cells']} qd={s['qd_score']:.2f} "
              f"best={s['best_accuracy']:.4f}", flush=True)

"""MAP-Elites archive for LLMatic-style quality diversity over NN architectures.

The archive is a 2-D grid over two *behavioral descriptors*:

  * axis 1 -- ``param_count`` on a log10 scale (network complexity / size)
  * axis 2 -- ``depth`` = number of layers (structural depth)

Each grid cell keeps a single *elite*: the highest-accuracy architecture whose
descriptors fall in that cell. This is the classic MAP-Elites illumination
scheme (Mouret & Clune, 2015) as used by LLMatic's network archive (Nasir et
al., 2024): fitness = accuracy, behavior = (complexity, depth).

Lifecycle in this pipeline: generation runs in a short-lived subprocess per
cycle, so the archive is *in-memory only* and rebuilt from the persisted corpus
at the start of each cycle via :meth:`init_from_corpus`. Because evaluated
models from earlier cycles are saved back into the corpus, re-initialising from
the corpus each cycle is what carries archive state across cycles -- no
cross-process state is needed.

This module is deliberately dependency-light (stdlib + ``ab.nn.api``) and has no
side effects on import.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# Cell contents
# --------------------------------------------------------------------------- #
@dataclass
class Elite:
    """The best-so-far architecture occupying one grid cell."""

    nn_name: str
    accuracy: float
    params: int
    depth: int
    nn_code: Optional[str] = None          # source text (needed for crossover prompts)
    nn_code_path: Optional[str] = None     # optional on-disk path, if known
    prm: Optional[dict] = None             # hyperparameters that achieved `accuracy`
    source: str = "corpus"                 # "corpus" | "generated"
    cell: Optional[tuple] = None           # (i, j) grid index, filled in by add()
    row: Optional[dict] = None             # full corpus row (template fields for prompts)

    def to_dict(self) -> dict:
        d = asdict(self)
        # nn_code / row can be large; keep the dump light for per-cycle stats files.
        if d.get("nn_code") is not None:
            d["nn_code"] = f"<{len(self.nn_code)} chars>"
        if d.get("row") is not None:
            d["row"] = f"<row: {len(self.row)} cols>"
        return d


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #
class MAPElitesArchive:
    """A 2-D MAP-Elites grid: (log param_count) x (depth) -> best accuracy.

    Parameters
    ----------
    param_bins, depth_bins:
        Grid resolution per axis (default 10 x 10 = 100 cells).
    param_log_range:
        (min, max) exponents for the log10 param axis. Default (5.0, 8.0),
        calibrated to the observed corpus (bulk of models ~1e7 params, tail down
        to ~1e5 after mutation). Out-of-range values clamp to the edge bins.
    depth_log_range:
        (min, max) exponents for the log10 *depth* (layer count) axis. Default
        (0.6, 2.2) ~= 4 .. 158 layers. Log scale is used because the corpus is
        bimodal in depth (a cluster near ~24 layers and another near ~120), with
        a long tail; a linear axis would collapse the dense region into one bin.
        Clamped.
    """

    def __init__(
        self,
        param_bins: int = 10,
        depth_bins: int = 10,
        param_log_range: tuple = (5.0, 8.0),
        depth_log_range: tuple = (0.6, 2.2),
        rng: Optional[random.Random] = None,
    ):
        self.param_bins = int(param_bins)
        self.depth_bins = int(depth_bins)
        self.param_log_lo, self.param_log_hi = float(param_log_range[0]), float(param_log_range[1])
        self.depth_log_lo, self.depth_log_hi = float(depth_log_range[0]), float(depth_log_range[1])
        self._rng = rng or random.Random()
        # cell (i, j) -> Elite
        self.cells: dict[tuple, Elite] = {}
        # bookkeeping
        self.n_add_attempts = 0
        self.n_improvements = 0

    # --------------------------- binning --------------------------------- #
    def _clamp_bin(self, value: float, lo: float, hi: float, nbins: int) -> int:
        """Map a scalar onto [0, nbins-1], clamping out-of-range values."""
        if hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        idx = int(frac * nbins)
        if idx < 0:
            return 0
        if idx >= nbins:
            return nbins - 1
        return idx

    def cell_index(self, params: float, depth: float) -> tuple:
        """Return the (i, j) grid cell for a (param_count, depth) pair.

        Both axes are log10-scaled (see class docstring).
        """
        params = max(float(params), 1.0)  # avoid log10(0)
        depth = max(float(depth), 1.0)
        i = self._clamp_bin(math.log10(params), self.param_log_lo, self.param_log_hi, self.param_bins)
        j = self._clamp_bin(math.log10(depth), self.depth_log_lo, self.depth_log_hi, self.depth_bins)
        return (i, j)

    # --------------------------- insertion ------------------------------- #
    def add(
        self,
        nn_name: str,
        accuracy: float,
        descriptors: dict,
        nn_code: Optional[str] = None,
        nn_code_path: Optional[str] = None,
        prm: Optional[dict] = None,
        source: str = "corpus",
        row: Optional[dict] = None,
    ) -> bool:
        """Insert a candidate; keep it only if it beats its cell's current elite.

        `descriptors` must contain ``params`` and ``depth``. Returns True if the
        candidate became (or replaced) the cell's elite.
        """
        self.n_add_attempts += 1
        try:
            params = descriptors["params"]
            depth = descriptors["depth"]
        except (KeyError, TypeError):
            return False
        if params is None or depth is None or accuracy is None:
            return False
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError):
            return False

        cell = self.cell_index(params, depth)
        incumbent = self.cells.get(cell)
        if incumbent is not None and incumbent.accuracy >= accuracy:
            return False

        self.cells[cell] = Elite(
            nn_name=nn_name,
            accuracy=accuracy,
            params=int(params),
            depth=int(depth),
            nn_code=nn_code,
            nn_code_path=nn_code_path,
            prm=prm,
            source=source,
            cell=cell,
            row=row,
        )
        self.n_improvements += 1
        return True

    # ----------------------- corpus initialisation ----------------------- #
    @staticmethod
    def load_descriptor_map() -> dict:
        """Build ``{nn_name: (params, depth)}`` from the ``nn_stat`` DB table.

        This is the fast descriptor source: one cached ``nn_stat_data()`` call
        (~15s cold, free warm) instead of the ``include_nn_stats=True`` join,
        which is unusable here (minutes). ``nn_stat`` has one row per (nn, prm);
        params/depth are architecture properties, so we take the first non-null
        per ``nn_name``. Returns ``{}`` on any failure -- never fatal.
        """
        try:
            from ab.nn import api as lemur

            st = lemur.nn_stat_data()
            if st is None or getattr(st, "empty", True):
                return {}
            cols = set(st.columns)
            if not {"nn_name", "total_params", "total_layers"}.issubset(cols):
                return {}
            st = st.dropna(subset=["total_params", "total_layers"])
            # first non-null (params/depth) per nn_name
            grp = st.groupby("nn_name")[["total_params", "total_layers"]].first()
            return {name: (row["total_params"], row["total_layers"]) for name, row in grp.iterrows()}
        except Exception:
            return {}

    def init_from_corpus(
        self,
        dataset: Optional[str] = None,
        nn_prefixes: Optional[tuple] = None,
        task: Optional[str] = None,
        metric: Optional[str] = None,
        df=None,
        descriptor_map: Optional[dict] = None,
    ) -> int:
        """Populate the archive from the persisted LEMUR corpus.

        Descriptors (``params``/``depth``) come from the ``nn_stat`` table via
        :meth:`load_descriptor_map`; accuracy / code / prm come from the corpus
        ``data()`` rows. Returns the number of elites inserted.

        Parameters
        ----------
        df:
            Optional pre-fetched corpus DataFrame (the one ``nn_gen`` already
            built). When ``None``, this calls ``ab.nn.api.data`` with the *same*
            signature ``nn_gen`` uses so the ``lru_cache`` makes it free at run
            time. Must have ``nn`` and ``accuracy`` columns.
        descriptor_map:
            Optional pre-built ``{nn_name: (params, depth)}`` map (to reuse
            across archives). When ``None``, :meth:`load_descriptor_map` is used.

        Import of ``ab.nn.api`` is deferred to call time so importing this module
        never triggers a DB read.
        """
        if df is None:
            from ab.nn import api as lemur

            kwargs = dict(only_best_accuracy=True)
            if task is not None:
                kwargs["task"] = task
            if dataset is not None:
                kwargs["dataset"] = dataset
            if metric is not None:
                kwargs["metric"] = metric
            if nn_prefixes is not None:
                kwargs["nn_prefixes"] = tuple(nn_prefixes)
            df = lemur.data(**kwargs)

        if df is None or getattr(df, "empty", True) or "nn" not in df.columns:
            return 0

        if descriptor_map is None:
            descriptor_map = self.load_descriptor_map()
        if not descriptor_map:
            return 0

        before = self.n_improvements
        for _, row in df.iterrows():
            desc = descriptor_map.get(row.get("nn"))
            if desc is None:
                continue
            params, depth = desc
            # NaN guard (pandas numeric NaN != itself)
            try:
                if params != params or depth != depth:  # noqa: PLR0124
                    continue
            except TypeError:
                continue
            self.add(
                nn_name=row.get("nn"),
                accuracy=row.get("accuracy"),
                descriptors={"params": params, "depth": depth},
                nn_code=row.get("nn_code") if "nn_code" in df.columns else None,
                prm=row.get("prm") if "prm" in df.columns else None,
                source="corpus",
                row=row.to_dict(),
            )
        return self.n_improvements - before

    # --------------------------- sampling -------------------------------- #
    def _filled(self) -> list:
        return list(self.cells.values())

    def sample_parent(self, strategy: str = "uniform_cell") -> Optional[Elite]:
        """Select one elite as a mutation seed.

        * ``uniform_cell`` (default): pick a filled cell uniformly at random.
          This is the MAP-Elites exploration default -- it spreads selection
          pressure across the behavior space rather than crowding high-accuracy
          regions.
        * ``best``: return the global best-accuracy elite (exploitation).
        """
        filled = self._filled()
        if not filled:
            return None
        if strategy == "best":
            return max(filled, key=lambda e: e.accuracy)
        return self._rng.choice(filled)

    def sample_parents_for_crossover(self, n: int = 2, distinct_cells: bool = True) -> list:
        """Select ``n`` elites from *different* cells for a crossover prompt.

        Drawing from distinct cells maximises structural diversity between the
        parents (different size/depth regions), which is the point of crossover.
        Falls back to sampling with replacement if too few cells are filled.
        """
        filled = self._filled()
        if not filled:
            return []
        if distinct_cells and len(filled) >= n:
            return self._rng.sample(filled, n)
        return [self._rng.choice(filled) for _ in range(n)]

    # ----------------------------- stats --------------------------------- #
    def get_stats(self) -> dict:
        """Return QD summary metrics for logging / thesis plots."""
        total = self.param_bins * self.depth_bins
        filled = self._filled()
        n_filled = len(filled)
        accs = [e.accuracy for e in filled]
        best = max(filled, key=lambda e: e.accuracy) if filled else None
        return {
            "grid": [self.param_bins, self.depth_bins],
            "total_cells": total,
            "filled_cells": n_filled,
            "coverage": n_filled / total if total else 0.0,
            "qd_score": sum(accs),                       # sum of elite accuracies
            "best_accuracy": best.accuracy if best else 0.0,
            "best_nn": best.nn_name if best else None,
            "best_cell": list(best.cell) if best else None,
            "mean_elite_accuracy": (sum(accs) / n_filled) if n_filled else 0.0,
            "add_attempts": self.n_add_attempts,
            "improvements": self.n_improvements,
        }

    def to_dict(self) -> dict:
        """Full serialisable snapshot (elite code is summarised, not inlined)."""
        return {
            "config": {
                "param_bins": self.param_bins,
                "depth_bins": self.depth_bins,
                "param_log_range": [self.param_log_lo, self.param_log_hi],
                "depth_log_range": [self.depth_log_lo, self.depth_log_hi],
            },
            "stats": self.get_stats(),
            "cells": {f"{i},{j}": e.to_dict() for (i, j), e in self.cells.items()},
        }

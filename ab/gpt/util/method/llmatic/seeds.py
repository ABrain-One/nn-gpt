"""LLMatic seed selection for the iterative generation pipeline.

Encapsulates the MAP-Elites seed-selection logic (archive-driven *mutation* every
cycle, plus periodic *crossover* combining two parents from different cells) so
``ab.gpt.util.Tune.nn_gen`` can call it as a single hook instead of the default
random LEMUR sampling.

Design: this helper simply returns the seed-row DataFrame that ``nn_gen`` already
consumes, leaving the rest of the pipeline untouched. For crossover, each row carries a
``__llmatic_prompt__`` column holding the fully-rendered two-parent prompt, which
``nn_gen``'s prompt-assembly step uses verbatim.

Robustness: any deviation (empty archive, insufficient diversity, or any
exception) returns ``None`` so the caller falls back to random sampling. Seed
selection can never break generation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LLMATIC_PROMPT_COL = "__llmatic_prompt__"


def cycle_from_path(out_path) -> int:
    """Extract the cycle number from a path like ``.../cycle_3/generation/...``."""
    m = re.search(r"cycle[_-]?(\d+)", str(out_path))
    return int(m.group(1)) if m else 1


def _crossover_every(llmatic_cfg: dict) -> int:
    try:
        return max(0, int(llmatic_cfg.get("crossover_every", 3)))
    except (TypeError, ValueError):
        return 3


def _minimal_row(elite, corpus_df=None) -> dict:
    """A seed row for an elite, preferring its stored corpus row.

    Falls back to a lookup in ``corpus_df`` (the DataFrame ``nn_gen`` already
    fetched) so the row keeps the real columns (nn_code, task, dataset, metric,
    prm, ...) that downstream pickling/eval expect.
    """
    if getattr(elite, "row", None):
        return dict(elite.row)
    if corpus_df is not None and "nn" in getattr(corpus_df, "columns", []):
        match = corpus_df[corpus_df["nn"] == elite.nn_name]
        if len(match):
            return match.iloc[0].to_dict()
    return {
        "nn": elite.nn_name,
        "nn_code": getattr(elite, "nn_code", "") or "",
        "accuracy": getattr(elite, "accuracy", None),
        "prm": getattr(elite, "prm", None),
    }


def build_archive(key_config: dict):
    """Build and populate a MAP-Elites archive matching ``nn_gen``'s seed query.

    Returns the archive, or ``None`` if it could not be built / is empty.
    """
    try:
        from ab.gpt.util.method.llmatic.archive import MAPElitesArchive
        from ab.gpt.util.Const import DEFAULT_DATASET, DEFAULT_NN_PREFIXES

        gen_dataset = key_config.get("dataset", DEFAULT_DATASET)
        gen_prefixes = tuple(key_config.get("nn_prefixes") or DEFAULT_NN_PREFIXES)
        task = key_config.get("task")

        archive = MAPElitesArchive()
        archive.init_from_corpus(dataset=gen_dataset, nn_prefixes=gen_prefixes, task=task)
        return archive if archive.cells else None
    except Exception as e:  # never break generation
        print(f"[llmatic] archive build failed ({e!r}) -> random sampling", flush=True)
        return None


def select_seed_rows(archive, corpus_df, key_config, cycle, test_nn, out_path=None):
    """Return a DataFrame of ``test_nn`` archive-selected seed rows, or ``None``.

    Mutation cycles return real corpus rows for archive-sampled parents (so the
    existing prompt template renders normally). Crossover cycles additionally
    attach a ``__llmatic_prompt__`` column with the rendered two-parent prompt.
    Returning ``None`` signals the caller to use the default random sampling.

    ``out_path`` (the cycle's generation dir) is where the per-cycle
    ``llmatic_archive_stats.json`` is written for the coverage/QD plots.
    """
    try:
        import pandas as pd
        from ab.gpt.util.method.llmatic.prompts import build_crossover_prompt

        cfg = key_config.get("llmatic") or {}
        stats = archive.get_stats()
        every = _crossover_every(cfg)
        is_crossover = every > 0 and cycle % every == 0 and stats.get("filled_cells", 0) >= 2
        operator = "crossover" if is_crossover else "mutation"

        nn_code_max_chars = key_config.get("nn_code_max_chars")
        rows = []
        if is_crossover:
            for i in range(test_nn):
                pair = archive.sample_parents_for_crossover(2)
                if len(pair) < 2:
                    return None  # not enough diversity -> caller falls back
                pa, pb = pair[0], pair[1]
                _, prompt_text = build_crossover_prompt(
                    _minimal_row(pa, corpus_df), _minimal_row(pb, corpus_df),
                    nn_code_max_chars=(nn_code_max_chars or 1200),
                )
                row = _minimal_row(pa, corpus_df)
                row["nn"] = f"{pa.nn_name}__xover{i}"
                row[LLMATIC_PROMPT_COL] = prompt_text
                rows.append(row)
        else:
            for i in range(test_nn):
                parent = archive.sample_parent("uniform_cell")
                if parent is None:
                    return None
                row = _minimal_row(parent, corpus_df)
                row["nn"] = f"{parent.nn_name}__mut{i}"
                rows.append(row)

        _dump_stats(archive, cycle, operator, test_nn, out_path)
        return pd.DataFrame(rows)
    except Exception as e:  # never break generation
        print(f"[llmatic] seed selection failed ({e!r}) -> random sampling", flush=True)
        return None


def _dump_stats(archive, cycle, operator, test_nn, out_path=None) -> None:
    """Log per-cycle archive stats and, if ``out_path`` is given, persist them as
    ``llmatic_archive_stats.json`` for the coverage/QD plots. Non-essential;
    failures are swallowed.
    """
    try:
        stats = archive.get_stats()
        stats.update(cycle=cycle, operator=operator, candidates=test_nn)
        if out_path is not None:
            dest = Path(out_path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "llmatic_archive_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[llmatic] cycle {cycle} [{operator}] candidates={test_nn} "
              f"coverage={stats.get('coverage', 0):.2f} "
              f"filled={stats.get('filled_cells', 0)}/{stats.get('total_cells', 0)} "
              f"qd={stats.get('qd_score', 0):.2f} best={stats.get('best_accuracy', 0):.4f}",
              flush=True)
    except Exception as e:
        print(f"[llmatic] could not dump archive stats: {e}", flush=True)

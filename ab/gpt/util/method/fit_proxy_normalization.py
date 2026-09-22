#!/usr/bin/env python3
"""
Fit proxy normalization statistics on the training-family architectures only
(so val/test scale never leaks in), from proxy_cache.jsonl, and write them to
proxy_norm_stats.json.

Use fit_and_save_norm_stats() as a library, or run standalone:
    python -m ab.gpt.util.method.fit_proxy_normalization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ab.gpt.util.method.zero_cost_proxies import PROXY_NAMES, fit_normalization_stats

ACC_DIR = Path(__file__).resolve().parents[4] / "out" / "acc_predict"
RAW_PATH = ACC_DIR / "llm_finetuning_data.jsonl"
CACHE_PATH = ACC_DIR / "proxy_cache.jsonl"
STATS_PATH = ACC_DIR / "proxy_norm_stats.json"


def _load_family_lookup(path: Path, family_fn: Callable[[str, str], str]) -> dict[tuple[str, str], str]:
    """(nn, dataset) -> architecture family, first occurrence per key."""
    lookup: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("nn"), rec.get("dataset"))
            if key in lookup or key[0] is None:
                continue
            lookup[key] = family_fn(rec.get("nn_code", "") or "", rec.get("task", "") or "")
    return lookup


def fit_and_save_norm_stats(
    raw_path: Path,
    cache_path: Path,
    stats_path: Path,
    family_fn: Callable[[str, str], str],
    train_families,
    verbose: bool = True,
) -> Path:
    """Fit proxy normalization stats on train-family architectures only and
    write them to stats_path. family_fn and train_families are passed in by the
    caller (keeps this module independent of AccPredictor)."""
    raw_path, cache_path, stats_path = Path(raw_path), Path(cache_path), Path(stats_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing: {cache_path} -- build the proxy cache first")

    family_lookup = _load_family_lookup(raw_path, family_fn)

    train_rows: list[dict] = []
    family_counts: dict[str, int] = {}
    missing_family = 0
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            family = family_lookup.get((rec.get("nn"), rec.get("dataset")))
            if family is None:
                missing_family += 1
                continue
            family_counts[family] = family_counts.get(family, 0) + 1
            if family in train_families:
                train_rows.append(rec)

    if verbose:
        print(f"[proxy-norm] Family distribution: {family_counts}")
        if missing_family:
            print(f"[proxy-norm] {missing_family} cached rows had no matching raw record (skipped)")
        print(f"[proxy-norm] Train-family architectures used to fit stats: {len(train_rows)}")

    if len(train_rows) < 10:
        raise RuntimeError(
            f"Only {len(train_rows)} train-family architectures available -- too few to fit "
            "reliable normalization stats. Ensure the proxy cache built to completion."
        )

    stats = fit_normalization_stats(train_rows, PROXY_NAMES)
    if verbose:
        for name, s in stats.items():
            print(f"[proxy-norm]   {name:<12} transform={s['transform']:<6} n={s['n']:<5} "
                  f"mean={s['mean']:.4f} std={s['std']:.4f}")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if verbose:
        print(f"[proxy-norm] Saved: {stats_path}")
    return stats_path


def main() -> None:
    from ab.gpt.act.AccPredictor import infer_arch_family, TRAIN_ARCH_FAMILIES
    fit_and_save_norm_stats(RAW_PATH, CACHE_PATH, STATS_PATH, infer_arch_family, TRAIN_ARCH_FAMILIES)


if __name__ == "__main__":
    main()

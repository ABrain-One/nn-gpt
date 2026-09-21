"""Stage 2: raw records -> ChatML, feature attachment, family split, dataset writing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from . import config
from .config import *  # noqa: F401,F403
from .common import _stream_jsonl, _write_jsonl, _active_system_prompt
from .features import (
    _compute_targets, _build_user_message, _arch_family, _build_reasoning_trace, infer_arch_family,
)
from ab.gpt.util.method.zero_cost_proxies import normalize_proxy_value, DEFAULT_PROXY_NORM_STATS

if TYPE_CHECKING:
    from ab.gpt.semantic_retrieval import SemanticIndex




def _record_to_chatml(record: dict) -> dict | None:
    required = {"task", "dataset", "metric", "nn", "nn_code", "best_accuracy", "epochs_to_best"}
    if not all(k in record for k in required):
        return None

    nn_code = record.get("nn_code", "") or ""
    if not isinstance(nn_code, str):
        return None

    targets = _compute_targets(record)
    answer_json = json.dumps(
        {
            "best_accuracy": round(targets["best_accuracy"], 2),
            "best_epoch": int(targets["best_epoch"]),
        },
        separators=(",", ":"),
    )
    if config.USE_CHAIN_OF_THOUGHT:
        assistant_content = _build_reasoning_trace(record, targets) + " " + answer_json
    else:
        assistant_content = answer_json
    return {
        "messages": [
            {"role": "system", "content": _active_system_prompt()},
            {"role": "user", "content": _build_user_message(record, nn_code)},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _split_by_architecture_family(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    family_to_records: dict[str, list[dict]] = {}
    for rec in records:
        raw = rec["raw"]
        nn_code = raw.get("nn_code", "") or ""
        family = _arch_family(raw, nn_code) if isinstance(nn_code, str) else "other"
        family_to_records.setdefault(family, []).append(rec)

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    for family, recs in family_to_records.items():
        if family in TRAIN_ARCH_FAMILIES:
            train_records.extend(recs)
        elif family in VAL_ARCH_FAMILIES:
            val_records.extend(recs)
        elif family in TEST_ARCH_FAMILIES:
            test_records.extend(recs)
    return train_records, val_records, test_records


def _load_proxy_cache(path: Path) -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return cache
    for rec in _stream_jsonl(path):
        cache[(rec.get("nn"), rec.get("dataset"))] = rec
    return cache


def _load_proxy_norm_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _attach_proxy_features(record: dict, proxy_cache: dict, norm_stats: dict) -> None:
    """Mutates record in place: adds normalized proxy_<name> keys, read by
    _format_proxy_lines via _build_user_message. Architectures with no
    cached proxies (~2.6%, see IMPROVEMENT_PLAN_Zero_Cost_Proxies.md Step 2)
    are left untouched -- _format_proxy_lines already handles the resulting
    all-missing case by omitting the block entirely."""
    proxies = proxy_cache.get((record.get("nn"), record.get("dataset")))
    if not proxies:
        return
    for name in config.ACTIVE_PROXY_NAMES:
        raw_value = proxies.get(name)
        if raw_value is not None:
            record[f"proxy_{name}"] = normalize_proxy_value(name, raw_value, norm_stats)


_PREDICTOR_PROXY_NORM_STATS: dict | None = None

# Embedded fallback so inference works without a proxy_norm_stats.json file; a
# local file, if present, overrides it (see _get_predictor_proxy_norm_stats).
_DEFAULT_PROXY_NORM_STATS = DEFAULT_PROXY_NORM_STATS


def _get_predictor_proxy_norm_stats() -> dict:
    """Load and cache the proxy normalization stats: the on-disk
    proxy_norm_stats.json if present, else the embedded defaults."""
    global _PREDICTOR_PROXY_NORM_STATS
    if _PREDICTOR_PROXY_NORM_STATS is None:
        stats = _load_proxy_norm_stats(PROXY_NORM_STATS_PATH)
        _PREDICTOR_PROXY_NORM_STATS = stats if stats else _DEFAULT_PROXY_NORM_STATS
    return _PREDICTOR_PROXY_NORM_STATS


def _attach_nn_stats(record: dict, cache: dict) -> None:
    """Mutates record in place: adds "nn_stat_flops" (the precomputed,
    train-only z-scored log-FLOPs) keyed by architecture name. Missing
    architectures are left untouched -- _format_nn_stats_lines omits the
    block when the value is absent."""
    value = cache.get(record.get("nn"))
    if value is not None:
        record["nn_stat_flops"] = value


def _attach_similar_architectures(record: dict, index: SemanticIndex, k: int) -> None:
    """Mutates record in place: adds a "similar_architectures" key, read by
    _format_similar_architectures_lines via _build_user_message. Reuses the
    query architecture's own precomputed embedding from the Step 2 cache
    (no fresh model inference here) and retrieves from the leakage-guarded
    train-only pool, excluding the query itself."""
    key = (record.get("nn"), record.get("dataset"))
    query_embedding = index.get_cached_embedding(*key)
    if query_embedding is None:
        return
    record["similar_architectures"] = index.retrieve(query_embedding, k=k, exclude_key=key)


def prepare_llm_datasets(
    input_path: Path = RAW_INPUT_PATH,
    output_dir: Path = ACC_DIR,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    proxy_cache: dict = {}
    proxy_norm_stats: dict = {}
    if config.USE_ZERO_COST_PROXIES:
        # Build the cache + stats on first run (cached to disk, resumable).
        if not PROXY_CACHE_PATH.exists():
            print("Proxy cache missing -- building it now...")
            from ab.gpt.util.method.compute_proxy_cache import build_proxy_cache
            build_proxy_cache(input_path=input_path, cache_path=PROXY_CACHE_PATH)
        if not PROXY_NORM_STATS_PATH.exists():
            print("Proxy normalization stats missing -- fitting them now (train-family only)...")
            from ab.gpt.util.method.fit_proxy_normalization import fit_and_save_norm_stats
            fit_and_save_norm_stats(
                raw_path=input_path,
                cache_path=PROXY_CACHE_PATH,
                stats_path=PROXY_NORM_STATS_PATH,
                family_fn=infer_arch_family,
                train_families=TRAIN_ARCH_FAMILIES,
            )
        proxy_cache = _load_proxy_cache(PROXY_CACHE_PATH)
        proxy_norm_stats = _load_proxy_norm_stats(PROXY_NORM_STATS_PATH)
        print(f"Loaded zero-cost proxy cache: {len(proxy_cache)} architectures")

    semantic_index = None
    if config.USE_SEMANTIC_RETRIEVAL:
        from ab.gpt.semantic_retrieval import SemanticIndex
        semantic_index = SemanticIndex.load()
        print(f"Loaded semantic embedding index: {len(semantic_index.all_meta)} architectures, "
              f"{len(semantic_index._train_keys)} in the training (retrievable) pool")

    nn_stats_cache: dict = {}
    if config.USE_NN_STATS:
        if not NN_STATS_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"USE_NN_STATS is True but {NN_STATS_CACHE_PATH} is missing. "
                "Run ab.gpt.build_nn_stat_cache first."
            )
        with open(NN_STATS_CACHE_PATH, encoding="utf-8") as fh:
            nn_stats_cache = json.load(fh)
        print(f"Loaded nn_stat FLOPs cache: {len(nn_stats_cache)} architectures")

    records: list[dict] = []
    skipped = 0
    for raw in tqdm(_stream_jsonl(input_path), desc="Processing"):
        if config.USE_ZERO_COST_PROXIES:
            _attach_proxy_features(raw, proxy_cache, proxy_norm_stats)
        if config.USE_NN_STATS:
            _attach_nn_stats(raw, nn_stats_cache)
        if config.USE_SEMANTIC_RETRIEVAL:
            _attach_similar_architectures(raw, semantic_index, SEMANTIC_RETRIEVAL_K)
        chatml = _record_to_chatml(raw)
        nn = raw.get("nn")
        if chatml is None or not raw.get("task") or not raw.get("dataset") or not nn:
            skipped += 1
            continue
        records.append({"nn": str(nn), "chatml": chatml, "raw": raw})

    if not records:
        raise ValueError(f"No valid records loaded from {input_path} (skipped {skipped})")

    train_data, val_data, test_data = _split_by_architecture_family(records)
    train_path = output_dir / "train_llm_dataset.jsonl"
    val_path = output_dir / "val_llm_dataset.jsonl"
    test_path = output_dir / "test_llm_dataset.jsonl"

    def _with_arch_id(rec: dict) -> dict:
        out = dict(rec["chatml"])
        out["architecture_id"] = rec["nn"]
        return out

    _write_jsonl(train_path, [_with_arch_id(r) for r in train_data])
    _write_jsonl(val_path, [_with_arch_id(r) for r in val_data])
    _write_jsonl(test_path, [_with_arch_id(r) for r in test_data])
    return train_path, val_path, test_path

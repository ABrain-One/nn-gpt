#!/usr/bin/env python3
"""Plot reward diagnostics for RL discovery runs."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


_PLACEHOLDER_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sW2KI4AAAAASUVORK5CYII="
)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _require_float(value: Any, *, field_name: str, line_no: int) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        raise ValueError(f"Line {line_no}: invalid {field_name} value {value!r}")
    return parsed


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _require_bool(value: Any, *, field_name: str, line_no: int) -> bool:
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError(f"Line {line_no}: invalid {field_name} value {value!r}")
    return parsed


def _float_or_nan(value: Optional[float]) -> float:
    return float(value) if value is not None else float("nan")


def _bool_or_nan(value: Optional[bool]) -> float:
    if value is None:
        return float("nan")
    return 1.0 if value else 0.0


def _mean_or_nan(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return float("nan")
    return float(statistics.fmean(filtered))


def _min_or_nan(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return float("nan")
    return float(min(filtered))


def _max_or_nan(values: Iterable[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return float("nan")
    return float(max(filtered))


def rolling_nanmean(values: Iterable[float], window: int) -> list[float]:
    series = list(values)
    if window <= 0:
        raise ValueError("window must be positive")
    rolled: list[float] = []
    for index in range(len(series)):
        start = max(0, index - window + 1)
        filtered = [float(value) for value in series[start : index + 1] if not math.isnan(float(value))]
        if not filtered:
            rolled.append(float("nan"))
        else:
            rolled.append(float(statistics.fmean(filtered)))
    return rolled


@dataclass
class RewardLogData:
    sample_index: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    reward_target_value: list[float] = field(default_factory=list)
    built_ok: list[float] = field(default_factory=list)
    forward_shape_ok: list[float] = field(default_factory=list)
    backward_ok: list[float] = field(default_factory=list)
    loss_drop_ok: list[float] = field(default_factory=list)
    timed_out: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    frozen_train_acc: list[float] = field(default_factory=list)
    frozen_test_acc: list[float] = field(default_factory=list)
    unfrozen_train_acc: list[float] = field(default_factory=list)
    unfrozen_test_acc: list[float] = field(default_factory=list)
    seed_accuracy_baseline: list[float] = field(default_factory=list)
    seed_train_acc_gap: list[float] = field(default_factory=list)
    group_baseline_train_acc: list[float] = field(default_factory=list)
    group_train_acc_gain: list[float] = field(default_factory=list)
    group_train_acc_improved: list[float] = field(default_factory=list)
    reward_batch_index: list[float] = field(default_factory=list)
    reward_group_id: list[int] = field(default_factory=list)
    group_warmup: list[float] = field(default_factory=list)
    loss_start: list[float] = field(default_factory=list)
    loss_end: list[float] = field(default_factory=list)
    loss_drop: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    estimated_total_seconds: list[float] = field(default_factory=list)
    eval_limit_seconds: list[float] = field(default_factory=list)
    warmup_dense_reward: list[float] = field(default_factory=list)
    anti_collapse_trainable_ok: list[float] = field(default_factory=list)
    xml_tag_exact: list[float] = field(default_factory=list)
    dual_backbone_ok: list[float] = field(default_factory=list)
    format_violation: list[float] = field(default_factory=list)
    cpu_prevalidate_failed: list[float] = field(default_factory=list)
    gpu_wait_timeout: list[float] = field(default_factory=list)
    progress_generation_total: list[float] = field(default_factory=list)
    progress_success_count: list[float] = field(default_factory=list)
    progress_success_rate: list[float] = field(default_factory=list)
    progress_warmup_trainable_count: list[float] = field(default_factory=list)
    progress_warmup_positive_count: list[float] = field(default_factory=list)
    progress_timeout_count: list[float] = field(default_factory=list)
    progress_improved_count: list[float] = field(default_factory=list)
    stage_name: list[str] = field(default_factory=list)
    stage_index: list[int] = field(default_factory=list)
    stage_uses_formal_eval: list[float] = field(default_factory=list)
    stage_uses_static_only: list[float] = field(default_factory=list)
    executable_candidate: list[float] = field(default_factory=list)
    discovery_candidate: list[float] = field(default_factory=list)
    formal_success_candidate: list[float] = field(default_factory=list)
    group_progress_by_group: dict[int, dict[str, Any]] = field(default_factory=dict)
    total_file_samples: int = 0
    current_run_sample_count: int = 0
    trimmed_stale_samples: int = 0

    @property
    def count(self) -> int:
        return len(self.reward)


def _parse_training_progress(log_dir: Path) -> dict[str, list[float]]:
    progress_path = log_dir / "training_progress.log"
    metrics = {
        "generation_total": [],
        "success_count": [],
        "success_rate": [],
        "warmup_trainable_count": [],
        "warmup_positive_count": [],
        "timeout_count": [],
        "improved_count": [],
    }
    if not progress_path.exists():
        return metrics

    pattern = re.compile(
        r"progress:\s*(?P<generation>\d+)\s+generation[，,]\s*"
        r"(?P<success>\d+)\s+success[，,]\s*success rate\s*(?P<success_rate>[0-9.]+)%"
        r".*?warmup_trainable_count=(?P<warmup_trainable>\d+)"
        r".*?warmup_positive_count=(?P<warmup_positive>\d+)"
        r".*?timeout_count=(?P<timeout>\d+)"
        r".*?improved_count=(?P<improved>\d+)"
    )
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            metrics["generation_total"].append(float(match.group("generation")))
            metrics["success_count"].append(float(match.group("success")))
            metrics["success_rate"].append(float(match.group("success_rate")) / 100.0)
            metrics["warmup_trainable_count"].append(float(match.group("warmup_trainable")))
            metrics["warmup_positive_count"].append(float(match.group("warmup_positive")))
            metrics["timeout_count"].append(float(match.group("timeout")))
            metrics["improved_count"].append(float(match.group("improved")))
    return metrics


def _load_group_progress(log_dir: Path) -> dict[int, dict[str, Any]]:
    progress_path = log_dir / "group_progress.jsonl"
    if not progress_path.exists():
        return {}

    progress_by_group: dict[int, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON in group_progress.jsonl: {exc}") from exc
            group_id = payload.get("group_id")
            if group_id is None:
                raise ValueError(f"Line {line_no}: group_progress row missing group_id")
            try:
                progress_by_group[int(group_id)] = payload
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {line_no}: invalid group_id {group_id!r}") from exc
    return progress_by_group


def _derive_forward_shape_ok(api_result: dict[str, Any], *, line_no: int) -> bool:
    if "forward_shape_ok" in api_result:
        return _require_bool(api_result["forward_shape_ok"], field_name="forward_shape_ok", line_no=line_no)
    if "forward_ok" in api_result:
        return _require_bool(api_result["forward_ok"], field_name="forward_ok", line_no=line_no)
    return False


def _derive_backward_ok(api_result: dict[str, Any], *, line_no: int) -> bool:
    if "backward_ok" in api_result:
        return _require_bool(api_result["backward_ok"], field_name="backward_ok", line_no=line_no)
    if "trained_step_ok" in api_result:
        return _require_bool(api_result["trained_step_ok"], field_name="trained_step_ok", line_no=line_no)
    return False


def _derive_loss_drop_ok(api_result: dict[str, Any], *, line_no: int) -> bool:
    if "loss_drop_ok" in api_result:
        return _require_bool(api_result["loss_drop_ok"], field_name="loss_drop_ok", line_no=line_no)
    loss_drop = _coerce_float(api_result.get("loss_drop"))
    if loss_drop is not None:
        return loss_drop > 0.0
    loss_start = _coerce_float(api_result.get("loss_start"))
    loss_end = _coerce_float(api_result.get("loss_end"))
    if loss_start is not None and loss_end is not None:
        return loss_end < loss_start
    return False


def _derive_format_violation(raw_extraction: Optional[dict[str, Any]]) -> float:
    if not isinstance(raw_extraction, dict):
        return float("nan")
    xml_tag_exact = _coerce_bool(raw_extraction.get("xml_tag_exact"))
    dual_backbone_ok = _coerce_bool(raw_extraction.get("dual_backbone_ok"))
    class_count = int(_coerce_float(raw_extraction.get("class_count")) or 0.0)
    import_count = int(_coerce_float(raw_extraction.get("import_count")) or 0.0)
    bad_signature_count = int(_coerce_float(raw_extraction.get("bad_signature_count")) or 0.0)
    violation = (
        xml_tag_exact is False
        or dual_backbone_ok is False
        or class_count > 0
        or import_count > 0
        or bad_signature_count > 0
    )
    return 1.0 if violation else 0.0


def _is_trainable_sample(data: RewardLogData, index: int) -> bool:
    return all(
        bool(series[index] == 1.0)
        for series in (data.built_ok, data.forward_shape_ok, data.backward_ok, data.loss_drop_ok)
    )


def load_reward_log(log_dir: Path | str) -> RewardLogData:
    resolved_log_dir = Path(log_dir).expanduser().resolve()
    sample_path = resolved_log_dir / "generation_samples.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing generation_samples.jsonl under {resolved_log_dir}")

    raw_records: list[dict[str, Any]] = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no}: reward row must be a JSON object")
            if "api_result" not in payload:
                raise ValueError(f"Line {line_no}: reward row missing api_result")
            if not isinstance(payload.get("api_result"), dict):
                raise ValueError(f"Line {line_no}: invalid api_result {payload.get('api_result')!r}")
            _require_float(payload.get("reward"), field_name="reward", line_no=line_no)
            payload["_line_no"] = line_no
            raw_records.append(payload)

    progress_metrics = _parse_training_progress(resolved_log_dir)
    group_progress_by_group = _load_group_progress(resolved_log_dir)
    total_file_samples = len(raw_records)
    current_run_target = None
    if progress_metrics["generation_total"]:
        latest_total = int(progress_metrics["generation_total"][-1])
        if latest_total > 0 and latest_total <= total_file_samples:
            current_run_target = latest_total
    if current_run_target is None:
        current_run_records = raw_records
    else:
        current_run_records = raw_records[-current_run_target:]

    data = RewardLogData(
        total_file_samples=total_file_samples,
        current_run_sample_count=len(current_run_records),
        trimmed_stale_samples=max(0, total_file_samples - len(current_run_records)),
        group_progress_by_group=group_progress_by_group,
    )

    data.progress_generation_total = list(progress_metrics["generation_total"])
    data.progress_success_count = list(progress_metrics["success_count"])
    data.progress_success_rate = list(progress_metrics["success_rate"])
    data.progress_warmup_trainable_count = list(progress_metrics["warmup_trainable_count"])
    data.progress_warmup_positive_count = list(progress_metrics["warmup_positive_count"])
    data.progress_timeout_count = list(progress_metrics["timeout_count"])
    data.progress_improved_count = list(progress_metrics["improved_count"])

    for sample_index, payload in enumerate(current_run_records, start=1):
        line_no = int(payload["_line_no"])
        api_result = dict(payload["api_result"])
        raw_extraction = api_result.get("raw_extraction")
        anti_collapse = api_result.get("anti_collapse")
        reward = _require_float(payload.get("reward"), field_name="reward", line_no=line_no)
        built_ok = _require_bool(api_result.get("built_ok"), field_name="built_ok", line_no=line_no)
        forward_shape_ok = _derive_forward_shape_ok(api_result, line_no=line_no)
        backward_ok = _derive_backward_ok(api_result, line_no=line_no)
        loss_drop_ok = _derive_loss_drop_ok(api_result, line_no=line_no)
        timed_out = (
            _require_bool(api_result["timed_out"], field_name="timed_out", line_no=line_no)
            if "timed_out" in api_result
            else False
        )
        seed_accuracy_baseline = _coerce_float(
            api_result.get("seed_accuracy_baseline", api_result.get("accuracy_baseline"))
        )
        reward_target_value = _coerce_float(
            api_result.get(
                "reward_target_value",
                api_result.get("frozen_test_acc", api_result.get("test_acc", api_result.get("val_metric"))),
            )
        )
        frozen_test_acc = _coerce_float(api_result.get("frozen_test_acc", api_result.get("test_acc", api_result.get("val_metric"))))
        frozen_train_acc = _coerce_float(api_result.get("frozen_train_acc", api_result.get("train_acc")))
        error_message = str(api_result.get("error") or "")
        error_stage = str(api_result.get("error_stage") or "")

        data.sample_index.append(sample_index)
        data.reward.append(reward)
        data.reward_target_value.append(_float_or_nan(reward_target_value))
        data.built_ok.append(_bool_or_nan(built_ok))
        data.forward_shape_ok.append(_bool_or_nan(forward_shape_ok))
        data.backward_ok.append(_bool_or_nan(backward_ok))
        data.loss_drop_ok.append(_bool_or_nan(loss_drop_ok))
        data.timed_out.append(_bool_or_nan(timed_out))
        data.train_acc.append(_float_or_nan(_coerce_float(api_result.get("train_acc"))))
        data.frozen_train_acc.append(_float_or_nan(frozen_train_acc))
        data.frozen_test_acc.append(_float_or_nan(frozen_test_acc))
        data.unfrozen_train_acc.append(_float_or_nan(_coerce_float(api_result.get("unfrozen_train_acc"))))
        data.unfrozen_test_acc.append(_float_or_nan(_coerce_float(api_result.get("unfrozen_test_acc"))))
        data.seed_accuracy_baseline.append(_float_or_nan(seed_accuracy_baseline))
        data.seed_train_acc_gap.append(_float_or_nan(_coerce_float(api_result.get("seed_train_acc_gap", api_result.get("train_acc_gain")))))
        data.group_baseline_train_acc.append(_float_or_nan(_coerce_float(api_result.get("group_baseline_train_acc"))))
        data.group_train_acc_gain.append(_float_or_nan(_coerce_float(api_result.get("group_train_acc_gain"))))
        data.group_train_acc_improved.append(_bool_or_nan(_coerce_bool(api_result.get("group_train_acc_improved"))))
        data.reward_batch_index.append(_float_or_nan(_coerce_float(api_result.get("reward_batch_index"))))
        data.reward_group_id.append(int(_coerce_float(api_result.get("reward_group_id")) or 0.0))
        data.group_warmup.append(_bool_or_nan(_coerce_bool(api_result.get("group_warmup"))))
        data.loss_start.append(_float_or_nan(_coerce_float(api_result.get("loss_start"))))
        data.loss_end.append(_float_or_nan(_coerce_float(api_result.get("loss_end"))))
        data.loss_drop.append(_float_or_nan(_coerce_float(api_result.get("loss_drop"))))
        data.val_metric.append(_float_or_nan(_coerce_float(api_result.get("val_metric"))))
        data.estimated_total_seconds.append(_float_or_nan(_coerce_float(api_result.get("estimated_total_seconds"))))
        data.eval_limit_seconds.append(_float_or_nan(_coerce_float(api_result.get("eval_limit_seconds"))))
        data.warmup_dense_reward.append(_float_or_nan(_coerce_float(api_result.get("warmup_dense_reward"))))
        data.stage_name.append(str(api_result.get("current_stage_name") or api_result.get("stage_name") or "unknown"))
        data.stage_index.append(int(_coerce_float(api_result.get("current_stage_index")) or 0.0))
        data.stage_uses_formal_eval.append(_bool_or_nan(_coerce_bool(api_result.get("stage_uses_formal_eval"))))
        data.stage_uses_static_only.append(_bool_or_nan(_coerce_bool(api_result.get("stage_uses_static_only"))))
        data.executable_candidate.append(_bool_or_nan(_coerce_bool(api_result.get("executable_candidate"))))
        data.discovery_candidate.append(_bool_or_nan(_coerce_bool(api_result.get("discovery_candidate"))))
        data.formal_success_candidate.append(_bool_or_nan(_coerce_bool(api_result.get("formal_success_candidate"))))
        data.anti_collapse_trainable_ok.append(
            _bool_or_nan(_coerce_bool(anti_collapse.get("trainable_ok"))) if isinstance(anti_collapse, dict) else float("nan")
        )
        data.xml_tag_exact.append(
            _bool_or_nan(_coerce_bool(raw_extraction.get("xml_tag_exact"))) if isinstance(raw_extraction, dict) else float("nan")
        )
        data.dual_backbone_ok.append(
            _bool_or_nan(_coerce_bool(raw_extraction.get("dual_backbone_ok"))) if isinstance(raw_extraction, dict) else float("nan")
        )
        data.format_violation.append(_derive_format_violation(raw_extraction))
        data.cpu_prevalidate_failed.append(1.0 if error_stage == "cpu_prevalidate" else 0.0)
        data.gpu_wait_timeout.append(
            1.0
            if ("waiting for available gpu reward worker" in error_message.lower() or "awaiting_gpu_headroom" in error_message.lower())
            else 0.0
        )

    return data


def compute_summary(data: RewardLogData, window: int = 20) -> dict[str, float]:
    del window
    trainable_mask = [_is_trainable_sample(data, idx) for idx in range(data.count)]
    warmup_indices = [idx for idx in range(data.count) if data.group_warmup[idx] == 1.0]
    warmup_trainable_indices = [idx for idx in warmup_indices if trainable_mask[idx]]

    return {
        "count": float(data.count),
        "reward_min": _min_or_nan(data.reward),
        "reward_max": _max_or_nan(data.reward),
        "reward_mean": _mean_or_nan(data.reward),
        "positive_reward_rate": _mean_or_nan([1.0 if value > 0.0 else 0.0 for value in data.reward]),
        "built_ok_rate": _mean_or_nan(data.built_ok),
        "forward_shape_ok_rate": _mean_or_nan(data.forward_shape_ok),
        "backward_ok_rate": _mean_or_nan(data.backward_ok),
        "loss_drop_ok_rate": _mean_or_nan(data.loss_drop_ok),
        "timed_out_rate": _mean_or_nan(data.timed_out),
        "train_acc_mean": _mean_or_nan(data.train_acc),
        "frozen_test_acc_mean": _mean_or_nan(data.frozen_test_acc),
        "reward_target_value_mean": _mean_or_nan(data.reward_target_value),
        "seed_accuracy_baseline_mean": _mean_or_nan(data.seed_accuracy_baseline),
        "group_train_acc_gain_mean": _mean_or_nan(data.group_train_acc_gain),
        "group_train_acc_improved_rate": _mean_or_nan(data.group_train_acc_improved),
        "warmup_dense_reward_mean": _mean_or_nan(data.warmup_dense_reward),
        "estimated_total_seconds_mean": _mean_or_nan(data.estimated_total_seconds),
        "warmup_trainable_rate": _mean_or_nan([1.0 if trainable_mask[idx] else 0.0 for idx in warmup_indices]),
        "warmup_positive_rate": _mean_or_nan([1.0 if data.reward[idx] > 0.0 else 0.0 for idx in warmup_trainable_indices]),
        "executable_candidate_rate": _mean_or_nan(data.executable_candidate),
        "discovery_candidate_rate": _mean_or_nan(data.discovery_candidate),
        "formal_success_candidate_rate": _mean_or_nan(data.formal_success_candidate),
        "anti_collapse_trainable_ok_rate": _mean_or_nan(data.anti_collapse_trainable_ok),
        "xml_tag_exact_rate": _mean_or_nan(data.xml_tag_exact),
        "dual_backbone_ok_rate": _mean_or_nan(data.dual_backbone_ok),
        "format_violation_rate": _mean_or_nan(data.format_violation),
        "cpu_prevalidate_failed_rate": _mean_or_nan(data.cpu_prevalidate_failed),
        "gpu_wait_timeout_rate": _mean_or_nan(data.gpu_wait_timeout),
        "progress_latest_total": data.progress_generation_total[-1] if data.progress_generation_total else float("nan"),
        "progress_latest_success_rate": data.progress_success_rate[-1] if data.progress_success_rate else float("nan"),
        "progress_latest_timeout_count": data.progress_timeout_count[-1] if data.progress_timeout_count else float("nan"),
        "progress_latest_improved_count": data.progress_improved_count[-1] if data.progress_improved_count else float("nan"),
        "trimmed_stale_samples": float(data.trimmed_stale_samples),
    }


def _group_means(data: RewardLogData, values: list[float]) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = {}
    for group_id, value in zip(data.reward_group_id, values):
        grouped.setdefault(int(group_id), []).append(value)
    cycles = sorted(grouped.keys())
    means = [_mean_or_nan(grouped[cycle]) for cycle in cycles]
    return cycles, means


def _cycle_metric_summary(group_ids: list[int], values: list[float]) -> dict[str, list[float] | list[int]]:
    grouped: dict[int, list[float]] = {}
    for group_id, value in zip(group_ids, values):
        grouped.setdefault(int(group_id), []).append(float(value))

    cycles = sorted(grouped.keys())
    avg_values: list[float] = []
    median_values: list[float] = []
    best_values: list[float] = []
    err_down: list[float] = []
    err_up: list[float] = []

    for cycle in cycles:
        valid_values = [value for value in grouped[cycle] if not math.isnan(float(value))]
        if not valid_values:
            avg_values.append(float("nan"))
            median_values.append(float("nan"))
            best_values.append(float("nan"))
            err_down.append(float("nan"))
            err_up.append(float("nan"))
            continue

        avg_value = float(statistics.fmean(valid_values))
        median_value = float(statistics.median(valid_values))
        best_value = float(max(valid_values))
        if len(valid_values) >= 2:
            sem = float(statistics.stdev(valid_values)) / math.sqrt(float(len(valid_values)))
            ci_half_width = 1.96 * sem
        else:
            ci_half_width = 0.0
        low_bound = max(0.0, avg_value - ci_half_width)
        high_bound = min(1.0, avg_value + ci_half_width)

        avg_values.append(avg_value)
        median_values.append(median_value)
        best_values.append(best_value)
        err_down.append(avg_value - low_bound)
        err_up.append(high_bound - avg_value)

    return {
        "cycles": cycles,
        "avg": avg_values,
        "median": median_values,
        "best": best_values,
        "err_down": err_down,
        "err_up": err_up,
    }


def _cycle_target_summary(data: RewardLogData) -> dict[str, list[float] | list[int]]:
    return _cycle_metric_summary(data.reward_group_id, data.reward_target_value)


def _group_progress_series(data: RewardLogData, key: str) -> tuple[list[int], list[float]]:
    cycles = sorted(int(group_id) for group_id in data.group_progress_by_group.keys())
    values = [
        _float_or_nan(_coerce_float(data.group_progress_by_group[cycle].get(key)))
        for cycle in cycles
    ]
    return cycles, values


def _cycle_xticks(cycles: list[int], *, max_ticks: int = 20) -> list[int]:
    if not cycles:
        return []
    if len(cycles) <= max_ticks:
        return list(cycles)
    step = max(1, math.ceil(len(cycles) / float(max_ticks)))
    ticks = [cycles[index] for index in range(0, len(cycles), step)]
    if ticks[-1] != cycles[-1]:
        ticks.append(cycles[-1])
    return sorted(set(int(cycle) for cycle in ticks))


def _valid_float_values(*series: Iterable[float]) -> list[float]:
    values: list[float] = []
    for current_series in series:
        for value in current_series:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(parsed) and not math.isinf(parsed):
                values.append(parsed)
    return values


def _unit_zoom_ylim(*series: Iterable[float], min_span: float = 0.08, pad_fraction: float = 0.12) -> tuple[float, float]:
    values = _valid_float_values(*series)
    if not values:
        return 0.0, 1.0

    lower = max(0.0, min(values))
    upper = min(1.0, max(values))
    center = (lower + upper) / 2.0
    span = max(float(min_span), upper - lower)
    padded_span = span * (1.0 + 2.0 * float(pad_fraction))
    y_min = max(0.0, center - padded_span / 2.0)
    y_max = min(1.0, center + padded_span / 2.0)

    if y_max - y_min < min_span:
        deficit = min_span - (y_max - y_min)
        y_min = max(0.0, y_min - deficit / 2.0)
        y_max = min(1.0, y_max + deficit / 2.0)
        if y_max - y_min < min_span:
            if y_min <= 0.0:
                y_max = min(1.0, y_min + min_span)
            elif y_max >= 1.0:
                y_min = max(0.0, y_max - min_span)
    return y_min, y_max


def _stage_segments(data: RewardLogData) -> list[tuple[int, int, str]]:
    if not data.stage_name:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 1
    current = data.stage_name[0]
    for sample_index, stage_name in zip(data.sample_index[1:], data.stage_name[1:]):
        if stage_name == current:
            continue
        segments.append((start, sample_index - 1, current))
        start = sample_index
        current = stage_name
    segments.append((start, data.sample_index[-1], current))
    return segments


def _stage_palette(stage_name: str) -> tuple[str, str]:
    palette = {
        "stage1_structure_explore": ("#FFF3E0", "#EF6C00"),
        "stage2_formal_explore": ("#E8F5E9", "#2E7D32"),
        "stage3_formal_optimize": ("#E3F2FD", "#1565C0"),
    }
    return palette.get(str(stage_name), ("#F5F5F5", "#616161"))


def _apply_stage_overlays(sample_axes: list[Any], data: RewardLogData) -> None:
    segments = _stage_segments(data)
    if not segments:
        for idx, is_warmup in enumerate(data.group_warmup):
            if is_warmup == 1.0:
                for ax in sample_axes:
                    ax.axvspan(idx + 0.5, idx + 1.5, color="#FFF3E0", alpha=0.18)
        return
    for start, end, stage_name in segments:
        fill_color, line_color = _stage_palette(stage_name)
        for ax in sample_axes:
            ax.axvspan(start - 0.5, end + 0.5, color=fill_color, alpha=0.18)
        if start > 1:
            for ax in sample_axes:
                ax.axvline(start - 0.5, color=line_color, linestyle="--", linewidth=1.2, alpha=0.9)


def compute_stage_summary(data: RewardLogData) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grouped_stage_groups: dict[str, int] = {}
    for payload in data.group_progress_by_group.values():
        stage_name = str(payload.get("stage_name") or "unknown")
        grouped_stage_groups[stage_name] = grouped_stage_groups.get(stage_name, 0) + 1
    for stage_name in sorted(set(data.stage_name) | set(grouped_stage_groups.keys())):
        indices = [idx for idx, current in enumerate(data.stage_name) if current == stage_name]
        summary[stage_name] = {
            "sample_count": len(indices),
            "group_count": int(grouped_stage_groups.get(stage_name, 0)),
            "reward_mean": _mean_or_nan([data.reward[idx] for idx in indices]),
            "reward_target_mean": _mean_or_nan([data.reward_target_value[idx] for idx in indices]),
            "executable_count": float(sum(1 for idx in indices if data.executable_candidate[idx] == 1.0)),
            "discovery_count": float(sum(1 for idx in indices if data.discovery_candidate[idx] == 1.0)),
            "formal_success_count": float(sum(1 for idx in indices if data.formal_success_candidate[idx] == 1.0)),
        }
    return summary


def _plot_dashboard(
    data: RewardLogData,
    summary: dict[str, float],
    *,
    output_path: Path,
    window: int,
    show_target: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=140)
    if not isinstance(axes, list):
        try:
            axes = list(axes.reshape(-1))
        except Exception:
            axes = [axes]

    x = data.sample_index
    stage1_only = bool(data.stage_name) and all(stage_name == "stage1_structure_explore" for stage_name in data.stage_name)
    reward_roll = rolling_nanmean(data.reward, window=window)
    target_roll = rolling_nanmean(data.reward_target_value, window=window)
    timeout_roll = [value * 100.0 for value in rolling_nanmean(data.timed_out, window=window)]
    trainable_roll = [
        value * 100.0
        for value in rolling_nanmean([1.0 if _is_trainable_sample(data, idx) else 0.0 for idx in range(data.count)], window=window)
    ]
    executable_roll = [value * 100.0 for value in rolling_nanmean(data.executable_candidate, window=window)]
    discovery_roll = [value * 100.0 for value in rolling_nanmean(data.discovery_candidate, window=window)]
    formal_roll = [value * 100.0 for value in rolling_nanmean(data.formal_success_candidate, window=window)]

    axes[0].plot(x, data.reward, color="#1565C0", linewidth=1.2, label="Reward")
    axes[0].plot(x, reward_roll, color="#EF6C00", linewidth=2.0, label=f"Reward MA{window}")
    axes[0].axhline(0.0, color="#616161", linewidth=1.0)
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(x, data.train_acc, color="#2E7D32", linewidth=1.2, label="Train Acc")
    axes[1].plot(x, data.frozen_test_acc, color="#6A1B9A", linewidth=1.2, label="Frozen Test Acc")
    if show_target:
        axes[1].plot(x, target_roll, color="#F57C00", linewidth=2.0, label=f"Reward Target MA{window}")
        axes[1].set_title("Stage1 Static Score / Reward Target" if stage1_only else "Accuracy / Reward Target")
        axes[1].set_ylabel("Score" if stage1_only else "Accuracy")
    else:
        axes[1].plot(x, data.unfrozen_test_acc, color="#1565C0", linewidth=1.2, label="Unfrozen Test Acc")
        axes[1].set_title("Actual Accuracy")
        axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylim(
        *_unit_zoom_ylim(
            data.train_acc,
            data.frozen_test_acc,
            target_roll if show_target else data.unfrozen_test_acc,
        )
    )
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(x, trainable_roll, color="#00897B", linewidth=2.0, label=f"Trainable Rate MA{window}")
    axes[2].plot(x, executable_roll, color="#1565C0", linewidth=1.8, label=f"Executable Rate MA{window}")
    axes[2].plot(x, discovery_roll, color="#EF6C00", linewidth=1.8, label=f"Discovery Rate MA{window}")
    axes[2].plot(x, formal_roll, color="#6A1B9A", linewidth=1.8, label=f"Formal Success Rate MA{window}")
    axes[2].plot(x, timeout_roll, color="#C62828", linewidth=2.0, label=f"Timeout Rate MA{window}")
    axes[2].set_title("Stability")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(loc="best", fontsize=8)

    if show_target:
        cycle_summary = _cycle_target_summary(data)
        cycles = [int(cycle) for cycle in cycle_summary["cycles"]]
        progress_cycles, closed_target = _group_progress_series(data, "closed_mean_reward_target_acc")
        if cycles:
            axes[3].errorbar(
                cycles,
                cycle_summary["avg"],
                yerr=[cycle_summary["err_down"], cycle_summary["err_up"]],
                fmt="none",
                ecolor="#FF9800",
                elinewidth=1.8,
                capsize=4,
                alpha=0.6,
                label="Cycle Avg Target 95% CI",
            )
            axes[3].scatter(
                cycles,
                cycle_summary["best"],
                marker="D",
                color="#E65100",
                s=60,
                edgecolors="white",
                linewidth=0.8,
                label="Cycle Best Target",
                zorder=5,
            )
            axes[3].scatter(
                cycles,
                cycle_summary["median"],
                marker="_",
                color="#D32F2F",
                s=180,
                linewidths=2.5,
                label="Cycle Median Target",
                zorder=6,
            )
            axes[3].scatter(
                cycles,
                cycle_summary["avg"],
                marker="o",
                color="#F57C00",
                s=40,
                edgecolors="black",
                linewidth=0.5,
                label="Cycle Avg Target",
                zorder=4,
            )
        if progress_cycles:
            axes[3].plot(progress_cycles, closed_target, color="#1565C0", linewidth=1.8, label="Closed Mean Target")
        cycle_axis = sorted(set(cycles) | set(progress_cycles))
        axes[3].set_title("Cycle Reward Target")
        axes[3].set_ylabel("Target Metric")
    else:
        cycle_summary = _cycle_metric_summary(data.reward_group_id, data.frozen_test_acc)
        cycles = [int(cycle) for cycle in cycle_summary["cycles"]]
        train_cycles, closed_train = _group_progress_series(data, "closed_mean_train_acc")
        test_cycles, closed_test = _group_progress_series(data, "closed_mean_test_acc")
        if cycles:
            axes[3].errorbar(
                cycles,
                cycle_summary["avg"],
                yerr=[cycle_summary["err_down"], cycle_summary["err_up"]],
                fmt="none",
                ecolor="#8E24AA",
                elinewidth=1.8,
                capsize=4,
                alpha=0.6,
                label="Cycle Avg Frozen Test 95% CI",
            )
            axes[3].scatter(
                cycles,
                cycle_summary["best"],
                marker="D",
                color="#6A1B9A",
                s=60,
                edgecolors="white",
                linewidth=0.8,
                label="Cycle Best Frozen Test",
                zorder=5,
            )
            axes[3].scatter(
                cycles,
                cycle_summary["median"],
                marker="_",
                color="#8E24AA",
                s=180,
                linewidths=2.5,
                label="Cycle Median Frozen Test",
                zorder=6,
            )
            axes[3].scatter(
                cycles,
                cycle_summary["avg"],
                marker="o",
                color="#AB47BC",
                s=40,
                edgecolors="black",
                linewidth=0.5,
                label="Cycle Avg Frozen Test",
                zorder=4,
            )
        if train_cycles:
            axes[3].plot(train_cycles, closed_train, color="#2E7D32", linewidth=1.8, label="Closed Mean Train")
        if test_cycles:
            axes[3].plot(test_cycles, closed_test, color="#1565C0", linewidth=1.8, label="Closed Mean Frozen Test")
        cycle_axis = sorted(set(cycles) | set(train_cycles) | set(test_cycles))
        axes[3].set_title("Cycle Actual Metrics")
        axes[3].set_ylabel("Accuracy")
    axes[3].set_xlabel("Cycle (Group ID)")
    if show_target:
        axes[3].set_ylim(
            *_unit_zoom_ylim(
                cycle_summary["avg"],
                cycle_summary["median"],
                cycle_summary["best"],
                closed_target,
            )
        )
    else:
        axes[3].set_ylim(
            *_unit_zoom_ylim(
                cycle_summary["avg"],
                cycle_summary["median"],
                cycle_summary["best"],
                closed_train,
                closed_test,
            )
        )
    if cycle_axis:
        axes[3].set_xticks(_cycle_xticks(cycle_axis))
        axes[3].set_xlim(min(cycle_axis) - 0.5, max(cycle_axis) + 0.5)
    axes[3].grid(True, linestyle="--", alpha=0.35)
    axes[3].legend(loc="best", fontsize=8)

    if data.progress_generation_total:
        axes[4].plot(data.progress_generation_total, data.progress_success_rate, color="#2E7D32", linewidth=2.0, label="Success Rate")
        axes[4].plot(data.progress_generation_total, data.progress_timeout_count, color="#C62828", linewidth=1.8, label="Timeout Count")
        axes[4].plot(data.progress_generation_total, data.progress_improved_count, color="#6A1B9A", linewidth=1.8, label="Improved Count")
    axes[4].set_title("Training Progress")
    axes[4].set_xlabel("Generation")
    axes[4].set_ylabel("Value")
    axes[4].grid(True, linestyle="--", alpha=0.35)
    axes[4].legend(loc="best", fontsize=8)

    axes[5].plot(x, data.cpu_prevalidate_failed, color="#5D4037", linewidth=1.8, label="CPU Prevalidate Fail")
    axes[5].plot(x, data.gpu_wait_timeout, color="#B71C1C", linewidth=1.8, label="GPU Wait Timeout")
    axes[5].plot(x, data.format_violation, color="#455A64", linewidth=1.8, label="Format Violation")
    axes[5].set_title("Failure Modes")
    axes[5].set_xlabel("Sample")
    axes[5].set_ylabel("Flag")
    axes[5].set_ylim(0, 1.05)
    axes[5].grid(True, linestyle="--", alpha=0.35)
    axes[5].legend(loc="best", fontsize=8)

    _apply_stage_overlays([axes[0], axes[1], axes[2], axes[5]], data)

    fig.suptitle("RL Reward Dashboard" if show_target else "RL Reward Dashboard (Actual Only)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)


def _write_stage_summary(path: Path, stage_summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stage_summary, handle, indent=2, ensure_ascii=False)


def _write_placeholder_png(path: Path) -> None:
    path.write_bytes(base64.b64decode(_PLACEHOLDER_PNG_BASE64))


def _print_summary(
    data: RewardLogData,
    summary: dict[str, float],
    *,
    log_dir: Path,
    output_path: Path,
    stage_summary: dict[str, Any],
) -> None:
    print(f"Log dir: {log_dir}")
    print(f"Samples: {int(summary['count'])}")
    print(
        "Reward:",
        json.dumps(
            {
                "min": summary["reward_min"],
                "max": summary["reward_max"],
                "mean": summary["reward_mean"],
                "positive_rate": summary["positive_reward_rate"],
            },
            ensure_ascii=False,
        ),
    )
    print(
        "Train accuracy:",
        json.dumps(
            {
                "mean": summary["train_acc_mean"],
                "frozen_test_mean": summary["frozen_test_acc_mean"],
                "reward_target_mean": summary["reward_target_value_mean"],
                "seed_baseline_mean": summary["seed_accuracy_baseline_mean"],
            },
            ensure_ascii=False,
        ),
    )
    print(
        "Training progress:",
        json.dumps(
            {
                "latest_total": summary["progress_latest_total"],
                "latest_success_rate": summary["progress_latest_success_rate"],
                "latest_timeout_count": summary["progress_latest_timeout_count"],
                "latest_improved_count": summary["progress_latest_improved_count"],
            },
            ensure_ascii=False,
        ),
    )
    print(
        "Failure modes:",
        json.dumps(
            {
                "cpu_prevalidate_failed_rate": summary["cpu_prevalidate_failed_rate"],
                "gpu_wait_timeout_rate": summary["gpu_wait_timeout_rate"],
                "format_violation_rate": summary["format_violation_rate"],
                "trimmed_stale_samples": summary["trimmed_stale_samples"],
            },
            ensure_ascii=False,
        ),
    )
    print("Stages:", json.dumps(stage_summary, ensure_ascii=False))
    print(f"Saved dashboard to: {output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="Directory containing reward logs")
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to <log-dir>/reward_dashboard_actual_only.png",
    )
    parser.add_argument(
        "--write-stage-overlay",
        action="store_true",
        help="Also write the target-overlay dashboard for debugging.",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir).expanduser().resolve()
    data = load_reward_log(log_dir)
    summary = compute_summary(data, window=max(1, int(args.window)))
    stage_summary = compute_stage_summary(data)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (log_dir / "reward_dashboard_actual_only.png")
    )
    stage_overlay_path = log_dir / "reward_dashboard_stage_overlay.png" if args.write_stage_overlay else None
    try:
        _plot_dashboard(
            data,
            summary,
            output_path=output_path,
            window=max(1, int(args.window)),
            show_target=False,
        )
        if stage_overlay_path is not None and stage_overlay_path != output_path:
            _plot_dashboard(data, summary, output_path=stage_overlay_path, window=max(1, int(args.window)))
    except ModuleNotFoundError as exc:
        if str(exc.name) != "matplotlib":
            raise
        _write_placeholder_png(output_path)
        if stage_overlay_path is not None and stage_overlay_path != output_path:
            _write_placeholder_png(stage_overlay_path)
        print("matplotlib not available; wrote placeholder PNG outputs instead")
    _write_stage_summary(log_dir / "stage_summary.json", stage_summary)
    _print_summary(data, summary, log_dir=log_dir, output_path=output_path, stage_summary=stage_summary)
    if stage_overlay_path is not None and stage_overlay_path != output_path:
        print(f"Saved stage-overlay dashboard to: {stage_overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

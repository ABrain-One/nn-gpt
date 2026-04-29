from typing import Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass, asdict, replace
import atexit
import ast
import csv
import gc
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import ab.gpt.rl_pipeline.reward_payload as RewardPayload
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


_NN_DATASET_IMPORT_READY = False


class PersistentEvalWorkerError(RuntimeError):
    pass


class EvalTimeException(RuntimeError):
    def __init__(
        self,
        *,
        estimated_total_seconds: float,
        eval_limit_seconds: float,
        phase: str,
        partial: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.estimated_total_seconds = float(estimated_total_seconds)
        self.eval_limit_seconds = float(eval_limit_seconds)
        self.phase = str(phase)
        self.partial = dict(partial or {})
        super().__init__(
            f"Estimated evaluation time {self.estimated_total_seconds:.1f}s exceeds "
            f"limit {self.eval_limit_seconds:.1f}s during {self.phase}"
        )


class FormalEvalTraceError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        hint: Optional[str] = None,
    ) -> None:
        self.stage = str(stage)
        self.error_type = str(error_type)
        self.error_message = str(error_message)
        self.context = dict(context or {})
        self.hint = None if hint is None else str(hint)
        super().__init__(self.error_message)


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_is_set(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw != ""


def _optional_positive_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC = "formal_multi_horizon_acc"
DEFAULT_FORMAL_REWARD_EPOCHS: tuple[int, ...] = (1, 5, 10)
FORMAL_REWARD_SCORE_HORIZONS: tuple[int, ...] = (1, 5, 10)


def _parse_formal_reward_epochs(raw: Optional[str]) -> list[int]:
    parsed_epochs: list[int] = []
    seen: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            epoch = int(token)
        except (TypeError, ValueError):
            continue
        if epoch <= 0 or epoch in seen:
            continue
        parsed_epochs.append(epoch)
        seen.add(epoch)
    if not parsed_epochs:
        return list(DEFAULT_FORMAL_REWARD_EPOCHS)
    parsed_epochs.sort()
    return parsed_epochs


def _resolve_formal_reward_epochs() -> list[int]:
    return _parse_formal_reward_epochs(os.environ.get("NNGPT_RL_FORMAL_REWARD_EPOCHS"))


def _visible_cuda_device_tokens() -> Optional[list[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw in {"", "-1"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _configured_cuda_device_tokens(env_name: str) -> Optional[list[str]]:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def get_distributed_runtime_info() -> Dict[str, Any]:
    visible_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    world_size = max(1, _safe_int_env("WORLD_SIZE", 1))
    rank = _safe_int_env("RANK", 0)
    raw_local_rank = _safe_int_env("LOCAL_RANK", 0)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = max(world_size, int(torch.distributed.get_world_size()))
        except Exception:
            pass
        try:
            rank = int(torch.distributed.get_rank())
        except Exception:
            pass

    if visible_gpu_count <= 0:
        local_rank = 0
    elif visible_gpu_count == 1:
        local_rank = 0
    elif 0 <= raw_local_rank < visible_gpu_count:
        local_rank = raw_local_rank
    else:
        local_rank = 0

    visible_gpu_tokens = _visible_cuda_device_tokens()
    if visible_gpu_tokens is None:
        visible_gpu_tokens = [str(index) for index in range(max(0, visible_gpu_count))]
    configured_train_gpu_tokens = _configured_cuda_device_tokens("NNGPT_TRAIN_GPU_TOKENS")
    configured_reward_gpu_tokens = _configured_cuda_device_tokens("NNGPT_REWARD_GPU_TOKENS")
    if configured_reward_gpu_tokens is None:
        configured_reward_gpu_tokens = _configured_cuda_device_tokens("NNGPT_AUX_GPU_TOKENS")

    train_gpu = local_rank if visible_gpu_count > 0 else None
    train_gpu_token = None
    if configured_train_gpu_tokens:
        configured_train_token = str(configured_train_gpu_tokens[0])
        if configured_train_token in visible_gpu_tokens:
            train_gpu = int(visible_gpu_tokens.index(configured_train_token))
            train_gpu_token = configured_train_token
    if train_gpu is not None and train_gpu_token is None:
        if visible_gpu_tokens and train_gpu < len(visible_gpu_tokens):
            train_gpu_token = str(visible_gpu_tokens[train_gpu])
        else:
            train_gpu_token = str(int(train_gpu))

    reward_gpu_tokens: list[str]
    if configured_reward_gpu_tokens is not None:
        reward_gpu_tokens = [
            str(token)
            for token in configured_reward_gpu_tokens
            if str(token) in visible_gpu_tokens
        ]
    elif world_size > 1:
        reward_gpu_tokens = [str(train_gpu_token)] if train_gpu_token is not None else []
    else:
        reward_gpu_tokens = [str(token) for token in visible_gpu_tokens]
    reward_gpu_indices = [
        int(visible_gpu_tokens.index(token))
        for token in reward_gpu_tokens
        if token in visible_gpu_tokens
    ]

    return {
        "distributed": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "raw_local_rank": raw_local_rank,
        "local_rank": local_rank,
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_tokens": list(visible_gpu_tokens or []),
        "train_gpu": train_gpu,
        "train_gpu_token": train_gpu_token,
        "reward_gpu_indices": list(reward_gpu_indices),
        "reward_gpu_tokens": list(reward_gpu_tokens),
    }


class EvalTimeBudget:
    def __init__(self, cfg: "EvalConfig") -> None:
        self.eval_limit_seconds = float(cfg.eval_limit_seconds)
        self.budget_probe_batches = max(1, int(cfg.budget_probe_batches))
        self.start_time = time.time()
        self.completed_units = 0
        self.observed_train_batches = 0
        self.estimated_total_seconds: Optional[float] = None
        self.total_units = 1

    def set_expected_work(
        self,
        *,
        train_batches: int,
        train_accuracy_batches: int,
        val_accuracy_batches: int,
    ) -> None:
        self.total_units = max(
            1,
            2
            + max(0, int(train_batches))
            + max(1, int(train_accuracy_batches))
            + max(1, int(val_accuracy_batches)),
        )

    def check(self, phase: str) -> None:
        self._maybe_raise_elapsed(phase)
        self._maybe_raise_estimate(phase)

    def mark_build_complete(self) -> None:
        self.completed_units += 1
        self.check("build")

    def mark_forward_complete(self) -> None:
        self.completed_units += 1
        self.check("forward")

    def mark_train_batch_start(self) -> None:
        self.check("train")

    def mark_train_batch_end(self) -> None:
        self.observed_train_batches += 1
        self.completed_units += 1
        self.check("train")

    def mark_accuracy_batch_start(self, phase: str) -> None:
        self.check(phase)

    def mark_accuracy_batch_end(self, phase: str) -> None:
        self.completed_units += 1
        self.check(phase)

    def _estimate_total_seconds(self) -> float:
        elapsed = max(1e-3, time.time() - self.start_time)
        completed_units = max(1, self.completed_units)
        estimated_total_seconds = elapsed * self.total_units / completed_units
        self.estimated_total_seconds = estimated_total_seconds
        return estimated_total_seconds

    def _maybe_raise_elapsed(self, phase: str) -> None:
        elapsed = max(1e-3, time.time() - self.start_time)
        if elapsed <= self.eval_limit_seconds:
            return
        estimated_total_seconds = max(elapsed, self._estimate_total_seconds())
        raise EvalTimeException(
            estimated_total_seconds=estimated_total_seconds,
            eval_limit_seconds=self.eval_limit_seconds,
            phase=phase,
        )

    def _maybe_raise_estimate(self, phase: str) -> None:
        if self.observed_train_batches < self.budget_probe_batches:
            return
        estimated_total_seconds = self._estimate_total_seconds()
        if estimated_total_seconds > self.eval_limit_seconds:
            raise EvalTimeException(
                estimated_total_seconds=estimated_total_seconds,
                eval_limit_seconds=self.eval_limit_seconds,
                phase=phase,
            )


def get_reward_worker_plan() -> Dict[str, Any]:
    runtime = get_distributed_runtime_info()
    visible_gpu_count = int(runtime["visible_gpu_count"])
    deepspeed_enabled = _safe_bool_env("NNGPT_SFT_USE_DEEPSPEED", False)
    reward_force_cpu = _safe_bool_env("NNGPT_REWARD_FORCE_CPU", False)
    train_gpu = runtime["train_gpu"]

    def _configured_reward_gpu_indices() -> list[int]:
        reward_gpu_indices = []
        for device_index in runtime.get("reward_gpu_indices", []) or []:
            try:
                parsed_index = int(device_index)
            except (TypeError, ValueError):
                continue
            if 0 <= parsed_index < visible_gpu_count:
                reward_gpu_indices.append(parsed_index)
        return reward_gpu_indices

    def _device_token(device_index: int) -> str:
        visible_gpu_tokens = list(runtime.get("visible_gpu_tokens") or [])
        if 0 <= device_index < len(visible_gpu_tokens):
            return str(visible_gpu_tokens[device_index])
        return str(int(device_index))

    def _cuda_device_memory_snapshot_gib(device_index: int) -> Dict[str, Any]:
        total_gib = None
        free_gib = None
        used_gib = None
        allocated_gib = None
        reserved_gib = None
        if torch.cuda.is_available():
            try:
                total_gib = float(torch.cuda.get_device_properties(device_index).total_memory) / float(1024 ** 3)
            except Exception:
                total_gib = None
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                free_gib = float(free_bytes) / float(1024 ** 3)
                used_gib = float(total_bytes - free_bytes) / float(1024 ** 3)
                if total_gib is None:
                    total_gib = float(total_bytes) / float(1024 ** 3)
            except Exception:
                pass
            try:
                allocated_gib = float(torch.cuda.memory_allocated(device_index)) / float(1024 ** 3)
            except Exception:
                allocated_gib = None
            try:
                reserved_gib = float(torch.cuda.memory_reserved(device_index)) / float(1024 ** 3)
            except Exception:
                reserved_gib = None
        return {
            "device_index": int(device_index),
            "device_token": _device_token(device_index),
            "total_gib": total_gib,
            "free_gib": free_gib,
            "used_gib": used_gib,
            "allocated_gib": allocated_gib,
            "reserved_gib": reserved_gib,
            "is_train_gpu": bool(train_gpu is not None and int(train_gpu) == int(device_index)),
        }

    def _ordered_reward_gpu_indices(
        reward_gpu_indices: list[int],
        gpu_memory_snapshots: list[Dict[str, Any]],
    ) -> list[int]:
        snapshot_by_index = {
            int(snapshot.get("device_index")): snapshot
            for snapshot in gpu_memory_snapshots
            if snapshot.get("device_index") is not None
        }

        def _sort_key(device_index: int) -> tuple[float, float, int, int]:
            snapshot = snapshot_by_index.get(int(device_index), {})
            free_gib = snapshot.get("free_gib")
            used_gib = snapshot.get("used_gib")
            is_train = int(device_index) == int(train_gpu) if train_gpu is not None else False
            return (
                1 if is_train else 0,
                -float(free_gib) if free_gib is not None else float("inf"),
                float(used_gib) if used_gib is not None else float("inf"),
                int(device_index),
            )

        return sorted((int(device_index) for device_index in reward_gpu_indices), key=_sort_key)

    def _expand_reward_gpu_assignments(
        per_gpu_worker_counts: list[int],
        gpu_priority_indices: Optional[list[int]] = None,
    ) -> tuple[list[int], list[str]]:
        reward_gpu_indices: list[int] = []
        reward_gpu_tokens: list[str] = []
        active_device_indices = [
            int(device_index)
            for device_index, worker_count in enumerate(per_gpu_worker_counts)
            if int(worker_count) > 0
        ]
        if gpu_priority_indices:
            prioritized_device_indices = [
                int(device_index)
                for device_index in gpu_priority_indices
                if 0 <= int(device_index) < len(per_gpu_worker_counts)
                and int(per_gpu_worker_counts[int(device_index)]) > 0
            ]
            remaining_device_indices = [
                device_index
                for device_index in active_device_indices
                if device_index not in set(prioritized_device_indices)
            ]
            active_device_indices = prioritized_device_indices + remaining_device_indices
        elif train_gpu is not None:
            train_device_index = int(train_gpu)
            active_device_indices = [
                device_index
                for device_index in active_device_indices
                if device_index != train_device_index
            ] + (
                [train_device_index]
                if train_device_index in set(
                    int(device_index)
                    for device_index, worker_count in enumerate(per_gpu_worker_counts)
                    if int(worker_count) > 0
                )
                else []
            )
        max_workers = max(per_gpu_worker_counts, default=0)
        for round_index in range(max_workers):
            for device_index in active_device_indices:
                worker_count = per_gpu_worker_counts[device_index]
                if round_index >= int(worker_count):
                    continue
                reward_gpu_indices.append(int(device_index))
                reward_gpu_tokens.append(_device_token(device_index))
        return reward_gpu_indices, reward_gpu_tokens

    def _cpu_reward_worker_plan(
        *,
        mode: str,
        reason: str,
        gpu_memory_snapshots: Optional[list[Dict[str, Any]]] = None,
        worker_budget_gib: Optional[float] = None,
        reserved_headroom_gib: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "reason": reason,
            "visible_gpu_count": visible_gpu_count,
            "train_gpu": train_gpu,
            "reward_gpu_indices": [],
            "reward_gpu_tokens": [],
            "per_gpu_worker_counts": [0] * max(0, visible_gpu_count),
            "shared_train_gpu": False,
            "pool_size": 1,
            "workers_per_gpu": 0,
            "deepspeed_enabled": deepspeed_enabled,
            "distributed": bool(runtime["distributed"]),
            "rank": int(runtime["rank"]),
            "local_rank": int(runtime["local_rank"]),
            "world_size": int(runtime["world_size"]),
            "dynamic_scaling": False,
            "gpu_memory_snapshots": list(gpu_memory_snapshots or []),
            "worker_budget_gib": worker_budget_gib,
            "reserved_headroom_gib": reserved_headroom_gib,
        }

    def _build_reward_worker_plan(
        *,
        mode: str,
        per_gpu_worker_counts: list[int],
        dynamic_scaling: bool,
        gpu_priority_indices: Optional[list[int]] = None,
        gpu_memory_snapshots: Optional[list[Dict[str, Any]]] = None,
        worker_budget_gib: Optional[float] = None,
        reserved_headroom_gib: Optional[float] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        reward_gpu_indices, reward_gpu_tokens = _expand_reward_gpu_assignments(
            per_gpu_worker_counts,
            gpu_priority_indices=gpu_priority_indices,
        )
        if not reward_gpu_indices:
            return _cpu_reward_worker_plan(
                mode=f"{mode}_cpu_fallback",
                reason=reason or "no_gpu_workers_selected",
                gpu_memory_snapshots=gpu_memory_snapshots,
            )
        return {
            "mode": mode,
            "reason": reason,
            "visible_gpu_count": visible_gpu_count,
            "train_gpu": train_gpu,
            "reward_gpu_indices": reward_gpu_indices,
            "reward_gpu_tokens": reward_gpu_tokens,
            "per_gpu_worker_counts": [int(count) for count in per_gpu_worker_counts],
            "shared_train_gpu": bool(
                train_gpu is not None and int(train_gpu) in set(int(index) for index in reward_gpu_indices)
            ),
            "pool_size": max(1, len(reward_gpu_indices)),
            "workers_per_gpu": max(per_gpu_worker_counts, default=0),
            "deepspeed_enabled": deepspeed_enabled,
            "distributed": bool(runtime["distributed"]),
            "rank": int(runtime["rank"]),
            "local_rank": int(runtime["local_rank"]),
            "world_size": int(runtime["world_size"]),
            "dynamic_scaling": bool(dynamic_scaling),
            "gpu_memory_snapshots": list(gpu_memory_snapshots or []),
            "worker_budget_gib": worker_budget_gib,
            "reserved_headroom_gib": reserved_headroom_gib,
        }

    def _select_best_snapshot_for_single_worker(
        gpu_memory_snapshots: list[Dict[str, Any]],
        *,
        reserved_headroom_gib: float,
        worker_budget_gib: float,
    ) -> Optional[Dict[str, Any]]:
        required_free_gib = float(reserved_headroom_gib) + float(worker_budget_gib)
        eligible = [
            snapshot
            for snapshot in gpu_memory_snapshots
            if snapshot.get("free_gib") is not None and float(snapshot.get("free_gib") or 0.0) >= required_free_gib
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda snapshot: float(snapshot.get("free_gib") or -1.0),
        )

    def _dynamic_reward_workers_for_gpu(
        snapshot: Dict[str, Any],
        *,
        reserved_headroom_gib: float,
        worker_budget_gib: float,
        min_workers_per_gpu: int,
        max_workers_per_gpu: int,
    ) -> int:
        if max_workers_per_gpu <= 0:
            return 0
        free_gib = snapshot.get("free_gib")
        if free_gib is None:
            return max(0, min(1, max_workers_per_gpu))
        available_gib = max(0.0, float(free_gib) - float(reserved_headroom_gib))
        worker_budget_gib = max(1.0, float(worker_budget_gib))
        worker_count = int(available_gib // worker_budget_gib)
        if worker_count > 0:
            worker_count = max(worker_count, int(min_workers_per_gpu))
        return max(0, min(int(max_workers_per_gpu), int(worker_count)))

    if reward_force_cpu:
        return _cpu_reward_worker_plan(mode="cpu_fallback", reason="reward_force_cpu")
    if visible_gpu_count <= 0:
        return _cpu_reward_worker_plan(mode="cpu_fallback", reason="no_visible_cuda_devices")
    configured_reward_gpu_indices = _configured_reward_gpu_indices()
    if not runtime["distributed"] and not configured_reward_gpu_indices:
        gpu_memory_snapshots = [
            _cuda_device_memory_snapshot_gib(0)
        ] if visible_gpu_count > 0 else []
        return _cpu_reward_worker_plan(
            mode="cpu_fallback",
            gpu_memory_snapshots=gpu_memory_snapshots,
            reason="no_dedicated_reward_gpu_available",
        )

    distributed_single_process_per_gpu = bool(runtime["distributed"])

    fixed_workers_override = _env_is_set("NNGPT_REWARD_WORKERS_PER_GPU")
    if fixed_workers_override:
        workers_per_gpu = max(1, _safe_int_env("NNGPT_REWARD_WORKERS_PER_GPU", 1))
        if distributed_single_process_per_gpu:
            workers_per_gpu = 1
        reward_gpu_indices = list(configured_reward_gpu_indices)
        gpu_memory_snapshots = [
            _cuda_device_memory_snapshot_gib(index)
            for index in reward_gpu_indices
        ]
        if not reward_gpu_indices:
            return _cpu_reward_worker_plan(
                mode="cpu_fallback",
                reason="configured_reward_gpu_unavailable",
                gpu_memory_snapshots=gpu_memory_snapshots,
            )
        gpu_priority_indices = _ordered_reward_gpu_indices(reward_gpu_indices, gpu_memory_snapshots)
        per_gpu_worker_counts = [0] * visible_gpu_count
        for reward_gpu_index in reward_gpu_indices:
            per_gpu_worker_counts[int(reward_gpu_index)] = workers_per_gpu
        mode = (
            "distributed_dedicated_fixed_pool"
            if runtime["distributed"]
            else "single_rank_dedicated_fixed_pool"
        )
        return _build_reward_worker_plan(
            mode=mode,
            per_gpu_worker_counts=per_gpu_worker_counts,
            dynamic_scaling=False,
            gpu_priority_indices=gpu_priority_indices,
            gpu_memory_snapshots=gpu_memory_snapshots,
            reason="fixed_workers_override",
        )

    dynamic_scaling = _safe_bool_env("NNGPT_REWARD_ENABLE_DYNAMIC_SCALING", True)
    if not dynamic_scaling:
        reward_gpu_indices = list(configured_reward_gpu_indices)
        gpu_memory_snapshots = [
            _cuda_device_memory_snapshot_gib(index)
            for index in reward_gpu_indices
        ]
        if not reward_gpu_indices:
            return _cpu_reward_worker_plan(
                mode="cpu_fallback",
                reason="configured_reward_gpu_unavailable",
                gpu_memory_snapshots=gpu_memory_snapshots,
            )
        gpu_priority_indices = _ordered_reward_gpu_indices(reward_gpu_indices, gpu_memory_snapshots)
        per_gpu_worker_counts = [0] * visible_gpu_count
        for reward_gpu_index in reward_gpu_indices:
            per_gpu_worker_counts[int(reward_gpu_index)] = 1
        mode = (
            "distributed_dedicated_static_pool"
            if runtime["distributed"]
            else "single_rank_dedicated_static_pool"
        )
        return _build_reward_worker_plan(
            mode=mode,
            per_gpu_worker_counts=per_gpu_worker_counts,
            dynamic_scaling=False,
            gpu_priority_indices=gpu_priority_indices,
            gpu_memory_snapshots=gpu_memory_snapshots,
            reason="dynamic_scaling_disabled",
        )

    reserved_headroom_default = 0.0
    worker_budget_default = 15.0
    reserved_headroom_gib = max(0.0, _safe_float_env("NNGPT_REWARD_RESERVED_HEADROOM_GIB", reserved_headroom_default))
    worker_budget_gib = max(1.0, _safe_float_env("NNGPT_REWARD_WORKER_BUDGET_GIB", worker_budget_default))
    default_min_workers = 1 if visible_gpu_count == 1 and not runtime["distributed"] else 0
    min_workers_per_gpu = max(0, _safe_int_env("NNGPT_REWARD_MIN_WORKERS_PER_GPU", default_min_workers))
    reward_gpu_indices = list(configured_reward_gpu_indices)
    gpu_memory_snapshots = [
        _cuda_device_memory_snapshot_gib(index)
        for index in reward_gpu_indices
    ]
    if not reward_gpu_indices:
        return _cpu_reward_worker_plan(
            mode="cpu_fallback",
            reason="configured_reward_gpu_unavailable",
            gpu_memory_snapshots=gpu_memory_snapshots,
            worker_budget_gib=worker_budget_gib,
            reserved_headroom_gib=reserved_headroom_gib,
        )
    gpu_priority_indices = _ordered_reward_gpu_indices(reward_gpu_indices, gpu_memory_snapshots)
    max_workers_per_gpu = max(
        min_workers_per_gpu,
        _safe_int_env(
            "NNGPT_REWARD_MAX_WORKERS_PER_GPU",
            1024,
        ),
    )
    if distributed_single_process_per_gpu:
        max_workers_per_gpu = 1
        min_workers_per_gpu = min(min_workers_per_gpu, max_workers_per_gpu)
    per_gpu_worker_counts = [0] * visible_gpu_count
    local_worker_counts = [
        _dynamic_reward_workers_for_gpu(
            snapshot,
            reserved_headroom_gib=reserved_headroom_gib,
            worker_budget_gib=worker_budget_gib,
            min_workers_per_gpu=min_workers_per_gpu,
            max_workers_per_gpu=max_workers_per_gpu,
        )
        for snapshot in gpu_memory_snapshots
    ]
    for reward_gpu_index, worker_count in zip(reward_gpu_indices, local_worker_counts):
        per_gpu_worker_counts[int(reward_gpu_index)] = int(worker_count)
    if sum(per_gpu_worker_counts) <= 0:
        return _cpu_reward_worker_plan(
            mode="reward_gpu_wait",
            reason="awaiting_gpu_headroom",
            gpu_memory_snapshots=gpu_memory_snapshots,
            worker_budget_gib=worker_budget_gib,
            reserved_headroom_gib=reserved_headroom_gib,
        )
    mode = (
        "distributed_dedicated_dynamic_pool"
        if runtime["distributed"]
        else "single_rank_dedicated_dynamic_pool"
    )

    return _build_reward_worker_plan(
        mode=mode,
        per_gpu_worker_counts=per_gpu_worker_counts,
        dynamic_scaling=True,
        gpu_priority_indices=gpu_priority_indices,
        gpu_memory_snapshots=gpu_memory_snapshots,
        worker_budget_gib=worker_budget_gib,
        reserved_headroom_gib=reserved_headroom_gib,
        reason="dynamic_memory_scaling",
    )


def _reward_worker_plan_signature(plan: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(plan.get("mode", "")),
        bool(plan.get("distributed", False)),
        int(plan.get("rank", 0)),
        int(plan.get("local_rank", 0)),
        int(plan.get("world_size", 1)),
        tuple(int(index) for index in plan.get("reward_gpu_indices", [])),
        tuple(str(token) for token in plan.get("reward_gpu_tokens", [])),
    )


def _clear_reward_cuda_state() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_fatal_cuda_worker_error(error: Optional[str]) -> bool:
    if not error:
        return False
    normalized = " ".join(str(error).split()).lower()
    fatal_patterns = (
        "device-side assert triggered",
        "cudaerrorassert",
        "an illegal memory access was encountered",
        "unspecified launch failure",
    )
    return any(pattern in normalized for pattern in fatal_patterns)


def _gpu_memory_for_log(device_index: Optional[int]) -> Dict[str, Any]:
    info = {
        "gpu_index": device_index,
        "gpu_name": "",
        "free_gib": None,
        "used_gib": None,
        "total_gib": None,
    }
    if device_index is None:
        return info

    try:
        normalized_index = int(device_index)
    except (TypeError, ValueError):
        return info

    gpu_rows = _query_nvidia_smi_rows("gpu", ["index", "memory.free", "memory.used", "memory.total", "name"])
    matched_row = next(
        (row for row in gpu_rows if str(row.get("index") or "").strip() == str(normalized_index)),
        None,
    )
    if matched_row is not None:
        try:
            info["free_gib"] = float(matched_row["memory.free"]) / 1024.0
            info["used_gib"] = float(matched_row["memory.used"]) / 1024.0
            info["total_gib"] = float(matched_row["memory.total"]) / 1024.0
        except (TypeError, ValueError):
            pass
        info["gpu_name"] = str(matched_row.get("name") or "")
        return info

    if not torch.cuda.is_available():
        return info
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(normalized_index)
        info["free_gib"] = float(free_bytes) / float(1024 ** 3)
        info["total_gib"] = float(total_bytes) / float(1024 ** 3)
        info["used_gib"] = float(total_bytes - free_bytes) / float(1024 ** 3)
    except Exception:
        pass
    try:
        info["gpu_name"] = str(torch.cuda.get_device_name(normalized_index))
    except Exception:
        pass
    return info


class _PersistentEvalWorkerSession:
    def __init__(
        self,
        *,
        assigned_gpu: Optional[int],
        assigned_cuda_visible_device: Optional[str],
        worker_slot: int,
    ) -> None:
        self._assigned_gpu = assigned_gpu
        self._assigned_cuda_visible_device = (
            None if assigned_cuda_visible_device is None else str(assigned_cuda_visible_device)
        )
        self._worker_slot = int(worker_slot)
        self._parent_conn, child_conn = mp.Pipe(duplex=True)
        child_fd = int(child_conn.fileno())
        os.set_inheritable(child_fd, True)
        child_env = os.environ.copy()
        if assigned_gpu is None:
            child_env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            child_env["CUDA_VISIBLE_DEVICES"] = (
                self._assigned_cuda_visible_device
                if self._assigned_cuda_visible_device is not None
                else str(int(assigned_gpu))
            )
            child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            sys.executable,
            "-u",
            "-m",
            "ab.gpt.util.reward_worker_bootstrap",
            "--conn-fd",
            str(child_fd),
            "--assigned-gpu",
            "" if assigned_gpu is None else str(int(assigned_gpu)),
            "--assigned-cuda-visible-device",
            "" if self._assigned_cuda_visible_device is None else self._assigned_cuda_visible_device,
        ]
        self._process = subprocess.Popen(
            command,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            close_fds=True,
            pass_fds=(child_fd,),
        )
        child_conn.close()
        self._worker_info = self._wait_for_ready(timeout=30.0)
        print(
            "[Reward Worker] Ready "
            f"slot={self._worker_slot} "
            f"pid={self._worker_info['pid']}, "
            f"assigned_gpu={self._worker_info['assigned_gpu']}, "
            f"assigned_cuda_visible_device={self._worker_info['assigned_cuda_visible_device']!r}, "
            f"worker_device={self._worker_info['worker_device']}, "
            f"cuda_visible_devices={self._worker_info['cuda_visible_devices']!r}, "
            f"cuda_available={self._worker_info['cuda_available']}, "
            f"cuda_device_count={self._worker_info['cuda_device_count']}, "
            f"rss_gib={self._worker_info['rss_gib']:.2f}, "
            f"torch_home={self._worker_info['torch_home']!r}, "
            f"physical_gpu_index={self._worker_info.get('physical_gpu_index')}, "
            f"physical_gpu_bus_id={self._worker_info.get('physical_gpu_bus_id', '')!r}, "
            f"physical_gpu_uuid={self._worker_info.get('physical_gpu_uuid', '')!r}, "
            f"physical_binding_verified={self._worker_info.get('physical_binding_verified', False)}"
        )

    def request(self, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        if self._process.poll() is not None:
            raise PersistentEvalWorkerError("Persistent eval worker exited before handling a request")
        try:
            self._parent_conn.send(payload)
        except Exception as exc:
            raise PersistentEvalWorkerError(f"Persistent eval worker send failed: {exc}") from exc

        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker timed out after {timeout:.0f} seconds")

        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker closed its pipe unexpectedly") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker returned a non-dict response")
        return response

    def _wait_for_ready(self, *, timeout: float) -> Dict[str, Any]:
        if self._process.poll() is not None:
            raise PersistentEvalWorkerError("Persistent eval worker exited during startup")
        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker did not send a startup handshake within {timeout:.0f} seconds"
            )
        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker closed its pipe during startup") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker startup recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker startup handshake was not a dict")
        if response.get("cmd") == "worker_init_error":
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker failed to initialize: {response.get('error', 'unknown error')}"
            )
        if response.get("cmd") != "worker_ready":
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker returned unexpected startup message: {response!r}"
            )

        required_keys = {
            "pid",
            "assigned_gpu",
            "assigned_cuda_visible_device",
            "worker_device",
            "cuda_visible_devices",
            "cuda_available",
            "cuda_device_count",
            "rss_gib",
            "torch_home",
            "physical_gpu_index",
            "physical_gpu_uuid",
            "physical_gpu_bus_id",
            "physical_gpu_name",
            "physical_binding_verified",
        }
        missing = sorted(required_keys.difference(response))
        if missing:
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker startup handshake missing fields: {', '.join(missing)}"
            )
        cuda_visible_devices = str(response["cuda_visible_devices"]).strip()
        cuda_device_count = int(response["cuda_device_count"])
        assigned_gpu = response["assigned_gpu"]
        assigned_cuda_visible_device = response["assigned_cuda_visible_device"]
        worker_device = str(response["worker_device"])
        if self._assigned_gpu is None:
            if worker_device != "cpu":
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"CPU reward worker returned unexpected device {worker_device!r}"
                )
            if cuda_visible_devices not in {"", "-1"} or cuda_device_count != 0:
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "CPU reward worker unexpectedly reported CUDA visibility "
                    f"(CUDA_VISIBLE_DEVICES={response['cuda_visible_devices']!r}, count={response['cuda_device_count']})"
                )
        else:
            if assigned_gpu != self._assigned_gpu:
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"Reward worker slot {self._worker_slot} bound unexpected GPU {assigned_gpu!r}, expected {self._assigned_gpu!r}"
                )
            if (
                self._assigned_cuda_visible_device is not None
                and cuda_visible_devices != self._assigned_cuda_visible_device
            ):
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "Reward worker slot "
                    f"{self._worker_slot} bound unexpected CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}, "
                    f"expected {self._assigned_cuda_visible_device!r}"
                )
            if worker_device != "cuda:0":
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"GPU reward worker returned unexpected local device {worker_device!r}"
                )
            if cuda_device_count != 1 or not bool(response["cuda_available"]):
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "GPU reward worker must expose exactly one visible CUDA device, but reported "
                    f"(available={response['cuda_available']}, count={response['cuda_device_count']})"
                )
            physical_gpu_index = response.get("physical_gpu_index")
            expected_visible_device = (
                None
                if self._assigned_cuda_visible_device is None
                else str(self._assigned_cuda_visible_device).strip()
            )
            if (
                expected_visible_device
                and expected_visible_device.isdigit()
                and bool(response.get("physical_binding_verified", False))
                and physical_gpu_index is not None
                and int(physical_gpu_index) != int(expected_visible_device)
            ):
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "Reward worker slot "
                    f"{self._worker_slot} resolved physical GPU {physical_gpu_index!r}, "
                    f"expected CUDA_VISIBLE_DEVICES token {expected_visible_device!r}"
                )
        return {
            "pid": int(response["pid"]),
            "assigned_gpu": assigned_gpu,
            "assigned_cuda_visible_device": (
                None if assigned_cuda_visible_device is None else str(assigned_cuda_visible_device)
            ),
            "worker_device": worker_device,
            "cuda_visible_devices": str(response["cuda_visible_devices"]),
            "cuda_available": bool(response["cuda_available"]),
            "cuda_device_count": int(response["cuda_device_count"]),
            "rss_gib": float(response["rss_gib"]),
            "torch_home": str(response["torch_home"]),
            "physical_gpu_index": response.get("physical_gpu_index"),
            "physical_gpu_uuid": str(response.get("physical_gpu_uuid") or ""),
            "physical_gpu_bus_id": str(response.get("physical_gpu_bus_id") or ""),
            "physical_gpu_name": str(response.get("physical_gpu_name") or ""),
            "physical_binding_verified": bool(response.get("physical_binding_verified", False)),
            "slot": self._worker_slot,
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {**self._worker_info, "alive": self._process.poll() is None}

    def close(self, *, force: bool = False) -> None:
        try:
            if not force and self._process.poll() is None:
                self._parent_conn.send({"cmd": "shutdown"})
        except Exception:
            pass

        if self._process.poll() is None:
            if force:
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)

        try:
            self._parent_conn.close()
        except Exception:
            pass


class _EvalWorkerPool:
    def __init__(self, *, plan: Optional[Dict[str, Any]] = None) -> None:
        self._plan = dict(plan or get_reward_worker_plan())
        self._sessions: list[_PersistentEvalWorkerSession] = []
        try:
            if self._plan["reward_gpu_indices"]:
                reward_gpu_tokens = list(self._plan.get("reward_gpu_tokens") or [])
                for slot, assigned_gpu in enumerate(self._plan["reward_gpu_indices"]):
                    assigned_cuda_visible_device = (
                        reward_gpu_tokens[slot]
                        if slot < len(reward_gpu_tokens)
                        else str(int(assigned_gpu))
                    )
                    self._sessions.append(
                        _PersistentEvalWorkerSession(
                            assigned_gpu=int(assigned_gpu),
                            assigned_cuda_visible_device=assigned_cuda_visible_device,
                            worker_slot=slot,
                        )
                    )
            else:
                self._sessions.append(
                    _PersistentEvalWorkerSession(
                        assigned_gpu=None,
                        assigned_cuda_visible_device=None,
                        worker_slot=0,
                    )
                )
        except Exception:
            for session in self._sessions:
                session.close(force=True)
            self._sessions = [
                _PersistentEvalWorkerSession(
                    assigned_gpu=None,
                    assigned_cuda_visible_device=None,
                    worker_slot=0,
                )
            ]
            self._plan = {
                **self._plan,
                "mode": "cpu_fallback",
                "reward_gpu_indices": [],
                "reward_gpu_tokens": [],
                "per_gpu_worker_counts": [0] * max(0, int(self._plan.get("visible_gpu_count", 0))),
                "shared_train_gpu": False,
                "pool_size": 1,
                "workers_per_gpu": 0,
            }
        print(
            "[Reward Worker Pool] "
            f"mode={self._plan['mode']} "
            f"visible_gpu_count={self._plan['visible_gpu_count']} "
            f"deepspeed_enabled={self._plan.get('deepspeed_enabled', False)} "
            f"rank={self._plan['rank']} "
            f"local_rank={self._plan['local_rank']} "
            f"world_size={self._plan['world_size']} "
            f"train_gpu={self._plan['train_gpu']} "
            f"workers_per_gpu={self._plan.get('workers_per_gpu', 1)} "
            f"per_gpu_worker_counts={self._plan.get('per_gpu_worker_counts', [])} "
            f"reward_gpu_indices={self._plan['reward_gpu_indices']} "
            f"reward_gpu_tokens={self._plan.get('reward_gpu_tokens', [])} "
            f"reason={self._plan.get('reason', '')!r}"
        )

    def worker_count(self) -> int:
        return len(self._sessions)

    def plan_signature(self) -> Tuple[Any, ...]:
        return _reward_worker_plan_signature(self._plan)

    def primary_device(self) -> str:
        if not self._sessions:
            return "cpu"
        return str(self._sessions[0].diagnostics().get("worker_device", "cpu"))

    def diagnostics(self) -> Dict[str, Any]:
        workers = [session.diagnostics() for session in self._sessions]
        total_rss_gib = sum(float(worker.get("rss_gib") or 0.0) for worker in workers)
        primary = workers[0] if workers else {}
        return {
            **primary,
            "mode": self._plan["mode"],
            "visible_gpu_count": self._plan["visible_gpu_count"],
            "distributed": bool(self._plan["distributed"]),
            "rank": self._plan["rank"],
            "local_rank": self._plan["local_rank"],
            "world_size": self._plan["world_size"],
            "train_gpu": self._plan["train_gpu"],
            "reward_gpu_indices": list(self._plan["reward_gpu_indices"]),
            "reward_gpu_tokens": list(self._plan.get("reward_gpu_tokens") or []),
            "shared_train_gpu": bool(self._plan["shared_train_gpu"]),
            "pool_size": len(workers),
            "workers_per_gpu": int(self._plan.get("workers_per_gpu", 1)),
            "per_gpu_worker_counts": list(self._plan.get("per_gpu_worker_counts") or []),
            "deepspeed_enabled": bool(self._plan.get("deepspeed_enabled", False)),
            "dynamic_scaling": bool(self._plan.get("dynamic_scaling", False)),
            "reason": str(self._plan.get("reason", "")),
            "gpu_memory_snapshots": list(self._plan.get("gpu_memory_snapshots") or []),
            "worker_budget_gib": self._plan.get("worker_budget_gib"),
            "reserved_headroom_gib": self._plan.get("reserved_headroom_gib"),
            "total_rss_gib": total_rss_gib,
            "workers": workers,
            "worker_pids": [worker["pid"] for worker in workers],
        }

    def _timeout_result_for_entry(
        self,
        entry: Dict[str, Any],
        *,
        error: str,
        estimated_total_seconds: float,
        assigned_gpu: Optional[int],
        worker_device: str,
        worker_slot: int,
    ) -> Dict[str, Any]:
        result = _timeout_eval_result(
            error=error,
            estimated_total_seconds=estimated_total_seconds,
            eval_limit_seconds=_effective_eval_limit_seconds(entry["effective_cfg"]),
            seed_accuracy_baseline=entry.get("seed_accuracy_baseline"),
            backbone_model_names=entry.get("backbone_model_names"),
        )
        result["assigned_gpu"] = assigned_gpu
        result["worker_device"] = worker_device
        result["worker_slot"] = worker_slot
        return result

    def _replace_session(self, slot: int) -> _PersistentEvalWorkerSession:
        old_session = self._sessions[slot]
        old_info = old_session.diagnostics()
        assigned_gpu = old_info.get("assigned_gpu")
        assigned_cuda_visible_device = old_info.get("assigned_cuda_visible_device")
        old_session.close(force=True)
        try:
            new_session = _PersistentEvalWorkerSession(
                assigned_gpu=assigned_gpu if assigned_gpu is not None else None,
                assigned_cuda_visible_device=assigned_cuda_visible_device,
                worker_slot=slot,
            )
        except Exception:
            new_session = _PersistentEvalWorkerSession(
                assigned_gpu=None,
                assigned_cuda_visible_device=None,
                worker_slot=slot,
            )
        self._sessions[slot] = new_session
        return new_session

    def _request_entry(self, slot: int, entry: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        session = self._sessions[slot]
        session_info = session.diagnostics()
        assigned_gpu = session_info.get("assigned_gpu")
        worker_device = str(session_info.get("worker_device", "cpu"))
        worker_slot = int(session_info.get("slot", slot))
        physical_gpu_index = session_info.get("physical_gpu_index")
        entry["payload"]["worker_slot"] = worker_slot
        reward_batch_index = entry["payload"].get("reward_batch_index")
        completion_index = entry["payload"].get("completion_index")
        reward_batch_size = max(1, int(entry["payload"].get("reward_batch_size", 1) or 1))
        reward_batch_position = max(1, int(entry["payload"].get("reward_batch_position", 1) or 1))
        gpu_log_index = physical_gpu_index if physical_gpu_index is not None else assigned_gpu
        gpu_memory = _gpu_memory_for_log(gpu_log_index)
        request_started_at = time.time()
        print(
            "[Reward Assign] "
            f"reward_batch_index={reward_batch_index} "
            f"batch_item={reward_batch_position}/{reward_batch_size} "
            f"completion_index={completion_index} "
            f"worker_slot={worker_slot} "
            f"assigned_gpu={assigned_gpu} "
            f"physical_gpu={physical_gpu_index} "
            f"worker_device={worker_device} "
            f"gpu_name={gpu_memory['gpu_name']!r} "
            f"gpu_free_gib={_format_mem_value(gpu_memory['free_gib'])} "
            f"gpu_used_gib={_format_mem_value(gpu_memory['used_gib'])} "
            f"gpu_total_gib={_format_mem_value(gpu_memory['total_gib'])} "
            f"timeout_seconds={timeout:.0f}"
        )
        try:
            result = session.request(entry["payload"], timeout=timeout)
        except PersistentEvalWorkerError as exc:
            elapsed_seconds = max(0.0, time.time() - request_started_at)
            print(
                "[Reward Assign] error "
                f"reward_batch_index={reward_batch_index} "
                f"batch_item={reward_batch_position}/{reward_batch_size} "
                f"completion_index={completion_index} "
                f"worker_slot={worker_slot} "
                f"assigned_gpu={assigned_gpu} "
                f"physical_gpu={physical_gpu_index} "
                f"worker_device={worker_device} "
                f"elapsed_seconds={elapsed_seconds:.2f} "
                f"error={type(exc).__name__}: {exc}"
            )
            self._replace_session(slot)
            return self._timeout_result_for_entry(
                entry,
                error=f"{type(exc).__name__}: {exc}",
                estimated_total_seconds=float(timeout),
                assigned_gpu=assigned_gpu,
                worker_device=worker_device,
                worker_slot=worker_slot,
            )
        result["assigned_gpu"] = assigned_gpu
        result["worker_device"] = worker_device
        result["worker_slot"] = worker_slot
        if bool(result.get("worker_restart_requested", False)):
            print(
                "[Reward Worker Pool] Restart "
                f"slot={worker_slot} "
                f"assigned_gpu={assigned_gpu} "
                f"worker_device={worker_device} "
                f"reason={result.get('error', 'unknown')!r}"
            )
            self._replace_session(slot)
        return result

    def request_entry(self, entry: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        entry["payload"]["worker_batch_first_item"] = True
        entry["payload"]["worker_batch_last_item"] = True
        entry["payload"]["reward_batch_size"] = 1
        entry["payload"]["reward_batch_position"] = 1
        return self._request_entry(0, entry, timeout=timeout)

    def map_entries(self, entries: list[Dict[str, Any]], *, timeout: float) -> list[Dict[str, Any]]:
        if not entries:
            return []
        if self.worker_count() <= 1:
            results = []
            batch_size = len(entries)
            for index, entry in enumerate(entries):
                entry["payload"]["worker_batch_first_item"] = index == 0
                entry["payload"]["worker_batch_last_item"] = index == (batch_size - 1)
                entry["payload"]["reward_batch_size"] = batch_size
                entry["payload"]["reward_batch_position"] = index + 1
                results.append(self._request_entry(0, entry, timeout=timeout))
            return results

        indexed_results: list[Optional[Dict[str, Any]]] = [None] * len(entries)
        assignments: list[list[tuple[int, Dict[str, Any]]]] = [[] for _ in self._sessions]
        for index, entry in enumerate(entries):
            entry["payload"]["reward_batch_size"] = len(entries)
            entry["payload"]["reward_batch_position"] = index + 1
            slot = index % self.worker_count()
            assignments[slot].append((index, entry))

        def _process_worker_tasks(slot: int, tasks: list[tuple[int, Dict[str, Any]]]) -> list[tuple[int, Dict[str, Any]]]:
            results: list[tuple[int, Dict[str, Any]]] = []
            for task_index, (index, entry) in enumerate(tasks):
                entry["payload"]["worker_batch_first_item"] = task_index == 0
                entry["payload"]["worker_batch_last_item"] = task_index == (len(tasks) - 1)
                results.append((index, self._request_entry(slot, entry, timeout=timeout)))
            return results

        non_empty_assignments = [(slot, tasks) for slot, tasks in enumerate(assignments) if tasks]
        with ThreadPoolExecutor(max_workers=len(non_empty_assignments)) as executor:
            futures = [
                executor.submit(_process_worker_tasks, slot, tasks)
                for slot, tasks in non_empty_assignments
            ]
            for future in futures:
                for index, result in future.result():
                    indexed_results[index] = result

        return [
            result
            if result is not None
            else self._timeout_result_for_entry(
                entries[index],
                error="PersistentEvalWorkerError: missing result from worker pool dispatch",
                estimated_total_seconds=timeout,
                assigned_gpu=None,
                worker_device="cpu",
                worker_slot=-1,
            )
            for index, result in enumerate(indexed_results)
        ]

    def close(self) -> None:
        for session in self._sessions:
            try:
                session.close()
            except Exception:
                pass
        self._sessions.clear()


_EVAL_WORKER_POOL: Optional[_EvalWorkerPool] = None


def _read_process_rss_gib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        return None
    return None


def _format_mem_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _extract_backbone_model_names_from_code(code: str) -> list[str]:
    if not code:
        return []
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            matches.setdefault(match.group(1), match.group(2))
    return [
        matches[attr_name]
        for attr_name in ("backbone_a", "backbone_b")
        if attr_name in matches
    ]


def _formal_eval_code_trace(code: str) -> Dict[str, Any]:
    text = str(code or "")
    return {
        "references_input_spec": "self._input_spec" in text,
        "assigns_input_spec": bool(re.search(r"self\._input_spec\s*=", text)),
        "references_pattern_attr": "self.pattern" in text,
        "line_count": len(text.splitlines()),
    }


def _ast_call_target_name(node: ast.Call) -> Optional[str]:
    func = getattr(node, "func", None)
    if isinstance(func, ast.Attribute):
        return str(func.attr)
    if isinstance(func, ast.Name):
        return str(func.id)
    return None


def _ast_is_self_method_call(node: ast.Call, method_name: str) -> bool:
    func = getattr(node, "func", None)
    if not isinstance(func, ast.Attribute):
        return False
    if str(getattr(func, "attr", "")) != str(method_name):
        return False
    owner = getattr(func, "value", None)
    return isinstance(owner, ast.Name) and str(getattr(owner, "id", "")) == "self"


def _cpu_prevalidation_failure_result(
    *,
    reward: float,
    error_message: str,
    error_type: Optional[str],
    error_context: Dict[str, Any],
    seed_accuracy_baseline: Optional[float],
    effective_cfg: "EvalConfig",
    backbone_model_names: Optional[list[str]],
) -> Dict[str, Any]:
    result = _empty_eval_result(
        reward=reward,
        error=error_message,
        seed_accuracy_baseline=seed_accuracy_baseline,
        eval_limit_seconds=_effective_eval_limit_seconds(effective_cfg),
        backbone_model_names=backbone_model_names,
    )
    _apply_error_trace(
        result,
        error_type=error_type,
        error_stage="cpu_prevalidate",
        error_context=error_context,
        error_hint=_infer_error_hint(error_message=error_message, context=error_context),
    )
    result["frozen_eval"] = _nested_eval_payload(
        result,
        eval_mode="frozen",
        backbone_frozen=True,
    )
    if bool(getattr(effective_cfg, "run_unfrozen_backbone_eval", False)):
        result["unfrozen_eval"] = _nested_eval_payload(
            result,
            eval_mode="unfrozen",
            backbone_frozen=False,
        )
    return result


def _cpu_smoke_prevalidate_reward_code(
    code: str,
    *,
    seed_accuracy_baseline: Optional[float],
    effective_cfg: "EvalConfig",
    backbone_model_names: Optional[list[str]],
    prm: Optional[Dict[str, Any]],
    code_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not bool(getattr(effective_cfg, "formal_nn_eval", False)):
        return None
    if not _safe_bool_env("NNGPT_RL_CPU_SMOKE_PREVALIDATE", True):
        return None

    input_shape = tuple(getattr(effective_cfg, "input_shape", (1, 3, 32, 32)))
    if len(input_shape) != 4:
        return None
    cpu_input_shape = (1, int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
    safe_prm = {
        "lr": 1e-2,
        "momentum": 0.9,
        "batch": max(1, min(2, int(getattr(effective_cfg, "default_batch_size", 2) or 2))),
        "epoch": 1,
        "transform": None,
        **dict(prm or {}),
    }
    safe_prm["batch"] = max(1, min(2, int(safe_prm.get("batch", 2) or 2)))
    safe_prm["epoch"] = 1

    model = None
    smoke_context = {
        "code_trace": code_trace,
        "cpu_smoke": True,
        "cpu_smoke_input_shape": cpu_input_shape,
    }
    original_cuda_is_available = getattr(torch.cuda, "is_available", None)
    original_cuda_device_count = getattr(torch.cuda, "device_count", None)
    try:
        if callable(original_cuda_is_available):
            torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
        if callable(original_cuda_device_count):
            torch.cuda.device_count = lambda: 0  # type: ignore[method-assign]
        builder = build_fn_from_code(
            code,
            cpu_input_shape,
            (int(getattr(effective_cfg, "n_classes", 10)),),
            safe_prm,
            "cpu",
        )
        model = builder()
        if not isinstance(model, nn.Module):
            raise RuntimeError("Net did not instantiate as torch.nn.Module")
        model.to("cpu")
        train_setup = getattr(model, "train_setup", None)
        if callable(train_setup):
            train_setup(safe_prm)
        forward_input = torch.randn(*cpu_input_shape, device="cpu")
        with torch.no_grad():
            output = model(forward_input)
        smoke_context["cpu_smoke_output_shape"] = tuple(output.shape) if hasattr(output, "shape") else None
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = f"{error_type}: {exc}"
        return _cpu_prevalidation_failure_result(
            reward=-3.0,
            error_message=error_message,
            error_type=error_type,
            error_context=smoke_context,
            seed_accuracy_baseline=seed_accuracy_baseline,
            effective_cfg=effective_cfg,
            backbone_model_names=backbone_model_names,
        )
    finally:
        if callable(original_cuda_is_available):
            torch.cuda.is_available = original_cuda_is_available  # type: ignore[method-assign]
        if callable(original_cuda_device_count):
            torch.cuda.device_count = original_cuda_device_count  # type: ignore[method-assign]
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
    return None


def _cpu_prevalidate_reward_code(
    code: str,
    *,
    seed_accuracy_baseline: Optional[float],
    effective_cfg: "EvalConfig",
    backbone_model_names: Optional[list[str]] = None,
    prm: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    text = str(code or "")
    code_trace = _formal_eval_code_trace(text)
    error_message: Optional[str] = None
    error_type = None

    try:
        tree = ast.parse(text or "")
    except SyntaxError as exc:
        error_type = "SyntaxError"
        error_message = f"SyntaxError: {exc}"
    else:
        infer_dimensions_calls = 0
        infer_dimensions_dynamically_calls = 0
        bad_dynamic_arg_count = None
        valid_self_dynamic_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target_name = _ast_call_target_name(node)
            if target_name == "infer_dimensions":
                infer_dimensions_calls += 1
            elif target_name == "infer_dimensions_dynamically":
                infer_dimensions_dynamically_calls += 1
                explicit_arg_count = len(getattr(node, "args", [])) + len(getattr(node, "keywords", []))
                if explicit_arg_count != 1 and bad_dynamic_arg_count is None:
                    bad_dynamic_arg_count = explicit_arg_count
                elif _ast_is_self_method_call(node, "infer_dimensions_dynamically"):
                    valid_self_dynamic_calls += 1

        if infer_dimensions_calls > 0:
            error_type = "AttributeError"
            error_message = "AttributeError: 'Net' object has no attribute 'infer_dimensions'"
        elif bad_dynamic_arg_count is not None:
            error_type = "TypeError"
            error_message = (
                "TypeError: Net.infer_dimensions_dynamically() takes 2 positional arguments "
                f"but {bad_dynamic_arg_count + 1} were given"
            )
        elif infer_dimensions_dynamically_calls <= 0 or valid_self_dynamic_calls <= 0:
            error_type = "RuntimeError"
            error_message = "RuntimeError: Net.__init__ must call self.infer_dimensions_dynamically(...)"
        elif not bool(code_trace.get("assigns_input_spec")):
            error_type = "AttributeError"
            error_message = "AttributeError: 'Net' object has no attribute '_input_spec'"

    if error_message is not None:
        return _cpu_prevalidation_failure_result(
            reward=-1.0,
            error_message=error_message,
            error_type=error_type,
            error_context={"code_trace": code_trace},
            seed_accuracy_baseline=seed_accuracy_baseline,
            effective_cfg=effective_cfg,
            backbone_model_names=backbone_model_names,
        )

    return _cpu_smoke_prevalidate_reward_code(
        text,
        seed_accuracy_baseline=seed_accuracy_baseline,
        effective_cfg=effective_cfg,
        backbone_model_names=backbone_model_names,
        prm=prm,
        code_trace=code_trace,
    )


def _infer_error_hint(
    *,
    error_message: Optional[str],
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    normalized = " ".join(str(error_message or "").split()).lower()
    trace_context = dict(context or {})
    code_trace = dict(trace_context.get("code_trace") or {})
    if "_input_spec" in normalized:
        if code_trace.get("references_input_spec") and not code_trace.get("assigns_input_spec"):
            return "generated Net references self._input_spec but never assigns it in __init__"
        return "generated Net is missing self._input_spec before downstream shape inference"
    if "must call self.infer_dimensions_dynamically" in normalized:
        return "generated Net never calls self.infer_dimensions_dynamically(...) from __init__ before classifier probing"
    if "unknown model" in normalized:
        return "one of the selected backbone names is not available in the TorchVision wrapper"
    if "accuracyexception" in normalized:
        return "nn-dataset accuracy floor rejected this candidate after formal eval"
    if "learntimeexception" in normalized:
        return "nn-dataset training-time guard rejected this candidate before completing formal eval"
    if "forward shape" in normalized:
        return "forward output shape is incompatible with the expected class dimension"
    return None


def _apply_error_trace(
    result: Dict[str, Any],
    *,
    error_type: Optional[str] = None,
    error_stage: Optional[str] = None,
    error_context: Optional[Dict[str, Any]] = None,
    error_hint: Optional[str] = None,
) -> Dict[str, Any]:
    if error_type is not None:
        result["error_type"] = str(error_type)
    if error_stage is not None:
        result["error_stage"] = str(error_stage)
    if error_context is not None:
        result["error_context"] = dict(error_context)
    if error_hint is not None:
        result["error_hint"] = str(error_hint)
    return result


def _base_eval_result(
    *,
    seed_accuracy_baseline: Optional[float] = None,
    eval_limit_seconds: Optional[int] = None,
    backbone_model_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return RewardPayload.base_eval_result(
        seed_accuracy_baseline=seed_accuracy_baseline,
        eval_limit_seconds=eval_limit_seconds,
        backbone_model_names=backbone_model_names,
        formal_reward_epochs=DEFAULT_FORMAL_REWARD_EPOCHS,
    )


def _nested_eval_payload(
    result: Optional[Dict[str, Any]],
    *,
    eval_mode: str,
    backbone_frozen: bool,
) -> Optional[Dict[str, Any]]:
    return RewardPayload.nested_eval_payload(
        result,
        eval_mode=eval_mode,
        backbone_frozen=backbone_frozen,
    )


def _merge_dual_eval_results(
    *,
    frozen_result: Dict[str, Any],
    unfrozen_result: Optional[Dict[str, Any]],
    cfg: "EvalConfig",
) -> Dict[str, Any]:
    return RewardPayload.merge_dual_eval_results(
        frozen_result=frozen_result,
        unfrozen_result=unfrozen_result,
        reward_target_metric=str(getattr(cfg, "reward_target_metric", "frozen_test_acc") or "frozen_test_acc"),
        formal_multi_horizon_reward_target_metric=FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC,
        coerce_optional_metric_float=_coerce_optional_metric_float,
    )

def _request_timeout_seconds(cfg: "EvalConfig") -> float:
    base_timeout = float(_effective_eval_limit_seconds(cfg))
    return max(360.0, base_timeout + 120.0)


def _is_cuda_oom_error_message(error_message: Optional[str]) -> bool:
    normalized = " ".join(str(error_message or "").split()).lower()
    oom_markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )
    return any(marker in normalized for marker in oom_markers)


def _halve_batch_size(batch_size: int, *, min_batch_size: int = 4) -> Optional[int]:
    current = max(1, int(batch_size))
    minimum = max(1, int(min_batch_size))
    if current <= minimum:
        return None
    next_batch = max(minimum, current // 2)
    if next_batch >= current:
        return None
    return int(next_batch)


def _replace_eval_batch_size(
    cfg: "EvalConfig",
    prm: Dict[str, Any],
    *,
    batch_size: int,
) -> tuple["EvalConfig", Dict[str, Any]]:
    resolved_batch_size = max(1, int(batch_size))
    next_cfg = replace(cfg, default_batch_size=resolved_batch_size)
    next_prm = dict(prm)
    next_prm["batch"] = resolved_batch_size
    return next_cfg, next_prm


def _adapt_formal_eval_inputs_for_worker(
    cfg: "EvalConfig",
    prm: Dict[str, Any],
) -> tuple["EvalConfig", Dict[str, Any]]:
    next_cfg = cfg
    next_prm = dict(prm)

    batch_override = _optional_positive_int_env("NNGPT_RL_FORMAL_BATCH_SIZE")
    if batch_override is not None:
        next_cfg, next_prm = _replace_eval_batch_size(
            next_cfg,
            next_prm,
            batch_size=batch_override,
        )

    num_workers_override = _optional_positive_int_env("NNGPT_RL_FORMAL_NUM_WORKERS")
    if num_workers_override is not None:
        next_prm["num_workers"] = int(num_workers_override)

    return next_cfg, next_prm


def _run_formal_eval_with_backoff(
    *,
    code: str,
    prm: Dict[str, Any],
    cfg: "EvalConfig",
    freeze_backbones: bool,
    seed_accuracy_baseline: Optional[float],
    backbone_model_names: list[str],
) -> tuple[Dict[str, Any], "EvalConfig", Dict[str, Any]]:
    active_cfg, active_prm = _adapt_formal_eval_inputs_for_worker(cfg, prm)
    attempt_batch_size = max(
        1,
        int(active_prm.get("batch", getattr(active_cfg, "default_batch_size", 32)) or 32),
    )
    min_batch_size = max(
        1,
        int(_optional_positive_int_env("NNGPT_RL_FORMAL_MIN_BATCH_SIZE") or 4),
    )

    while True:
        attempt_cfg, attempt_prm = _replace_eval_batch_size(
            active_cfg,
            active_prm,
            batch_size=attempt_batch_size,
        )
        result = _formal_eval_with_nn_dataset(
            code,
            prm=attempt_prm,
            cfg=attempt_cfg,
            freeze_backbones=freeze_backbones,
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )
        if not _is_cuda_oom_error_message(result.get("error")):
            return result, attempt_cfg, attempt_prm

        next_batch_size = _halve_batch_size(
            attempt_batch_size,
            min_batch_size=min_batch_size,
        )
        if next_batch_size is None:
            return result, attempt_cfg, attempt_prm

        print(
            "[Formal Eval Retry] CUDA OOM during formal eval, "
            f"retrying with batch {attempt_batch_size} -> {next_batch_size}",
            flush=True,
        )
        attempt_batch_size = next_batch_size


def _preview_eval_request(
    *,
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Optional[Dict[str, Any]],
    device: str,
    seed_accuracy_baseline: Optional[float],
    cfg: Optional["EvalConfig"],
) -> Dict[str, Any]:
    requested_device = str(device or "cpu")
    effective_cfg = _build_effective_eval_cfg(
        cfg=cfg,
        prm=prm,
        device=requested_device,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    backbone_model_names = _extract_backbone_model_names_from_code(code)
    prevalidated_result = _cpu_prevalidate_reward_code(
        code,
        seed_accuracy_baseline=seed_accuracy_baseline,
        effective_cfg=effective_cfg,
        backbone_model_names=backbone_model_names,
        prm=prm,
    )
    return {
        "requested_device": requested_device,
        "effective_cfg": effective_cfg,
        "request_timeout": _request_timeout_seconds(effective_cfg),
        "requires_gpu": bool(torch.cuda.is_available() and requested_device.startswith("cuda")),
        "backbone_model_names": backbone_model_names,
        "prevalidated_result": prevalidated_result,
    }

def compute_cv_reward_simple(
    *,
    built_ok: bool,
    forward_shape_ok: Optional[bool] = None,
    backward_ok: Optional[bool] = None,
    loss_drop_ok: bool = False,
    val_metric: Optional[float],
    val_metric_baseline: Optional[float] = None,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    flops_g: Optional[float] = None,
    critic_score: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    forward_ok: Optional[bool] = None,
    trained_step_ok: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Simple scalar reward for CV model generation.
    Returns a dict with 'reward' and component terms.
    """
    w = {
        "build": 0.2,          # bonus if model builds
        "forward_shape": 0.2,  # bonus if logits shape is correct
        "backward": 0.3,       # bonus if backward + optimizer step succeed
        "loss_drop": 0.6,      # bonus if short-run training shows learning signal
        "metric_gain": 0.0,    # validation stays logging-only in RL reward
        "eff_latency": 0.0,   # optional: -latency_ms * coeff
        "eff_params":  0.0,   # optional: -params_m  * coeff
        "eff_flops":   0.0,   # optional: -flops_g   * coeff
        "critic": 0.5,        # optional: critic score in [0,1]
        "kl": 0.02,           # optional: KL penalty coeff
        "clip_lo": -2.0,
        "clip_hi":  2.0,
    }
    if weights:
        w.update(weights)

    shape_ok = bool(forward_shape_ok if forward_shape_ok is not None else forward_ok)
    train_ok = bool(backward_ok if backward_ok is not None else trained_step_ok)

    r_build = w["build"] if built_ok else 0.0
    r_forward_shape = w["forward_shape"] if shape_ok else 0.0
    r_backward = w["backward"] if train_ok else 0.0
    r_loss_drop = w["loss_drop"] if loss_drop_ok else 0.0

    r_metric = 0.0
    if (val_metric is not None) and (val_metric_baseline is not None):
        r_metric = w["metric_gain"] * (val_metric - val_metric_baseline)

    r_eff = 0.0
    if latency_ms is not None and w["eff_latency"] != 0.0:
        r_eff += - w["eff_latency"] * float(latency_ms)
    if params_m is not None and w["eff_params"] != 0.0:
        r_eff += - w["eff_params"]  * float(params_m)
    if flops_g is not None and w["eff_flops"] != 0.0:
        r_eff += - w["eff_flops"]   * float(flops_g)

    r_critic = 0.0
    if critic_score is not None:
        critic_score = max(0.0, min(1.0, float(critic_score)))
        r_critic = w["critic"] * critic_score

    r_kl = 0.0
    if kl_div is not None and kl_div > 0.0:
        r_kl = - w["kl"] * float(kl_div)

    reward = r_build + r_forward_shape + r_backward + r_loss_drop + r_metric + r_eff + r_critic + r_kl
    reward = max(w["clip_lo"], min(w["clip_hi"], reward))

    return {
        "reward": reward,
        "r_build": r_build,
        "r_forward_shape": r_forward_shape,
        "r_backward": r_backward,
        "r_loss_drop": r_loss_drop,
        # Legacy aliases kept so older logging / callers do not break.
        "r_forward": r_forward_shape,
        "r_trainstep": r_backward,
        "r_metric": r_metric,
        "r_eff": r_eff,
        "r_critic": r_critic,
        "r_kl": r_kl,
    }

def _count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _set_dual_backbones_trainable(model: nn.Module, *, freeze_backbones: bool) -> None:
    for backbone_name in ("backbone_a", "backbone_b"):
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        for param in backbone.parameters():
            param.requires_grad = not freeze_backbones
        if freeze_backbones:
            backbone.eval()
        else:
            backbone.train()


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


@torch.no_grad()
def _quick_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cpu",
    n_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Try a single forward pass. Returns shape + latency diagnostics.
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    try:
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") and torch.cuda.is_available() else None
        dt = (time.time() - t0) * 1000.0
        output_shape = tuple(y.shape) if hasattr(y, "shape") else None
        shape_ok = bool(
            isinstance(y, torch.Tensor)
            and y.dim() == 2
            and y.shape[0] == input_shape[0]
            and (n_classes is None or y.shape[1] == n_classes)
        )
        return {
            "forward_ok": True,
            "forward_shape_ok": shape_ok,
            "latency_ms": dt,
            "output_shape": output_shape,
        }
    except Exception as exc:
        return {
            "forward_ok": False,
            "forward_shape_ok": False,
            "latency_ms": None,
            "output_shape": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _train_steps(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    n_classes: int = 10,
    time_budget: Optional[EvalTimeBudget] = None,
    freeze_backbones: bool = True,
) -> Dict[str, Any]:
    """
    Train for full epochs over the provided loader, optionally capped by max_steps.
    Returns training diagnostics.
    """
    model.train()
    _set_dual_backbones_trainable(model, freeze_backbones=freeze_backbones)
    loss_start = None
    loss_end = None
    steps_completed = 0
    epoch_loss_series: list[float] = []
    epochs_completed = 0
    try:
        trainable_params = _trainable_parameters(model)
        if not trainable_params:
            return {
                "backward_ok": False,
                "trained_step_ok": False,
                "loss_start": None,
                "loss_end": None,
                "loss_drop": None,
                "loss_drop_ok": False,
                "steps_completed": 0,
                "best_epoch_loss": None,
                "avg_epoch_loss": None,
                "epochs_completed": 0,
                "epoch_loss_series": [],
                "training_context_metric_name": "best_epoch_loss",
                "training_context_metric_value": None,
                "error": "RuntimeError: no trainable parameters remain after freezing backbones",
            }
        opt = torch.optim.SGD(trainable_params, lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        effective_epochs = max(1, int(epochs))
        effective_max_steps = None if max_steps is None or int(max_steps) <= 0 else int(max_steps)
        stop_early = False
        for _epoch_index in range(effective_epochs):
            epoch_loss_sum = 0.0
            epoch_step_count = 0
            for x, y in train_loader:
                if time_budget is not None:
                    try:
                        time_budget.mark_train_batch_start()
                    except EvalTimeException as exc:
                        exc.partial.update(
                            {
                                "backward_ok": False,
                                "trained_step_ok": False,
                                "loss_start": loss_start,
                                "loss_end": loss_end,
                                "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
                                "loss_drop_ok": False,
                                "steps_completed": steps_completed,
                                "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                                "avg_epoch_loss": (
                                    float(sum(epoch_loss_series) / len(epoch_loss_series))
                                    if epoch_loss_series
                                    else None
                                ),
                                "epochs_completed": epochs_completed,
                                "epoch_loss_series": list(epoch_loss_series),
                                "training_context_metric_name": "best_epoch_loss",
                                "training_context_metric_value": (
                                    min(epoch_loss_series) if epoch_loss_series else None
                                ),
                            }
                        )
                        raise
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
                    return {
                        "backward_ok": False,
                        "trained_step_ok": False,
                        "loss_start": None,
                        "loss_end": None,
                        "loss_drop": None,
                        "loss_drop_ok": False,
                        "steps_completed": steps_completed,
                        "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                        "avg_epoch_loss": (
                            float(sum(epoch_loss_series) / len(epoch_loss_series))
                            if epoch_loss_series
                            else None
                        ),
                        "epochs_completed": epochs_completed,
                        "epoch_loss_series": list(epoch_loss_series),
                        "training_context_metric_name": "best_epoch_loss",
                        "training_context_metric_value": min(epoch_loss_series) if epoch_loss_series else None,
                        "error": f"RuntimeError: logits must be (N, C), got {tuple(logits.shape) if hasattr(logits, 'shape') else type(logits)}",
                    }
                if logits.shape[0] != y.shape[0] or logits.shape[1] != n_classes:
                    return {
                        "backward_ok": False,
                        "trained_step_ok": False,
                        "loss_start": None,
                        "loss_end": None,
                        "loss_drop": None,
                        "loss_drop_ok": False,
                        "steps_completed": steps_completed,
                        "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                        "avg_epoch_loss": (
                            float(sum(epoch_loss_series) / len(epoch_loss_series))
                            if epoch_loss_series
                            else None
                        ),
                        "epochs_completed": epochs_completed,
                        "epoch_loss_series": list(epoch_loss_series),
                        "training_context_metric_name": "best_epoch_loss",
                        "training_context_metric_value": min(epoch_loss_series) if epoch_loss_series else None,
                        "error": f"RuntimeError: logits shape {tuple(logits.shape)} incompatible with labels {tuple(y.shape)} / classes {n_classes}",
                    }
                loss = criterion(logits, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    return {
                        "backward_ok": False,
                        "trained_step_ok": False,
                        "loss_start": None,
                        "loss_end": None,
                        "loss_drop": None,
                        "loss_drop_ok": False,
                        "steps_completed": steps_completed,
                        "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                        "avg_epoch_loss": (
                            float(sum(epoch_loss_series) / len(epoch_loss_series))
                            if epoch_loss_series
                            else None
                        ),
                        "epochs_completed": epochs_completed,
                        "epoch_loss_series": list(epoch_loss_series),
                        "training_context_metric_name": "best_epoch_loss",
                        "training_context_metric_value": min(epoch_loss_series) if epoch_loss_series else None,
                        "error": "RuntimeError: loss is NaN or Inf",
                    }
                loss_value = float(loss.detach().item())
                if loss_start is None:
                    loss_start = loss_value
                epoch_loss_sum += loss_value
                epoch_step_count += 1
                loss.backward()
                opt.step()
                loss_end = loss_value
                steps_completed += 1
                if time_budget is not None:
                    try:
                        time_budget.mark_train_batch_end()
                    except EvalTimeException as exc:
                        exc.partial.update(
                            {
                                "backward_ok": False,
                                "trained_step_ok": False,
                                "loss_start": loss_start,
                                "loss_end": loss_end,
                                "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
                                "loss_drop_ok": False,
                                "steps_completed": steps_completed,
                                "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                                "avg_epoch_loss": (
                                    float(sum(epoch_loss_series) / len(epoch_loss_series))
                                    if epoch_loss_series
                                    else None
                                ),
                                "epochs_completed": epochs_completed,
                                "epoch_loss_series": list(epoch_loss_series),
                                "training_context_metric_name": "best_epoch_loss",
                                "training_context_metric_value": (
                                    min(epoch_loss_series) if epoch_loss_series else None
                                ),
                            }
                        )
                        raise
                if effective_max_steps is not None and steps_completed >= effective_max_steps:
                    stop_early = True
                    break
            if epoch_step_count > 0:
                epoch_loss_series.append(float(epoch_loss_sum / float(epoch_step_count)))
                epochs_completed += 1
            if stop_early:
                break
        if steps_completed == 0 or loss_start is None or loss_end is None:
            return {
                "backward_ok": False,
                "trained_step_ok": False,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "loss_drop": None,
                "loss_drop_ok": False,
                "steps_completed": steps_completed,
                "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
                "avg_epoch_loss": (
                    float(sum(epoch_loss_series) / len(epoch_loss_series))
                    if epoch_loss_series
                    else None
                ),
                "epochs_completed": epochs_completed,
                "epoch_loss_series": list(epoch_loss_series),
                "training_context_metric_name": "best_epoch_loss",
                "training_context_metric_value": min(epoch_loss_series) if epoch_loss_series else None,
                "error": "RuntimeError: no training steps completed",
            }
        loss_drop = float(loss_start - loss_end)
        rel_drop_ok = loss_start > 0.0 and (loss_end <= loss_start * 0.98)
        loss_drop_ok = bool(loss_end < (loss_start - 1e-3) or rel_drop_ok)
        best_epoch_loss = min(epoch_loss_series) if epoch_loss_series else loss_end
        avg_epoch_loss = (
            float(sum(epoch_loss_series) / len(epoch_loss_series))
            if epoch_loss_series
            else loss_end
        )
        return {
            "backward_ok": True,
            "trained_step_ok": True,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": loss_drop,
            "loss_drop_ok": loss_drop_ok,
            "steps_completed": steps_completed,
            "best_epoch_loss": best_epoch_loss,
            "avg_epoch_loss": avg_epoch_loss,
            "epochs_completed": epochs_completed,
            "epoch_loss_series": list(epoch_loss_series),
            "training_context_metric_name": "best_epoch_loss",
            "training_context_metric_value": best_epoch_loss,
        }
    except EvalTimeException:
        raise
    except Exception as exc:
        return {
            "backward_ok": False,
            "trained_step_ok": False,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
            "loss_drop_ok": False,
            "steps_completed": steps_completed,
            "best_epoch_loss": min(epoch_loss_series) if epoch_loss_series else None,
            "avg_epoch_loss": (
                float(sum(epoch_loss_series) / len(epoch_loss_series))
                if epoch_loss_series
                else None
            ),
            "epochs_completed": epochs_completed,
            "epoch_loss_series": list(epoch_loss_series),
            "training_context_metric_name": "best_epoch_loss",
            "training_context_metric_value": min(epoch_loss_series) if epoch_loss_series else None,
            "error": f"{type(exc).__name__}: {exc}",
        }


@torch.no_grad()
def _quick_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str = "cpu",
    max_batches: Optional[int] = 10,
    time_budget: Optional[EvalTimeBudget] = None,
    phase: str = "accuracy",
) -> float:
    """
    Evaluate accuracy in [0,1] on the provided loader without interpolation or fallback.
    """
    model.eval()
    correct = 0
    total = 0
    batches_seen = 0

    for x, y in data_loader:
        if time_budget is not None:
            try:
                time_budget.mark_accuracy_batch_start(phase)
            except EvalTimeException as exc:
                exc.partial.update(
                    {
                        "batches_seen": batches_seen,
                        "partial_accuracy": (correct / total) if total > 0 else None,
                    }
                )
                raise
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if logits.dim() != 2:
            raise RuntimeError(f"logits must be (N,C), got {tuple(logits.shape)}")
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        batches_seen += 1
        if time_budget is not None:
            try:
                time_budget.mark_accuracy_batch_end(phase)
            except EvalTimeException as exc:
                exc.partial.update(
                    {
                        "batches_seen": batches_seen,
                        "partial_accuracy": (correct / total) if total > 0 else None,
                    }
                )
                raise
        if max_batches is not None and int(max_batches) > 0 and batches_seen >= int(max_batches):
            break

    return (correct / total) if total > 0 else 0.0


def _build_cifar10_loaders(cfg: "EvalConfig") -> Tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    height = int(cfg.input_shape[2])
    width = int(cfg.input_shape[3])
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=cfg.download,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=False,
        download=cfg.download,
        transform=val_transform,
    )

    if 0 < cfg.train_subset_size < len(train_dataset):
        train_dataset = Subset(train_dataset, range(cfg.train_subset_size))
    if (not bool(getattr(cfg, "full_test_acc", False))) and 0 < cfg.val_subset_size < len(val_dataset):
        val_dataset = Subset(val_dataset, range(cfg.val_subset_size))

    train_batch = max(1, min(cfg.default_batch_size, len(train_dataset)))
    val_batch = max(1, min(cfg.default_batch_size, len(val_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    return train_loader, val_loader


# ------------------------------------
# Public API: evaluate & produce reward
# ------------------------------------

@dataclass
class EvalConfig:
    device: str = "cpu"
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32)  # (B,C,H,W)
    n_classes: int = 10
    train_epochs: int = 1
    train_steps: Optional[int] = None
    max_val_batches: int = 2
    default_batch_size: int = 32
    train_subset_size: int = 256
    val_subset_size: int = 128
    data_root: str = "data_v2"
    download: bool = True
    # Optional efficiency logging
    measure_latency: bool = True
    # Optional PPO/critic
    kl_div: Optional[float] = None
    critic_fn: Optional[Callable[[nn.Module, Dict[str, Any]], float]] = None
    # Optional reward weights
    weights: Optional[Dict[str, float]] = None
    eval_limit_seconds: int = 270
    budget_probe_batches: int = 2
    run_unfrozen_backbone_eval: bool = False
    full_test_acc: bool = False
    reward_target_metric: str = "frozen_test_acc"
    formal_nn_eval: bool = False
    static_only: bool = False
    formal_task: str = "img-classification"
    formal_dataset: str = "cifar-10"
    formal_metric: str = "acc"
    formal_epoch_limit_minutes: Optional[float] = None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _normalize_optional_steps(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _build_effective_eval_cfg(
    *,
    cfg: Optional["EvalConfig"],
    prm: Optional[Dict[str, Any]],
    device: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
) -> "EvalConfig":
    prm_payload = dict(prm or {})
    base_cfg = cfg if cfg is not None else EvalConfig(
        device=device,
        input_shape=in_shape,
        n_classes=int(out_shape[0]),
        measure_latency=True,
        kl_div=None,
        critic_fn=None,
        weights=None,
    )
    base_reward_target_metric = str(getattr(base_cfg, "reward_target_metric", "frozen_test_acc") or "frozen_test_acc")
    return replace(
        base_cfg,
        device=device,
        input_shape=tuple(base_cfg.input_shape if cfg is not None else in_shape),
        n_classes=int(base_cfg.n_classes if cfg is not None else out_shape[0]),
        train_epochs=_coerce_positive_int(
            prm_payload.get("epoch"),
            getattr(base_cfg, "train_epochs", 1),
        ),
        train_steps=_normalize_optional_steps(getattr(base_cfg, "train_steps", None)),
        default_batch_size=_coerce_positive_int(
            prm_payload.get("batch"),
            getattr(base_cfg, "default_batch_size", 32),
        ),
        run_unfrozen_backbone_eval=False,
        reward_target_metric=(
            base_reward_target_metric
            if bool(getattr(base_cfg, "static_only", False))
            else FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
            if bool(getattr(base_cfg, "formal_nn_eval", False))
            and base_reward_target_metric == FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
            else (
                "frozen_train_acc"
                if base_reward_target_metric == "frozen_train_acc"
                else "frozen_test_acc"
            )
        ),
    )


def _default_formal_epoch_limit_minutes() -> float:
    try:
        from ab.nn.util.Const import default_epoch_limit_minutes  # type: ignore

        return max((1.0 / 60.0), float(default_epoch_limit_minutes))
    except Exception:
        return 30.0


def _resolve_formal_epoch_limit_minutes(cfg: "EvalConfig") -> float:
    explicit_limit = getattr(cfg, "formal_epoch_limit_minutes", None)
    if explicit_limit is not None:
        try:
            parsed = float(explicit_limit)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0.0:
            return parsed

    requested_minutes = float(getattr(cfg, "eval_limit_seconds", 0) or 0) / 60.0
    return max(_default_formal_epoch_limit_minutes(), requested_minutes)


def _effective_eval_limit_seconds(cfg: "EvalConfig") -> int:
    if bool(getattr(cfg, "formal_nn_eval", False)):
        return max(1, int(round(_resolve_formal_epoch_limit_minutes(cfg) * 60.0)))
    return max(1, int(getattr(cfg, "eval_limit_seconds", 270)))


def _is_formal_nn_eval_enabled(cfg: Optional["EvalConfig"]) -> bool:
    return bool(cfg is not None and getattr(cfg, "formal_nn_eval", False))


def _is_static_only_eval_enabled(cfg: Optional["EvalConfig"]) -> bool:
    return bool(cfg is not None and getattr(cfg, "static_only", False))


def _ensure_nn_dataset_importable() -> None:
    global _NN_DATASET_IMPORT_READY
    if _NN_DATASET_IMPORT_READY:
        return
    importlib.invalidate_caches()
    importlib.import_module("ab.nn.api")
    _NN_DATASET_IMPORT_READY = True


def _patch_nn_dataset_dataroll_for_reward() -> Optional[Tuple[type, Any, Any]]:
    try:
        import ab.nn.util.Train as nn_train  # type: ignore
        from ab.nn.util.Exception import LearnTimeException  # type: ignore
    except Exception:
        return None

    data_roll_cls = getattr(nn_train, "DataRoll", None)
    if data_roll_cls is None:
        return None

    original_init = getattr(data_roll_cls, "__init__", None)
    original_next = getattr(data_roll_cls, "__next__", None)
    if original_init is None or original_next is None:
        return None

    warmup_batches_default = 8
    measured_batches_default = 8
    warmup_batches = max(0, _safe_int_env("NNGPT_REWARD_FORMAL_WARMUP_BATCHES", warmup_batches_default))
    min_measured_batches = max(
        1,
        _safe_int_env("NNGPT_REWARD_FORMAL_MIN_MEASURED_BATCHES", measured_batches_default),
    )

    def _patched_init(self, dataset, epoch_limit_minutes):
        original_init(self, dataset, epoch_limit_minutes)
        try:
            total_batches = int(getattr(self, "total", 0) or 0)
        except (TypeError, ValueError):
            total_batches = 0
        effective_warmup = int(warmup_batches)
        if total_batches > 0:
            effective_warmup = min(effective_warmup, max(0, total_batches - 1))
        self._nngpt_reward_warmup_batches = effective_warmup
        self._nngpt_reward_min_measured_batches = int(min_measured_batches)
        self._nngpt_reward_warmup_elapsed = None
        self._nngpt_reward_steady_start_time = float(getattr(self, "init_time", time.time()))

    def _patched_next(self):
        now = time.time()
        try:
            current_n = int(getattr(self, "n", 0) or 0)
        except (TypeError, ValueError):
            current_n = 0
        try:
            total_batches = int(getattr(self, "total", 0) or 0)
        except (TypeError, ValueError):
            total_batches = 0

        warmup_count = int(getattr(self, "_nngpt_reward_warmup_batches", 0) or 0)
        if getattr(self, "_nngpt_reward_warmup_elapsed", None) is None and current_n >= warmup_count:
            if warmup_count > 0:
                self._nngpt_reward_warmup_elapsed = max(
                    0.0,
                    now - float(getattr(self, "init_time", now)),
                )
                self._nngpt_reward_steady_start_time = now
            else:
                self._nngpt_reward_warmup_elapsed = 0.0
                self._nngpt_reward_steady_start_time = float(getattr(self, "init_time", now))

        warmup_elapsed = getattr(self, "_nngpt_reward_warmup_elapsed", None)
        if warmup_elapsed is not None and total_batches > 0:
            measured_batches = max(0, current_n - warmup_count)
            min_batches = int(getattr(self, "_nngpt_reward_min_measured_batches", 1) or 1)
            if measured_batches >= min_batches:
                steady_start_time = float(getattr(self, "_nngpt_reward_steady_start_time", now))
                steady_elapsed = max(1e-1, now - steady_start_time)
                remaining_profiled_batches = max(0, total_batches - warmup_count)
                estimated_time = (
                    float(warmup_elapsed)
                    + (remaining_profiled_batches * steady_elapsed / max(1, measured_batches))
                ) / 60.0
                if estimated_time > float(getattr(self, "epoch_limit_minutes", 0.0) or 0.0):
                    total_elapsed = max(1e-1, now - float(getattr(self, "init_time", now)))
                    raise LearnTimeException(estimated_time, self.epoch_limit_minutes, total_elapsed)
        return self.it.__next__()

    data_roll_cls.__init__ = _patched_init
    data_roll_cls.__next__ = _patched_next
    return data_roll_cls, original_init, original_next


def _restore_nn_dataset_dataroll_patch(patch_token: Optional[Tuple[type, Any, Any]]) -> None:
    if patch_token is None:
        return
    data_roll_cls, original_init, original_next = patch_token
    data_roll_cls.__init__ = original_init
    data_roll_cls.__next__ = original_next


def _patch_nn_dataset_loader_utils_for_reward() -> Optional[Dict[str, Any]]:
    try:
        import ab.nn.util.Train as nn_train  # type: ignore
        import ab.nn.util.Util as nn_util  # type: ignore
    except Exception:
        return None

    original_train_loader = getattr(nn_train, "train_loader_f", None)
    original_test_loader = getattr(nn_train, "test_loader_f", None)
    original_util_train_loader = getattr(nn_util, "train_loader_f", None)
    original_util_test_loader = getattr(nn_util, "test_loader_f", None)
    if (
        original_train_loader is None
        or original_test_loader is None
        or original_util_train_loader is None
        or original_util_test_loader is None
    ):
        return None

    pin_memory = bool(torch.cuda.is_available()) and _safe_bool_env(
        "NNGPT_RL_FORMAL_PIN_MEMORY",
        True,
    )
    persistent_workers = _safe_bool_env(
        "NNGPT_RL_FORMAL_PERSISTENT_WORKERS",
        True,
    )
    prefetch_factor = _optional_positive_int_env("NNGPT_RL_FORMAL_PREFETCH_FACTOR")

    def _resolved_num_workers(dataset: Any, default_num_workers: int) -> int:
        value = getattr(dataset, "num_workers", default_num_workers)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default_num_workers or 0)
        return max(0, parsed)

    def _build_loader(dataset: Any, batch: int, num_workers: int, *, shuffle: bool) -> DataLoader:
        resolved_num_workers = _resolved_num_workers(dataset, num_workers)
        loader_kwargs: Dict[str, Any] = {
            "batch_size": batch,
            "shuffle": shuffle,
            "num_workers": resolved_num_workers,
            "collate_fn": getattr(dataset, "collate_fn", None),
        }
        if pin_memory:
            loader_kwargs["pin_memory"] = True
        if resolved_num_workers > 0:
            if persistent_workers:
                loader_kwargs["persistent_workers"] = True
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        return DataLoader(dataset, **loader_kwargs)

    def _train_loader(dataset: Any, batch: int, num_workers: int) -> DataLoader:
        return _build_loader(dataset, batch, num_workers, shuffle=True)

    def _test_loader(dataset: Any, batch: int, num_workers: int) -> DataLoader:
        return _build_loader(dataset, batch, num_workers, shuffle=False)

    nn_train.train_loader_f = _train_loader
    nn_train.test_loader_f = _test_loader
    nn_util.train_loader_f = _train_loader
    nn_util.test_loader_f = _test_loader
    return {
        "nn_train": nn_train,
        "nn_util": nn_util,
        "train_loader_f": original_train_loader,
        "test_loader_f": original_test_loader,
        "util_train_loader_f": original_util_train_loader,
        "util_test_loader_f": original_util_test_loader,
    }


def _restore_nn_dataset_loader_utils_patch(patch_token: Optional[Dict[str, Any]]) -> None:
    if not isinstance(patch_token, dict):
        return
    nn_train = patch_token.get("nn_train")
    nn_util = patch_token.get("nn_util")
    if nn_train is not None:
        nn_train.train_loader_f = patch_token.get("train_loader_f")
        nn_train.test_loader_f = patch_token.get("test_loader_f")
    if nn_util is not None:
        nn_util.train_loader_f = patch_token.get("util_train_loader_f")
        nn_util.test_loader_f = patch_token.get("util_test_loader_f")


def _configure_formal_eval_cuda_backend(device: str) -> Optional[Dict[str, Any]]:
    if (not torch.cuda.is_available()) or (not str(device).startswith("cuda")):
        return None

    previous_state = {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if _safe_bool_env("NNGPT_RL_FORMAL_CUDNN_BENCHMARK", True):
        torch.backends.cudnn.benchmark = True
    return previous_state


def _restore_formal_eval_cuda_backend(state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    try:
        torch.backends.cudnn.benchmark = bool(state.get("cudnn_benchmark", False))
    except Exception:
        pass


def _coerce_optional_metric_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _formal_epoch_stat_sort_key(path: Path) -> tuple[int, Any]:
    try:
        return 0, int(path.stem)
    except (TypeError, ValueError):
        return 1, path.name


def _extract_formal_epoch_stat_record(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
    return None


def _formal_empty_horizon_summary(*, epochs_completed_for_horizon: int = 0) -> Dict[str, Any]:
    return {
        "test_acc": None,
        "train_acc": None,
        "best_epoch_loss": None,
        "avg_epoch_loss": None,
        "loss_drop": None,
        "epochs_completed_for_horizon": int(max(0, epochs_completed_for_horizon)),
        "reached_horizon": False,
    }


def _formal_horizon_summary(epoch_rows: list[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    relevant_rows = [
        dict(item)
        for item in epoch_rows
        if int(item.get("epoch", 0) or 0) <= int(horizon)
    ]
    epochs_completed_for_horizon = max(
        [int(item.get("epoch", 0) or 0) for item in relevant_rows],
        default=0,
    )
    if horizon <= 0 or epochs_completed_for_horizon < int(horizon):
        return _formal_empty_horizon_summary(
            epochs_completed_for_horizon=epochs_completed_for_horizon,
        )
    summary = _formal_history_core_from_epoch_rows(relevant_rows)
    return {
        "test_acc": summary.get("test_acc"),
        "train_acc": summary.get("train_acc"),
        "best_epoch_loss": summary.get("best_epoch_loss"),
        "avg_epoch_loss": summary.get("avg_epoch_loss"),
        "loss_drop": summary.get("loss_drop"),
        "epochs_completed_for_horizon": int(horizon),
        "reached_horizon": True,
    }


def _formal_horizon_payloads(
    epoch_rows: list[Dict[str, Any]],
    *,
    configured_horizons: Optional[list[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    requested_horizons = list(configured_horizons or [])
    ordered_horizons = sorted(
        {
            int(horizon)
            for horizon in [*FORMAL_REWARD_SCORE_HORIZONS, *requested_horizons]
            if int(horizon) > 0
        }
    )
    for horizon in ordered_horizons:
        payloads[str(horizon)] = _formal_horizon_summary(epoch_rows, horizon)
    return payloads


def _formal_horizon_metric_map(
    payloads: Dict[str, Dict[str, Any]],
    field_name: str,
) -> Dict[str, Optional[float]]:
    return {
        str(horizon): _coerce_optional_metric_float((payloads.get(str(horizon)) or {}).get(field_name))
        for horizon in FORMAL_REWARD_SCORE_HORIZONS
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _formal_horizon_score_map(payloads: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    def _completed(horizon: int) -> bool:
        return bool((payloads.get(str(horizon)) or {}).get("reached_horizon"))

    def _metric(horizon: int, field_name: str) -> Optional[float]:
        if not _completed(horizon):
            return None
        return _coerce_optional_metric_float((payloads.get(str(horizon)) or {}).get(field_name))

    acc1 = _metric(1, "test_acc")
    acc5 = _metric(5, "test_acc")
    acc10 = _metric(10, "test_acc")
    train10 = _metric(10, "train_acc")

    score1 = _clip(float(acc1 or 0.0), 0.0, 1.0) if acc1 is not None else 0.0
    score5 = (
        _clip(float(acc5) + 0.25 * max(0.0, float(acc5) - float(acc1 or 0.0)), 0.0, 1.0)
        if acc5 is not None
        else 0.0
    )
    overfit10 = (
        max(0.0, float(train10) - float(acc10) - 0.08)
        if train10 is not None and acc10 is not None
        else 0.0
    )
    score10 = (
        _clip(float(acc10) + 0.20 * max(0.0, float(acc10) - float(acc5 or 0.0)) - 0.50 * overfit10, 0.0, 1.0)
        if acc10 is not None
        else 0.0
    )
    return {
        "1": float(score1),
        "5": float(score5),
        "10": float(score10),
    }


def _formal_multi_horizon_target_value(payloads: Dict[str, Dict[str, Any]]) -> float:
    scores = _formal_horizon_score_map(payloads)
    return float(
        0.20 * float(scores.get("1", 0.0) or 0.0)
        + 0.35 * float(scores.get("5", 0.0) or 0.0)
        + 0.45 * float(scores.get("10", 0.0) or 0.0)
    )


def _formal_history_core_from_epoch_rows(epoch_rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not epoch_rows:
        return {
            "loss_start": None,
            "loss_end": None,
            "loss_drop": None,
            "loss_drop_ok": False,
            "best_epoch_loss": None,
            "avg_epoch_loss": None,
            "epochs_completed": 0,
            "epoch_loss_series": [],
            "training_context_metric_name": "best_epoch_loss",
            "training_context_metric_value": None,
            "train_acc": None,
            "test_acc": None,
        }

    test_loss_series = [float(item["test_loss"]) for item in epoch_rows if item.get("test_loss") is not None]
    train_loss_series = [float(item["train_loss"]) for item in epoch_rows if item.get("train_loss") is not None]
    if len(test_loss_series) == len(epoch_rows) and test_loss_series:
        loss_series = list(test_loss_series)
    elif train_loss_series:
        loss_series = list(train_loss_series)
    else:
        loss_series = list(test_loss_series)

    epochs_completed = max(int(item.get("epoch", 0) or 0) for item in epoch_rows)
    if epochs_completed <= 0:
        epochs_completed = len(loss_series)

    best_epoch_loss = min(loss_series) if loss_series else None
    avg_epoch_loss = (
        float(sum(loss_series) / len(loss_series))
        if loss_series
        else None
    )
    loss_start = loss_series[0] if loss_series else None
    loss_end = loss_series[-1] if loss_series else None
    loss_drop = (
        float(loss_start - loss_end)
        if loss_start is not None and loss_end is not None
        else None
    )
    loss_drop_ok = False
    if loss_start is not None and loss_end is not None:
        rel_drop_ok = loss_start > 0.0 and (loss_end <= loss_start * 0.98)
        loss_drop_ok = bool(loss_end < (loss_start - 1e-3) or rel_drop_ok)

    train_acc = next(
        (item["train_accuracy"] for item in reversed(epoch_rows) if item.get("train_accuracy") is not None),
        None,
    )
    test_acc = next(
        (item["test_accuracy"] for item in reversed(epoch_rows) if item.get("test_accuracy") is not None),
        None,
    )
    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_drop": loss_drop,
        "loss_drop_ok": loss_drop_ok,
        "best_epoch_loss": best_epoch_loss,
        "avg_epoch_loss": avg_epoch_loss,
        "epochs_completed": int(epochs_completed or 0),
        "epoch_loss_series": list(loss_series),
        "training_context_metric_name": "best_epoch_loss",
        "training_context_metric_value": best_epoch_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def _formal_history_from_epoch_rows(epoch_rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _formal_history_core_from_epoch_rows(epoch_rows)
    formal_reward_epochs = _resolve_formal_reward_epochs()
    formal_horizons = _formal_horizon_payloads(
        epoch_rows,
        configured_horizons=formal_reward_epochs,
    )
    summary.update(
        {
            "formal_horizons": formal_horizons,
            "formal_horizon_test_acc": _formal_horizon_metric_map(formal_horizons, "test_acc"),
            "formal_horizon_train_acc": _formal_horizon_metric_map(formal_horizons, "train_acc"),
            "formal_horizon_scores": _formal_horizon_score_map(formal_horizons),
            "formal_reward_target_value": _formal_multi_horizon_target_value(formal_horizons),
        }
    )
    return summary


def _append_formal_epoch_row(epoch_rows: list[Dict[str, Any]], record: Dict[str, Any], epoch_hint: Any) -> None:
    if not isinstance(record, dict):
        return
    try:
        epoch_index = int(record.get("epoch") or epoch_hint)
    except (TypeError, ValueError):
        epoch_index = len(epoch_rows) + 1
    epoch_rows.append(
        {
            "epoch": epoch_index,
            "train_loss": _coerce_optional_metric_float(record.get("train_loss")),
            "test_loss": _coerce_optional_metric_float(record.get("test_loss")),
            "train_accuracy": _coerce_optional_metric_float(record.get("train_accuracy")),
            "test_accuracy": _coerce_optional_metric_float(
                record.get("accuracy", record.get("test_accuracy"))
            ),
        }
    )


def _load_formal_training_history(
    stats_dir: Path,
    *,
    summary_path: Optional[Path] = None,
    min_summary_mtime: Optional[float] = None,
) -> Dict[str, Any]:
    epoch_rows: list[Dict[str, Any]] = []
    for stat_path in sorted(stats_dir.glob("*.json"), key=_formal_epoch_stat_sort_key):
        try:
            payload = json.loads(stat_path.read_text())
        except Exception:
            continue
        record = _extract_formal_epoch_stat_record(payload)
        if not isinstance(record, dict):
            continue
        _append_formal_epoch_row(epoch_rows, record, stat_path.stem)

    if epoch_rows:
        return _formal_history_from_epoch_rows(epoch_rows)

    if summary_path is not None:
        try:
            resolved_summary_path = Path(summary_path).expanduser()
            if not resolved_summary_path.is_absolute():
                resolved_summary_path = resolved_summary_path.resolve()
            if not resolved_summary_path.exists():
                raise FileNotFoundError(resolved_summary_path)
            if (
                min_summary_mtime is not None
                and resolved_summary_path.stat().st_mtime < float(min_summary_mtime)
            ):
                raise FileNotFoundError(f"stale summary: {resolved_summary_path}")
            summary_payload = json.loads(resolved_summary_path.read_text())

            epoch_details = summary_payload.get("epoch_details") or summary_payload.get("epoch_history") or []
            for item in epoch_details:
                _append_formal_epoch_row(epoch_rows, item, len(epoch_rows) + 1)
            if not epoch_rows:
                learning_curves = summary_payload.get("learning_curves") or {}
                curve_epochs = learning_curves.get("epochs") or []
                train_loss_curve = learning_curves.get("train_loss") or []
                test_loss_curve = learning_curves.get("test_loss") or []
                train_acc_curve = learning_curves.get("train_accuracy") or []
                test_acc_curve = learning_curves.get("test_accuracy") or []
                curve_len = max(
                    len(curve_epochs),
                    len(train_loss_curve),
                    len(test_loss_curve),
                    len(train_acc_curve),
                    len(test_acc_curve),
                )
                for index in range(curve_len):
                    _append_formal_epoch_row(
                        epoch_rows,
                        {
                            "epoch": curve_epochs[index] if index < len(curve_epochs) else index + 1,
                            "train_loss": train_loss_curve[index] if index < len(train_loss_curve) else None,
                            "test_loss": test_loss_curve[index] if index < len(test_loss_curve) else None,
                            "train_accuracy": train_acc_curve[index] if index < len(train_acc_curve) else None,
                            "test_accuracy": test_acc_curve[index] if index < len(test_acc_curve) else None,
                        },
                        index + 1,
                    )
            if epoch_rows:
                return _formal_history_from_epoch_rows(epoch_rows)
        except Exception:
            pass

    return _formal_history_from_epoch_rows([])


def _formal_eval_with_nn_dataset(
    code: str,
    *,
    prm: Dict[str, Any],
    cfg: "EvalConfig",
    freeze_backbones: bool,
    seed_accuracy_baseline: Optional[float],
    backbone_model_names: list[str],
) -> Dict[str, Any]:
    effective_eval_limit_seconds = _effective_eval_limit_seconds(cfg)
    epoch_limit_minutes = _resolve_formal_epoch_limit_minutes(cfg)
    formal_reward_epochs = _resolve_formal_reward_epochs()
    formal_reward_max_epoch = max(formal_reward_epochs) if formal_reward_epochs else int(max(DEFAULT_FORMAL_REWARD_EPOCHS))
    formal_trace_context: Dict[str, Any] = {
        "freeze_backbones": bool(freeze_backbones),
        "formal_task": str(getattr(cfg, "formal_task", "img-classification")),
        "formal_dataset": str(getattr(cfg, "formal_dataset", "cifar-10")),
        "formal_metric": str(getattr(cfg, "formal_metric", "acc")),
        "backbone_model_names": list(backbone_model_names),
        "code_trace": _formal_eval_code_trace(code),
        "epoch_limit_minutes": epoch_limit_minutes,
        "formal_reward_epochs": list(formal_reward_epochs),
        "formal_reward_max_epoch": int(formal_reward_max_epoch),
    }
    safe_prm = dict(prm)
    safe_prm["freeze_backbones"] = bool(freeze_backbones)
    safe_prm["epoch"] = int(formal_reward_max_epoch)
    formal_trace_context.update(
        {
            "transform": str(safe_prm.get("transform")),
            "batch": int(safe_prm.get("batch", 0) or 0),
            "epoch": int(safe_prm.get("epoch", 0) or 0),
            "num_workers": int(safe_prm.get("num_workers", 1) or 1),
            "trainer_device": str(cfg.device),
            "trainer_in_shape": tuple(cfg.input_shape),
            "dataset_out_shape": (int(cfg.n_classes),),
            "formal_eval_backend": "ab.nn.api.check_nn",
        }
    )

    def _failure_result(
        *,
        stage: str,
        error_message: str,
        built_ok: bool,
        forward_ok: bool,
        forward_shape_ok: bool,
        latency_ms: Optional[float],
        params_m: Optional[float],
        estimated_total_seconds: Optional[float] = None,
        timed_out: bool = False,
    ) -> Dict[str, Any]:
        error_type = error_message.split(":", 1)[0].strip() or "RuntimeError"
        components = compute_cv_reward_simple(
            built_ok=built_ok,
            forward_shape_ok=forward_shape_ok,
            backward_ok=False,
            loss_drop_ok=False,
            val_metric=None,
            val_metric_baseline=None,
            latency_ms=latency_ms,
            params_m=params_m,
            flops_g=None,
            critic_score=None,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )
        result = {
            "reward": components["reward"],
            "components": components,
            **{
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=effective_eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "built_ok": built_ok,
                "forward_ok": forward_ok,
                "forward_shape_ok": forward_shape_ok,
                "latency_ms": latency_ms,
                "params_m": params_m,
                "estimated_total_seconds": estimated_total_seconds,
                "timed_out": timed_out,
                "error": error_message,
                "formal_reward_epochs": list(formal_reward_epochs),
                "formal_reward_max_epoch": int(formal_reward_max_epoch),
            },
        }
        return _apply_error_trace(
            result,
            error_type=error_type,
            error_stage=stage,
            error_context=formal_trace_context,
            error_hint=_infer_error_hint(
                error_message=error_message,
                context=formal_trace_context,
            ),
        )

    try:
        _ensure_nn_dataset_importable()
        import ab.nn.api as nn_api  # type: ignore
    except Exception as exc:
        return _failure_result(
            stage="import_nn_dataset",
            error_message=(
                "RuntimeError: Formal nn-dataset evaluation requested, but `ab.nn.api` "
                "is not importable from the current Python environment."
            ),
            built_ok=False,
            forward_ok=False,
            forward_shape_ok=False,
            latency_ms=None,
            params_m=None,
        )

    preflight_model = None
    params_m: Optional[float] = None
    forward_ok = False
    forward_shape_ok = False
    latency_ms: Optional[float] = None
    try:
        build_fn = build_fn_from_code(
            code,
            tuple(cfg.input_shape),
            (int(cfg.n_classes),),
            safe_prm,
            str(cfg.device),
        )
        preflight_model = build_fn()
        preflight_model.to(str(cfg.device))
        params_m = _count_params_m(preflight_model)
        formal_trace_context["params_m"] = params_m
        forward_result = _quick_forward(
            preflight_model,
            tuple(cfg.input_shape),
            device=str(cfg.device),
            n_classes=int(cfg.n_classes),
        )
        forward_ok = bool(forward_result.get("forward_ok"))
        forward_shape_ok = bool(forward_result.get("forward_shape_ok"))
        latency_ms = forward_result.get("latency_ms")
        formal_trace_context["forward_output_shape"] = forward_result.get("output_shape")
        if not forward_ok or not forward_shape_ok:
            output_shape = forward_result.get("output_shape")
            if forward_result.get("error"):
                error_message = str(forward_result.get("error"))
            else:
                error_message = (
                    "RuntimeError: formal eval forward shape "
                    f"{output_shape!r} incompatible with expected classes {int(cfg.n_classes)}"
                )
            return _failure_result(
                stage="preflight_forward",
                error_message=error_message,
                built_ok=True,
                forward_ok=forward_ok,
                forward_shape_ok=forward_shape_ok,
                latency_ms=latency_ms,
                params_m=params_m,
                estimated_total_seconds=0.0,
            )
    except Exception as exc:
        return _failure_result(
            stage="preflight_build",
            error_message=f"{type(exc).__name__}: {exc}",
            built_ok=False,
            forward_ok=False,
            forward_shape_ok=False,
            latency_ms=None,
            params_m=params_m,
        )
    finally:
        if preflight_model is not None:
            try:
                preflight_model.to("cpu")
            except Exception:
                pass
            del preflight_model
        _clear_reward_cuda_state()

    started_at = time.time()
    data_roll_patch = None
    loader_patch = None
    cuda_backend_state = None
    try:
        unique_code = (
            f"# reward formal eval nonce pid={os.getpid()} ns={time.time_ns()} "
            f"freeze={int(bool(freeze_backbones))}\n{code}"
        )
        data_roll_patch = _patch_nn_dataset_dataroll_for_reward()
        loader_patch = _patch_nn_dataset_loader_utils_for_reward()
        cuda_backend_state = _configure_formal_eval_cuda_backend(str(cfg.device))
        formal_trace_context["reward_dataroll_warmup_batches"] = _safe_int_env(
            "NNGPT_REWARD_FORMAL_WARMUP_BATCHES",
            8,
        )
        formal_trace_context["reward_dataroll_min_measured_batches"] = _safe_int_env(
            "NNGPT_REWARD_FORMAL_MIN_MEASURED_BATCHES",
            8,
        )
        formal_trace_context["reward_loader_pin_memory"] = bool(torch.cuda.is_available()) and _safe_bool_env(
            "NNGPT_RL_FORMAL_PIN_MEMORY",
            True,
        )
        formal_trace_context["reward_loader_persistent_workers"] = _safe_bool_env(
            "NNGPT_RL_FORMAL_PERSISTENT_WORKERS",
            True,
        )
        formal_trace_context["reward_loader_prefetch_factor"] = _optional_positive_int_env(
            "NNGPT_RL_FORMAL_PREFETCH_FACTOR"
        )
        formal_trace_context["reward_cuda_cudnn_benchmark"] = _safe_bool_env(
            "NNGPT_RL_FORMAL_CUDNN_BENCHMARK",
            True,
        )
        with tempfile.TemporaryDirectory(prefix="nngpt_reward_formal_") as temp_stats_dir:
            summary_path = (Path.cwd() / "out" / "training_summary.json").resolve()
            _model_name, test_acc, _accuracy_to_time, _code_score = nn_api.check_nn(
                unique_code,
                str(getattr(cfg, "formal_task", "img-classification")),
                str(getattr(cfg, "formal_dataset", "cifar-10")),
                str(getattr(cfg, "formal_metric", "acc")),
                safe_prm,
                False,
                None,
                temp_stats_dir,
                False,
                epoch_limit_minutes,
            )
            formal_history = _load_formal_training_history(
                Path(temp_stats_dir),
                summary_path=summary_path,
                min_summary_mtime=started_at - 1.0,
            )
        formal_duration_seconds = max(0.0, time.time() - started_at)
        formal_trace_context["formal_eval_duration_seconds"] = formal_duration_seconds
        components = compute_cv_reward_simple(
            built_ok=True,
            forward_shape_ok=True,
            backward_ok=True,
            loss_drop_ok=True,
            val_metric=float(test_acc),
            val_metric_baseline=None,
            latency_ms=latency_ms,
            params_m=params_m,
            flops_g=None,
            critic_score=None,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )
        train_acc = formal_history.get("train_acc")
        seed_train_acc_gap = None
        seed_train_acc_improved = False
        if train_acc is not None and seed_accuracy_baseline is not None:
            seed_train_acc_gap = float(train_acc - seed_accuracy_baseline)
            seed_train_acc_improved = bool(seed_train_acc_gap > 0.0)
        return {
            "reward": components["reward"],
            "components": components,
            **{
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=effective_eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "test_acc": float(test_acc),
                "val_metric": float(test_acc),
                "built_ok": True,
                "forward_ok": True,
                "forward_shape_ok": True,
                "trained_step_ok": True,
                "backward_ok": True,
                "loss_start": formal_history.get("loss_start"),
                "loss_end": formal_history.get("loss_end"),
                "loss_drop": formal_history.get("loss_drop"),
                "loss_drop_ok": bool(formal_history.get("loss_drop_ok", True)),
                "best_epoch_loss": formal_history.get("best_epoch_loss"),
                "avg_epoch_loss": formal_history.get("avg_epoch_loss"),
                "epochs_completed": int(formal_history.get("epochs_completed", 0) or 0),
                "epoch_loss_series": list(formal_history.get("epoch_loss_series") or []),
                "training_context_metric_name": str(
                    formal_history.get("training_context_metric_name") or "best_epoch_loss"
                ),
                "training_context_metric_value": formal_history.get("training_context_metric_value"),
                "train_acc": train_acc,
                "seed_train_acc_gap": seed_train_acc_gap,
                "seed_train_acc_improved": seed_train_acc_improved,
                "train_acc_gain": seed_train_acc_gap,
                "train_acc_improved": seed_train_acc_improved,
                "latency_ms": latency_ms,
                "params_m": params_m,
                "estimated_total_seconds": formal_duration_seconds,
                "formal_reward_epochs": list(formal_reward_epochs),
                "formal_reward_max_epoch": int(formal_reward_max_epoch),
                "formal_horizon_test_acc": dict(formal_history.get("formal_horizon_test_acc") or {}),
                "formal_horizon_train_acc": dict(formal_history.get("formal_horizon_train_acc") or {}),
                "formal_horizon_scores": dict(formal_history.get("formal_horizon_scores") or {}),
                "formal_reward_target_value": _coerce_optional_metric_float(
                    formal_history.get("formal_reward_target_value")
                ),
            },
        }
    except Exception as exc:
        formal_duration_seconds = max(0.0, time.time() - started_at)
        formal_trace_context["formal_eval_duration_seconds"] = formal_duration_seconds
        estimated_total_seconds = formal_duration_seconds
        if hasattr(exc, "estimated_training_time"):
            try:
                formal_trace_context["estimated_training_time_minutes"] = float(exc.estimated_training_time)
                estimated_total_seconds = float(exc.estimated_training_time) * 60.0
            except Exception:
                pass
        if hasattr(exc, "accuracy"):
            try:
                formal_trace_context["reported_accuracy"] = float(exc.accuracy)
            except Exception:
                pass
        if hasattr(exc, "duration"):
            try:
                formal_trace_context["reported_duration_seconds"] = float(exc.duration)
            except Exception:
                pass
        return _failure_result(
            stage="check_nn",
            error_message=f"{type(exc).__name__}: {exc}",
            built_ok=True,
            forward_ok=forward_ok,
            forward_shape_ok=forward_shape_ok,
            latency_ms=latency_ms,
            params_m=params_m,
            estimated_total_seconds=estimated_total_seconds,
            timed_out=type(exc).__name__ == "LearnTimeException",
        )
    finally:
        _restore_formal_eval_cuda_backend(cuda_backend_state)
        _restore_nn_dataset_loader_utils_patch(loader_patch)
        _restore_nn_dataset_dataroll_patch(data_roll_patch)
        _clear_reward_cuda_state()


def _empty_eval_result(
    *,
    reward: float = 0.0,
    error: Optional[str] = None,
    seed_accuracy_baseline: Optional[float] = None,
    eval_limit_seconds: Optional[int] = None,
    backbone_model_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return RewardPayload.empty_eval_result(
        reward=reward,
        error=error,
        seed_accuracy_baseline=seed_accuracy_baseline,
        eval_limit_seconds=eval_limit_seconds,
        backbone_model_names=backbone_model_names,
        formal_reward_epochs=DEFAULT_FORMAL_REWARD_EPOCHS,
    )


def _timeout_eval_result(
    *,
    error: str,
    estimated_total_seconds: Optional[float],
    eval_limit_seconds: Optional[int],
    seed_accuracy_baseline: Optional[float] = None,
    backbone_model_names: Optional[list[str]] = None,
    built_ok: bool = False,
    forward_ok: bool = False,
    forward_shape_ok: bool = False,
    trained_step_ok: bool = False,
    backward_ok: bool = False,
    loss_start: Optional[float] = None,
    loss_end: Optional[float] = None,
    loss_drop: Optional[float] = None,
    loss_drop_ok: bool = False,
    best_epoch_loss: Optional[float] = None,
    avg_epoch_loss: Optional[float] = None,
    epochs_completed: int = 0,
    epoch_loss_series: Optional[list[float]] = None,
    training_context_metric_name: Optional[str] = "best_epoch_loss",
    training_context_metric_value: Optional[float] = None,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    components = compute_cv_reward_simple(
        built_ok=built_ok,
        forward_shape_ok=forward_shape_ok,
        backward_ok=backward_ok,
        loss_drop_ok=loss_drop_ok,
        val_metric=None,
        val_metric_baseline=None,
        latency_ms=latency_ms,
        params_m=params_m,
        flops_g=None,
        critic_score=None,
        kl_div=kl_div,
        weights=weights,
    )
    return RewardPayload.timeout_eval_result(
        components=components,
        error=error,
        estimated_total_seconds=estimated_total_seconds,
        eval_limit_seconds=eval_limit_seconds,
        seed_accuracy_baseline=seed_accuracy_baseline,
        backbone_model_names=backbone_model_names,
        built_ok=built_ok,
        forward_ok=forward_ok,
        forward_shape_ok=forward_shape_ok,
        trained_step_ok=trained_step_ok,
        backward_ok=backward_ok,
        loss_start=loss_start,
        loss_end=loss_end,
        loss_drop=loss_drop,
        loss_drop_ok=loss_drop_ok,
        best_epoch_loss=best_epoch_loss,
        avg_epoch_loss=avg_epoch_loss,
        epochs_completed=epochs_completed,
        epoch_loss_series=epoch_loss_series,
        training_context_metric_name=training_context_metric_name,
        training_context_metric_value=training_context_metric_value,
        latency_ms=latency_ms,
        params_m=params_m,
        formal_reward_epochs=DEFAULT_FORMAL_REWARD_EPOCHS,
    )

def evaluate_and_reward(
    *,
    # Either pass a ready model OR a builder that returns nn.Module
    model: Optional[nn.Module] = None,
    build_fn: Optional[Callable[[], nn.Module]] = None,

    # Optional loaders; if None, CIFAR-10 proxy loaders will be built
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,

    # Validation accuracy stays logging-only. Seed accuracy is logged but no longer drives RL reward.
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,

    # Config & weights
    cfg: EvalConfig = EvalConfig(),
    backbone_model_names: Optional[list[str]] = None,
    freeze_backbones: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build model (if builder provided)
      2) Forward sanity
      3) Train full epoch(s)
      4) Quick validation (accuracy)
      5) (optional) critic score / KL
      6) Compute scalar reward
    Returns:
      {
        'reward': float,
        'components': {...},
        'val_metric': float or None,
        'built_ok': bool,
        'forward_ok': bool,
        'forward_shape_ok': bool,
        'trained_step_ok': bool,
        'backward_ok': bool,
        'loss_start': float | None,
        'loss_end': float | None,
        'loss_drop': float | None,
        'loss_drop_ok': bool,
        'latency_ms': float or None,
        'params_m': float,
    }
    """
    device = cfg.device
    built_ok = False
    forward_ok = False
    forward_shape_ok = False
    trained_step_ok = False
    backward_ok = False
    latency_ms = None
    params_m = None
    val_metric = None
    critic_score = None
    loss_start = None
    loss_end = None
    loss_drop = None
    loss_drop_ok = False
    best_epoch_loss = None
    avg_epoch_loss = None
    epochs_completed = 0
    epoch_loss_series: list[float] = []
    training_context_metric_name: Optional[str] = "best_epoch_loss"
    training_context_metric_value = None
    train_acc = None
    seed_train_acc_gap = None
    seed_train_acc_improved = False
    time_budget = EvalTimeBudget(cfg)

    # 1) Build or use provided model
    mdl = model
    local_train_loader = None
    local_val_loader = None

    try:
        effective_train_epochs = _coerce_positive_int(getattr(cfg, "train_epochs", 1), 1)
        effective_train_step_cap = _normalize_optional_steps(getattr(cfg, "train_steps", None))
        if mdl is None and build_fn is not None:
            try:
                mdl = build_fn()
            except Exception:
                mdl = None
        if mdl is not None and isinstance(mdl, nn.Module):
            try:
                mdl.to(device)
                params_m = _count_params_m(mdl)
                built_ok = True
                time_budget.mark_build_complete()
            except Exception:
                built_ok = False
                mdl = None
        else:
            time_budget.check("build")

        # If build failed, compute minimal reward and return
        if not built_ok or mdl is None:
            time_budget.check("build")
            components = compute_cv_reward_simple(
                built_ok=False,
                forward_shape_ok=False,
                backward_ok=False,
                loss_drop_ok=False,
                val_metric=None,
                val_metric_baseline=val_metric_baseline,
                latency_ms=None,
                params_m=None,
                flops_g=None,
                critic_score=None,
                kl_div=cfg.kl_div,
                weights=cfg.weights
            )
            return {
                "reward": components["reward"],
                "components": components,
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
            }

        # 2) Forward sanity (and optional latency)
        forward_result = _quick_forward(
            mdl,
            cfg.input_shape,
            device=device,
            n_classes=cfg.n_classes,
        )
        forward_ok = bool(forward_result["forward_ok"])
        forward_shape_ok = bool(forward_result["forward_shape_ok"])
        if cfg.measure_latency:
            latency_ms = forward_result["latency_ms"]
        if forward_ok:
            time_budget.mark_forward_complete()
        else:
            time_budget.check("forward")

        # 3) Train full epoch(s) on the reward subset.
        if train_loader is None or val_loader is None:
            local_train_loader, local_val_loader = _build_cifar10_loaders(cfg)
        used_train_loader = train_loader if train_loader is not None else local_train_loader
        used_val_loader = val_loader if val_loader is not None else local_val_loader
        expected_train_batches = max(1, len(used_train_loader) * effective_train_epochs)
        if effective_train_step_cap is not None:
            expected_train_batches = min(expected_train_batches, effective_train_step_cap)
        effective_val_batches = (
            len(used_val_loader)
            if bool(getattr(cfg, "full_test_acc", False))
            else max(1, min(cfg.max_val_batches, len(used_val_loader)))
        )
        time_budget.set_expected_work(
            train_batches=expected_train_batches,
            train_accuracy_batches=max(1, len(used_train_loader)),
            val_accuracy_batches=effective_val_batches,
        )
        try:
            train_result = _train_steps(
                mdl,
                used_train_loader,
                epochs=effective_train_epochs,
                max_steps=effective_train_step_cap,
                device=device,
                n_classes=cfg.n_classes,
                time_budget=time_budget,
                freeze_backbones=freeze_backbones,
            )
            backward_ok = bool(train_result["backward_ok"])
            trained_step_ok = bool(train_result["trained_step_ok"])
            loss_start = train_result["loss_start"]
            loss_end = train_result["loss_end"]
            loss_drop = train_result["loss_drop"]
            loss_drop_ok = bool(train_result["loss_drop_ok"])
            best_epoch_loss = train_result.get("best_epoch_loss")
            avg_epoch_loss = train_result.get("avg_epoch_loss")
            epochs_completed = int(train_result.get("epochs_completed", 0) or 0)
            epoch_loss_series = list(train_result.get("epoch_loss_series") or [])
            training_context_metric_name = train_result.get("training_context_metric_name", "best_epoch_loss")
            training_context_metric_value = train_result.get("training_context_metric_value")

            # 4) Quick validation (accuracy)
            train_metric_batches = max(1, len(used_train_loader))
            train_acc = _quick_accuracy(
                mdl,
                used_train_loader,
                device=device,
                max_batches=train_metric_batches,
                time_budget=time_budget,
                phase="train_accuracy",
            )
            val_metric = _quick_accuracy(
                mdl,
                used_val_loader,
                device=device,
                max_batches=None if bool(getattr(cfg, "full_test_acc", False)) else cfg.max_val_batches,
                time_budget=time_budget,
                phase="val_accuracy",
            )
            if seed_accuracy_baseline is not None:
                seed_train_acc_gap = float(train_acc - seed_accuracy_baseline)
                seed_train_acc_improved = bool(seed_train_acc_gap > 0.0)
        except EvalTimeException as exc:
            partial = exc.partial or {}
            loss_start = partial.get("loss_start", loss_start)
            loss_end = partial.get("loss_end", loss_end)
            loss_drop = partial.get("loss_drop", loss_drop)
            backward_ok = bool(partial.get("backward_ok", False))
            trained_step_ok = bool(partial.get("trained_step_ok", False))
            best_epoch_loss = partial.get("best_epoch_loss", best_epoch_loss)
            avg_epoch_loss = partial.get("avg_epoch_loss", avg_epoch_loss)
            epochs_completed = int(partial.get("epochs_completed", epochs_completed) or 0)
            epoch_loss_series = list(partial.get("epoch_loss_series") or epoch_loss_series)
            training_context_metric_name = partial.get(
                "training_context_metric_name",
                training_context_metric_name,
            )
            training_context_metric_value = partial.get(
                "training_context_metric_value",
                training_context_metric_value,
            )
            return _timeout_eval_result(
                error=f"{type(exc).__name__}: {exc}",
                estimated_total_seconds=exc.estimated_total_seconds,
                eval_limit_seconds=cfg.eval_limit_seconds,
                seed_accuracy_baseline=seed_accuracy_baseline,
                backbone_model_names=backbone_model_names,
                built_ok=built_ok,
                forward_ok=forward_ok,
                forward_shape_ok=forward_shape_ok,
                trained_step_ok=trained_step_ok,
                backward_ok=backward_ok,
                loss_start=loss_start,
                loss_end=loss_end,
                loss_drop=loss_drop,
                loss_drop_ok=False,
                best_epoch_loss=best_epoch_loss,
                avg_epoch_loss=avg_epoch_loss,
                epochs_completed=epochs_completed,
                epoch_loss_series=epoch_loss_series,
                training_context_metric_name=training_context_metric_name,
                training_context_metric_value=training_context_metric_value,
                latency_ms=latency_ms,
                params_m=params_m,
                kl_div=cfg.kl_div,
                weights=cfg.weights,
            )

        # 5) Optional critic score
        if cfg.critic_fn is not None:
            try:
                critic_score = float(cfg.critic_fn(mdl, {
                    "built_ok": built_ok,
                    "forward_ok": forward_ok,
                    "forward_shape_ok": forward_shape_ok,
                    "trained_step_ok": trained_step_ok,
                    "backward_ok": backward_ok,
                    "loss_start": loss_start,
                    "loss_end": loss_end,
                    "loss_drop": loss_drop,
                    "loss_drop_ok": loss_drop_ok,
                    "val_metric": val_metric,
                    "params_m": params_m,
                    "latency_ms": latency_ms
                }))
                critic_score = max(0.0, min(1.0, critic_score))
            except Exception:
                critic_score = None

        # 6) Compute reward
        components = compute_cv_reward_simple(
            built_ok=built_ok,
            forward_shape_ok=forward_shape_ok,
            backward_ok=backward_ok,
            loss_drop_ok=loss_drop_ok,
            val_metric=val_metric,
            val_metric_baseline=val_metric_baseline,
            latency_ms=latency_ms,
            params_m=params_m,
            flops_g=None,                 # flops left as None (optional to integrate)
            critic_score=critic_score,
            kl_div=cfg.kl_div,
            weights=cfg.weights
        )

        return {
            "reward": components["reward"],
            "components": components,
            **{
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "test_acc": val_metric,
                "val_metric": val_metric,
                "built_ok": built_ok,
                "forward_ok": forward_ok,
                "forward_shape_ok": forward_shape_ok,
                "trained_step_ok": trained_step_ok,
                "backward_ok": backward_ok,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "loss_drop": loss_drop,
                "loss_drop_ok": loss_drop_ok,
                "best_epoch_loss": best_epoch_loss,
                "avg_epoch_loss": avg_epoch_loss,
                "epochs_completed": epochs_completed,
                "epoch_loss_series": list(epoch_loss_series),
                "training_context_metric_name": training_context_metric_name,
                "training_context_metric_value": training_context_metric_value,
                "train_acc": train_acc,
                "seed_train_acc_gap": seed_train_acc_gap,
                "seed_train_acc_improved": seed_train_acc_improved,
                "train_acc_gain": seed_train_acc_gap,
                "train_acc_improved": seed_train_acc_improved,
                "latency_ms": latency_ms,
                "params_m": params_m,
                "estimated_total_seconds": time_budget.estimated_total_seconds,
            },
        }
    except EvalTimeException as exc:
        partial = exc.partial or {}
        loss_start = partial.get("loss_start", loss_start)
        loss_end = partial.get("loss_end", loss_end)
        loss_drop = partial.get("loss_drop", loss_drop)
        backward_ok = bool(partial.get("backward_ok", backward_ok))
        trained_step_ok = bool(partial.get("trained_step_ok", trained_step_ok))
        best_epoch_loss = partial.get("best_epoch_loss", best_epoch_loss)
        avg_epoch_loss = partial.get("avg_epoch_loss", avg_epoch_loss)
        epochs_completed = int(partial.get("epochs_completed", epochs_completed) or 0)
        epoch_loss_series = list(partial.get("epoch_loss_series") or epoch_loss_series)
        training_context_metric_name = partial.get(
            "training_context_metric_name",
            training_context_metric_name,
        )
        training_context_metric_value = partial.get(
            "training_context_metric_value",
            training_context_metric_value,
        )
        return _timeout_eval_result(
            error=f"{type(exc).__name__}: {exc}",
            estimated_total_seconds=exc.estimated_total_seconds,
            eval_limit_seconds=cfg.eval_limit_seconds,
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
            built_ok=built_ok,
            forward_ok=forward_ok,
            forward_shape_ok=forward_shape_ok,
            trained_step_ok=trained_step_ok,
            backward_ok=backward_ok,
            loss_start=loss_start,
            loss_end=loss_end,
            loss_drop=loss_drop,
            loss_drop_ok=False,
            best_epoch_loss=best_epoch_loss,
            avg_epoch_loss=avg_epoch_loss,
            epochs_completed=epochs_completed,
            epoch_loss_series=epoch_loss_series,
            training_context_metric_name=training_context_metric_name,
            training_context_metric_value=training_context_metric_value,
            latency_ms=latency_ms,
            params_m=params_m,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )

    finally:
        # Clean up memory explicitly and GUARANTEED
        if mdl is not None:
            try:
                mdl.to("cpu")
            except Exception:
                pass
            del mdl
        if local_train_loader is not None:
            del local_train_loader
        if local_val_loader is not None:
            del local_val_loader
        _clear_reward_cuda_state()


def evaluate_static_build_and_forward(
    *,
    model: Optional[nn.Module] = None,
    build_fn: Optional[Callable[[], nn.Module]] = None,
    cfg: EvalConfig = EvalConfig(),
    seed_accuracy_baseline: Optional[float] = None,
    backbone_model_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    device = str(cfg.device)
    mdl = model
    built_ok = False
    forward_ok = False
    forward_shape_ok = False
    latency_ms = None
    params_m = None
    estimated_total_seconds = 0.0
    started_at = time.time()
    try:
        if mdl is None and build_fn is not None:
            mdl = build_fn()
        if mdl is None or not isinstance(mdl, nn.Module):
            components = compute_cv_reward_simple(
                built_ok=False,
                forward_shape_ok=False,
                backward_ok=False,
                loss_drop_ok=False,
                val_metric=None,
                val_metric_baseline=None,
                latency_ms=None,
                params_m=None,
                flops_g=None,
                critic_score=None,
                kl_div=cfg.kl_div,
                weights=cfg.weights,
            )
            return {
                "reward": components["reward"],
                "components": components,
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "static_only": True,
            }

        mdl.to(device)
        params_m = _count_params_m(mdl)
        built_ok = True
        forward_result = _quick_forward(
            mdl,
            cfg.input_shape,
            device=device,
            n_classes=cfg.n_classes,
        )
        forward_ok = bool(forward_result.get("forward_ok"))
        forward_shape_ok = bool(forward_result.get("forward_shape_ok"))
        if cfg.measure_latency:
            latency_ms = forward_result.get("latency_ms")
        estimated_total_seconds = max(0.0, time.time() - started_at)
        components = compute_cv_reward_simple(
            built_ok=built_ok,
            forward_shape_ok=forward_shape_ok,
            backward_ok=False,
            loss_drop_ok=False,
            val_metric=None,
            val_metric_baseline=None,
            latency_ms=latency_ms,
            params_m=params_m,
            flops_g=None,
            critic_score=None,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )
        return {
            "reward": components["reward"],
            "components": components,
            **_base_eval_result(
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=cfg.eval_limit_seconds,
                backbone_model_names=backbone_model_names,
            ),
            "built_ok": built_ok,
            "forward_ok": forward_ok,
            "forward_shape_ok": forward_shape_ok,
            "latency_ms": latency_ms,
            "params_m": params_m,
            "estimated_total_seconds": estimated_total_seconds,
            "static_only": True,
        }
    finally:
        if mdl is not None:
            try:
                mdl.to("cpu")
            except Exception:
                pass
            del mdl
        _clear_reward_cuda_state()


def _safe_exec_code(code: str) -> Dict[str, Any]:
    """
    Execute user-provided model code with torch/nn in scope and return the namespace.
    Expect class Net(in_shape, out_shape, prm, device) to exist.
    """
    ns: Dict[str, Any] = {"__builtins__": __builtins__, "torch": torch, "nn": nn}
    exec(code, ns, ns)
    return ns


def build_fn_from_code(
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Dict[str, Any],
    device_str: str = "cpu",
) -> Callable[[], nn.Module]:
    """
    Returns a builder function that instantiates Net(in_shape, out_shape, prm, device).
    """
    ns = _safe_exec_code(code)
    Net = ns.get("Net", None)
    if not isinstance(Net, type):
        raise RuntimeError("Net class not found or invalid in provided code.")
    device = torch.device(device_str)

    def _builder() -> nn.Module:
        return Net(in_shape, out_shape, prm, device)

    return _builder


def _serialize_eval_cfg(cfg: EvalConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _deserialize_eval_cfg(cfg_payload: Optional[Dict[str, Any]]) -> EvalConfig:
    if cfg_payload is None:
        return EvalConfig()
    return EvalConfig(**cfg_payload)


def _loader_cache_key(cfg: EvalConfig) -> Tuple[Any, ...]:
    return (
        tuple(cfg.input_shape),
        cfg.default_batch_size,
        cfg.train_subset_size,
        cfg.val_subset_size,
        cfg.data_root,
        cfg.download,
        bool(getattr(cfg, "full_test_acc", False)),
    )


def _get_or_create_eval_worker_pool() -> _EvalWorkerPool:
    global _EVAL_WORKER_POOL
    desired_plan = get_reward_worker_plan()
    desired_signature = _reward_worker_plan_signature(desired_plan)
    if _EVAL_WORKER_POOL is not None:
        diagnostics = _EVAL_WORKER_POOL.diagnostics()
        workers = diagnostics.get("workers", [])
        current_signature = _EVAL_WORKER_POOL.plan_signature()
        if current_signature != desired_signature:
            print(
                "[Reward Worker Pool] Reconfigure "
                f"old_mode={diagnostics.get('mode')} "
                f"new_mode={desired_plan.get('mode')} "
                f"old_reward_gpu_indices={diagnostics.get('reward_gpu_indices')} "
                f"new_reward_gpu_indices={desired_plan.get('reward_gpu_indices')}"
            )
            _EVAL_WORKER_POOL.close()
            _EVAL_WORKER_POOL = None
        elif not workers or not all(bool(worker.get("alive", False)) for worker in workers):
            _EVAL_WORKER_POOL.close()
            _EVAL_WORKER_POOL = None
    if _EVAL_WORKER_POOL is None:
        _EVAL_WORKER_POOL = _EvalWorkerPool(plan=desired_plan)
    return _EVAL_WORKER_POOL


def _await_eval_worker_pool(*, require_gpu: bool, timeout: float) -> Optional[_EvalWorkerPool]:
    if not require_gpu:
        return _get_or_create_eval_worker_pool()
    if not torch.cuda.is_available():
        return None

    deadline = time.time() + max(0.0, float(timeout))
    logged_wait = False
    while True:
        pool = _get_or_create_eval_worker_pool()
        diagnostics = pool.diagnostics()
        has_gpu_worker = any(worker.get("assigned_gpu") is not None for worker in diagnostics.get("workers", []))
        if has_gpu_worker:
            return pool

        remaining = max(0.0, deadline - time.time())
        if remaining <= 0.0:
            return None
        if not logged_wait:
            print(
                "[Reward Worker Pool] Waiting "
                f"mode={diagnostics.get('mode')} "
                f"reason={diagnostics.get('reason', '')!r} "
                f"timeout_seconds={timeout:.0f}"
            )
            logged_wait = True
        shutdown_eval_worker()
        time.sleep(min(5.0, remaining))


def _timeout_waiting_for_gpu_result(
    *,
    error: str,
    preview: Dict[str, Any],
    seed_accuracy_baseline: Optional[float],
) -> Dict[str, Any]:
    effective_cfg = preview["effective_cfg"]
    return _timeout_eval_result(
        error=error,
        estimated_total_seconds=float(preview["request_timeout"]),
        eval_limit_seconds=_effective_eval_limit_seconds(effective_cfg),
        seed_accuracy_baseline=seed_accuracy_baseline,
        backbone_model_names=preview["backbone_model_names"],
        kl_div=effective_cfg.kl_div,
        weights=effective_cfg.weights,
    )


def shutdown_eval_worker() -> None:
    global _EVAL_WORKER_POOL
    if _EVAL_WORKER_POOL is None:
        return
    info = _EVAL_WORKER_POOL.diagnostics()
    print(
        "[Reward Worker Pool] Shutdown "
        f"mode={info['mode']} "
        f"pool_size={info['pool_size']} "
        f"workers_per_gpu={info.get('workers_per_gpu', 1)} "
        f"per_gpu_worker_counts={info.get('per_gpu_worker_counts', [])} "
        f"worker_pids={info['worker_pids']} "
        f"total_rss_gib={info['total_rss_gib']:.2f}"
    )
    _EVAL_WORKER_POOL.close()
    _EVAL_WORKER_POOL = None


def get_eval_worker_diagnostics() -> Optional[Dict[str, Any]]:
    if _EVAL_WORKER_POOL is None:
        return None
    return _EVAL_WORKER_POOL.diagnostics()


def _reward_worker_summary(worker: Dict[str, Any]) -> str:
    return (
        f"slot={worker.get('slot')} "
        f"pid={worker.get('pid')} "
        f"assigned_gpu={worker.get('assigned_gpu')} "
        f"assigned_cuda_visible_device={worker.get('assigned_cuda_visible_device')!r} "
        f"physical_gpu_index={worker.get('physical_gpu_index')} "
        f"physical_gpu_bus_id={str(worker.get('physical_gpu_bus_id') or '')!r} "
        f"physical_binding_verified={bool(worker.get('physical_binding_verified', False))} "
        f"alive={bool(worker.get('alive', False))}"
    )


def _validate_reward_worker_binding(worker: Dict[str, Any]) -> Optional[str]:
    slot = worker.get("slot")
    pid = worker.get("pid")
    assigned_gpu = worker.get("assigned_gpu")
    if not bool(worker.get("alive", False)):
        return f"slot={slot} pid={pid} is not alive"
    if assigned_gpu is None:
        if str(worker.get("worker_device", "")) != "cpu":
            return f"slot={slot} pid={pid} expected cpu worker, got {worker.get('worker_device')!r}"
        return None

    if str(worker.get("worker_device", "")) != "cuda:0":
        return f"slot={slot} pid={pid} expected worker_device='cuda:0', got {worker.get('worker_device')!r}"
    if not bool(worker.get("cuda_available", False)):
        return f"slot={slot} pid={pid} reported cuda_available=False"
    if int(worker.get("cuda_device_count", 0) or 0) != 1:
        return (
            f"slot={slot} pid={pid} expected exactly one visible CUDA device, "
            f"got {worker.get('cuda_device_count')!r}"
        )
    if not bool(worker.get("physical_binding_verified", False)):
        return f"slot={slot} pid={pid} could not verify physical GPU binding"

    visible_device_token = str(worker.get("assigned_cuda_visible_device") or "").strip()
    physical_gpu_index = worker.get("physical_gpu_index")
    if visible_device_token.isdigit():
        if physical_gpu_index is None:
            return (
                f"slot={slot} pid={pid} missing physical_gpu_index for "
                f"assigned token {visible_device_token!r}"
            )
        if int(physical_gpu_index) != int(visible_device_token):
            return (
                f"slot={slot} pid={pid} bound physical GPU {physical_gpu_index!r}, "
                f"expected token {visible_device_token!r}"
            )
    if not str(worker.get("physical_gpu_bus_id") or "").strip():
        return f"slot={slot} pid={pid} missing physical_gpu_bus_id"
    if not str(worker.get("physical_gpu_uuid") or "").strip():
        return f"slot={slot} pid={pid} missing physical_gpu_uuid"
    return None


def prewarm_eval_workers(*, timeout_seconds: float = 60.0, require_gpu: bool = False) -> Dict[str, Any]:
    desired_plan = get_reward_worker_plan()
    planned_gpu_workers = int(
        sum(max(0, int(count)) for count in desired_plan.get("per_gpu_worker_counts", []) or [])
    )
    should_require_gpu = bool(require_gpu or planned_gpu_workers > 0 or desired_plan.get("mode") == "reward_gpu_wait")
    print(
        "[Reward Worker Warmup] start "
        f"mode={desired_plan.get('mode')} "
        f"require_gpu={should_require_gpu} "
        f"planned_gpu_workers={planned_gpu_workers} "
        f"per_gpu_worker_counts={desired_plan.get('per_gpu_worker_counts', [])} "
        f"reward_gpu_indices={desired_plan.get('reward_gpu_indices', [])} "
        f"reward_gpu_tokens={desired_plan.get('reward_gpu_tokens', [])} "
        f"timeout_seconds={timeout_seconds:.0f}"
    )
    pool = _await_eval_worker_pool(require_gpu=should_require_gpu, timeout=float(timeout_seconds))
    if pool is None:
        raise PersistentEvalWorkerError(
            "Reward worker warmup timed out while waiting for GPU workers "
            f"(timeout_seconds={timeout_seconds:.0f}, mode={desired_plan.get('mode')!r})"
        )
    diagnostics = pool.diagnostics()
    workers = list(diagnostics.get("workers", []) or [])
    failure_reasons = [
        reason
        for reason in (_validate_reward_worker_binding(worker) for worker in workers)
        if reason is not None
    ]
    if should_require_gpu and not any(worker.get("assigned_gpu") is not None for worker in workers):
        failure_reasons.append("warmup completed without any GPU reward worker")

    worker_summaries = [_reward_worker_summary(worker) for worker in workers]
    if failure_reasons:
        print(
            "[Reward Worker Warmup] failure "
            f"mode={diagnostics.get('mode')} "
            f"workers={worker_summaries} "
            f"reasons={failure_reasons}"
        )
        shutdown_eval_worker()
        raise PersistentEvalWorkerError(
            "Reward worker warmup failed: " + "; ".join(str(reason) for reason in failure_reasons)
        )

    print(
        "[Reward Worker Warmup] end "
        f"mode={diagnostics.get('mode')} "
        f"pool_size={diagnostics.get('pool_size')} "
        f"verified_workers={len(workers)} "
        f"worker_pids={diagnostics.get('worker_pids', [])} "
        f"workers={worker_summaries}"
    )
    return diagnostics


def _worker_cuda_memory_gib() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated() / float(1024 ** 3)
        reserved = torch.cuda.memory_reserved() / float(1024 ** 3)
        return allocated, reserved
    except RuntimeError:
        return None, None


def _worker_visible_cuda_memory_gib() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = float(free_bytes) / float(1024 ** 3)
        total_gib = float(total_bytes) / float(1024 ** 3)
        used_gib = float(total_bytes - free_bytes) / float(1024 ** 3)
        return free_gib, used_gib, total_gib
    except RuntimeError:
        return None, None, None


def _query_nvidia_smi_rows(query_type: str, field_names: list[str]) -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query_type}={','.join(field_names)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return []
    output = str(completed.stdout or "").strip()
    if not output:
        return []
    reader = csv.reader(output.splitlines())
    rows: list[dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        normalized = [str(value).strip() for value in row]
        if len(normalized) != len(field_names):
            continue
        rows.append({name: value for name, value in zip(field_names, normalized)})
    return rows


def _prime_worker_cuda_context(worker_device: str) -> None:
    if not worker_device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    try:
        probe = torch.empty(1, device=worker_device)
        del probe
        torch.cuda.synchronize()
    except Exception:
        pass


def _resolve_worker_physical_gpu_info(
    *,
    worker_device: str,
    assigned_cuda_visible_device: Optional[str],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "physical_gpu_index": None,
        "physical_gpu_uuid": "",
        "physical_gpu_bus_id": "",
        "physical_gpu_name": "",
        "physical_binding_verified": False,
    }
    if not worker_device.startswith("cuda") or not torch.cuda.is_available():
        return info

    _prime_worker_cuda_context(worker_device)
    pid = str(os.getpid())
    matched_uuid = ""
    for _attempt in range(10):
        app_rows = _query_nvidia_smi_rows("compute-apps", ["pid", "gpu_uuid"])
        matched_row = next((row for row in app_rows if str(row.get("pid", "")).strip() == pid), None)
        if matched_row is not None:
            matched_uuid = str(matched_row.get("gpu_uuid") or "").strip()
            if matched_uuid:
                break
        time.sleep(0.1)
    gpu_rows = _query_nvidia_smi_rows("gpu", ["index", "uuid", "pci.bus_id", "name"])
    if matched_uuid:
        matched_gpu_row = next(
            (row for row in gpu_rows if str(row.get("uuid") or "").strip() == matched_uuid),
            None,
        )
        if matched_gpu_row is not None:
            info["physical_gpu_uuid"] = matched_uuid
            info["physical_gpu_index"] = int(matched_gpu_row["index"])
            info["physical_gpu_bus_id"] = str(matched_gpu_row.get("pci.bus_id") or "")
            info["physical_gpu_name"] = str(matched_gpu_row.get("name") or "")
            info["physical_binding_verified"] = True
            return info

    visible_device_token = (
        None if assigned_cuda_visible_device is None else str(assigned_cuda_visible_device).strip()
    )
    if visible_device_token and visible_device_token.isdigit():
        matched_gpu_row = next(
            (row for row in gpu_rows if str(row.get("index") or "").strip() == visible_device_token),
            None,
        )
        if matched_gpu_row is not None:
            info["physical_gpu_index"] = int(matched_gpu_row["index"])
            info["physical_gpu_uuid"] = str(matched_gpu_row.get("uuid") or "")
            info["physical_gpu_bus_id"] = str(matched_gpu_row.get("pci.bus_id") or "")
            info["physical_gpu_name"] = str(matched_gpu_row.get("name") or "")
    return info


def _log_reward_worker_memory(
    stage: str,
    *,
    request: Optional[Dict[str, Any]],
    assigned_gpu: Optional[int],
    worker_device: str,
    reward_batch_index: Optional[int],
    completion_index: Optional[int],
    error: Optional[str] = None,
) -> None:
    cuda_allocated_gib, cuda_reserved_gib = _worker_cuda_memory_gib()
    cuda_free_gib, cuda_used_gib, cuda_total_gib = _worker_visible_cuda_memory_gib()
    normalized_error = None if not error else " ".join(str(error).split())
    message = (
        "[Reward Worker Memory] "
        f"stage={stage} "
        f"pid={os.getpid()} "
        f"worker_slot={request.get('worker_slot') if isinstance(request, dict) else None} "
        f"assigned_gpu={assigned_gpu} "
        f"worker_device={worker_device} "
        f"reward_batch_index={reward_batch_index} "
        f"completion_index={completion_index} "
        f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
        f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
        f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
        f"cuda_free_gib={_format_mem_value(cuda_free_gib)} "
        f"cuda_used_gib={_format_mem_value(cuda_used_gib)} "
        f"cuda_total_gib={_format_mem_value(cuda_total_gib)} "
        f"torch_home={os.environ.get('TORCH_HOME', '')!r}"
    )
    if normalized_error is not None:
        message += f" error={normalized_error!r}"
    print(message)


def _persistent_eval_worker_entry(
    conn,
    assigned_gpu: Optional[int],
    assigned_cuda_visible_device: Optional[str],
) -> None:
    worker_device = "cuda:0" if assigned_gpu is not None else "cpu"
    try:
        physical_gpu_info = _resolve_worker_physical_gpu_info(
            worker_device=worker_device,
            assigned_cuda_visible_device=assigned_cuda_visible_device,
        )
        conn.send(
            {
                "cmd": "worker_ready",
                "pid": os.getpid(),
                "assigned_gpu": assigned_gpu,
                "assigned_cuda_visible_device": assigned_cuda_visible_device,
                "worker_device": worker_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "rss_gib": float(_read_process_rss_gib() or 0.0),
                "torch_home": os.environ.get("TORCH_HOME", ""),
                "physical_gpu_index": physical_gpu_info.get("physical_gpu_index"),
                "physical_gpu_uuid": physical_gpu_info.get("physical_gpu_uuid", ""),
                "physical_gpu_bus_id": physical_gpu_info.get("physical_gpu_bus_id", ""),
                "physical_gpu_name": physical_gpu_info.get("physical_gpu_name", ""),
                "physical_binding_verified": bool(
                    physical_gpu_info.get("physical_binding_verified", False)
                ),
            }
        )
        _persistent_eval_worker_loop(conn, worker_device=worker_device, assigned_gpu=assigned_gpu)
    except Exception as exc:
        try:
            conn.send(
                {
                    "cmd": "worker_init_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def evaluate_code_and_reward(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
    reward_batch_index: Optional[int] = None,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    preview = _preview_eval_request(
        code=code,
        in_shape=in_shape,
        out_shape=out_shape,
        prm=prm,
        device=device,
        seed_accuracy_baseline=seed_accuracy_baseline,
        cfg=cfg,
    )
    prevalidated_result = preview.get("prevalidated_result")
    if prevalidated_result is not None:
        return prevalidated_result

    worker_pool = _await_eval_worker_pool(
        require_gpu=bool(preview["requires_gpu"]),
        timeout=float(preview["request_timeout"]),
    )
    if worker_pool is None:
        return _timeout_waiting_for_gpu_result(
            error="PersistentEvalWorkerError: timed out waiting for available GPU reward worker",
            preview=preview,
            seed_accuracy_baseline=seed_accuracy_baseline,
        )
    entry = _prepare_eval_request_entry(
        code=code,
        in_shape=in_shape,
        out_shape=out_shape,
        prm=prm,
        device=device,
        val_metric_baseline=val_metric_baseline,
        seed_accuracy_baseline=seed_accuracy_baseline,
        cfg=cfg,
        reward_batch_index=reward_batch_index,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
        worker_pool=worker_pool,
    )
    return worker_pool.request_entry(entry, timeout=float(entry["request_timeout"]))


def _prepare_eval_request_entry(
    *,
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Optional[Dict[str, Any]],
    device: str,
    val_metric_baseline: Optional[float],
    seed_accuracy_baseline: Optional[float],
    cfg: Optional[EvalConfig],
    reward_batch_index: Optional[int],
    completion_index: Optional[int],
    batch_last_item: bool,
    worker_pool: Optional["_EvalWorkerPool"] = None,
) -> Dict[str, Any]:
    worker_pool = worker_pool or _get_or_create_eval_worker_pool()
    worker_device = worker_pool.primary_device()
    effective_cfg = _build_effective_eval_cfg(
        cfg=cfg,
        prm=prm,
        device=worker_device,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    return {
        "payload": {
            "cmd": "evaluate",
            "code": code,
            "in_shape": tuple(in_shape),
            "out_shape": tuple(out_shape),
            "prm": prm,
            "device": worker_device,
            "val_metric_baseline": val_metric_baseline,
            "seed_accuracy_baseline": seed_accuracy_baseline,
            "cfg": _serialize_eval_cfg(effective_cfg),
            "reward_batch_index": reward_batch_index,
            "completion_index": completion_index,
            "batch_last_item": batch_last_item,
        },
        "effective_cfg": effective_cfg,
        "request_timeout": _request_timeout_seconds(effective_cfg),
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "backbone_model_names": _extract_backbone_model_names_from_code(code),
    }


def evaluate_code_and_reward_batch(
    request_specs: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if not request_specs:
        return []
    indexed_previews = []
    indexed_specs = []
    indexed_results: list[Optional[Dict[str, Any]]] = [None] * len(request_specs)
    require_gpu = False
    max_timeout = 0.0

    for index, spec in enumerate(request_specs):
        preview = _preview_eval_request(
            code=str(spec["code"]),
            in_shape=tuple(spec.get("in_shape", (8, 3, 32, 32))),
            out_shape=tuple(spec.get("out_shape", (10,))),
            prm=spec.get("prm"),
            device=str(spec.get("device", "cpu")),
            seed_accuracy_baseline=spec.get("seed_accuracy_baseline"),
            cfg=spec.get("cfg"),
        )
        if preview.get("prevalidated_result") is not None:
            indexed_results[index] = preview["prevalidated_result"]
            continue
        indexed_previews.append((index, preview))
        indexed_specs.append((index, spec))
        require_gpu = require_gpu or bool(preview["requires_gpu"])
        max_timeout = max(max_timeout, float(preview["request_timeout"]))

    if not indexed_specs:
        return [
            result
            if result is not None
            else _empty_eval_result(error="PersistentEvalWorkerError: missing prevalidated reward result")
            for result in indexed_results
        ]

    worker_pool = _await_eval_worker_pool(
        require_gpu=require_gpu,
        timeout=max_timeout,
    )
    if worker_pool is None:
        for index, preview in indexed_previews:
            indexed_results[index] = _timeout_waiting_for_gpu_result(
                error="PersistentEvalWorkerError: timed out waiting for available GPU reward worker",
                preview=preview,
                seed_accuracy_baseline=request_specs[index].get("seed_accuracy_baseline"),
            )
        return [
            result
            if result is not None
            else _empty_eval_result(error="PersistentEvalWorkerError: missing prevalidated reward result")
            for result in indexed_results
        ]

    entries = [
        _prepare_eval_request_entry(
            code=str(spec["code"]),
            in_shape=tuple(spec.get("in_shape", (8, 3, 32, 32))),
            out_shape=tuple(spec.get("out_shape", (10,))),
            prm=spec.get("prm"),
            device=str(spec.get("device", "cpu")),
            val_metric_baseline=spec.get("val_metric_baseline"),
            seed_accuracy_baseline=spec.get("seed_accuracy_baseline"),
            cfg=spec.get("cfg"),
            reward_batch_index=spec.get("reward_batch_index"),
            completion_index=spec.get("completion_index"),
            batch_last_item=bool(spec.get("batch_last_item", False)),
            worker_pool=worker_pool,
        )
        for _, spec in indexed_specs
    ]
    timeout = max(float(entry["request_timeout"]) for entry in entries)
    mapped_results = worker_pool.map_entries(entries, timeout=timeout)
    for (index, _spec), result in zip(indexed_specs, mapped_results):
        indexed_results[index] = result

    return [
        result
        if result is not None
        else _empty_eval_result(error="PersistentEvalWorkerError: missing prevalidated reward result")
        for result in indexed_results
    ]


def _persistent_eval_worker_loop(conn, *, worker_device: str, assigned_gpu: Optional[int]) -> None:
    loader_cache: Dict[Tuple[Any, ...], Tuple[DataLoader, DataLoader]] = {}
    last_reward_batch_index = None
    last_completion_index = None
    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break

            if not isinstance(request, dict):
                conn.send(_empty_eval_result(error="Persistent eval worker received a non-dict request"))
                continue

            if request.get("cmd") == "shutdown":
                break
            if request.get("cmd") != "evaluate":
                conn.send(_empty_eval_result(error=f"Persistent eval worker received unknown command: {request.get('cmd')!r}"))
                continue

            cfg = _deserialize_eval_cfg(request.get("cfg"))
            train_loader = None
            val_loader = None
            if (not _is_formal_nn_eval_enabled(cfg)) and (not _is_static_only_eval_enabled(cfg)):
                loader_key = _loader_cache_key(cfg)
                if loader_key not in loader_cache:
                    loader_cache[loader_key] = _build_cifar10_loaders(cfg)
                train_loader, val_loader = loader_cache[loader_key]
            reward_batch_index = request.get("reward_batch_index")
            completion_index = request.get("completion_index")
            last_reward_batch_index = reward_batch_index
            last_completion_index = completion_index
            try:
                if worker_device.startswith("cuda") and torch.cuda.is_available():
                    _clear_reward_cuda_state()
                result = _evaluate_code_and_reward_direct(
                    request["code"],
                    in_shape=tuple(request["in_shape"]),
                    out_shape=tuple(request["out_shape"]),
                    prm=request.get("prm"),
                    device=request.get("device", "cpu"),
                    val_metric_baseline=request.get("val_metric_baseline"),
                    seed_accuracy_baseline=request.get("seed_accuracy_baseline"),
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if _is_cuda_oom_error_message(error_text) or _is_fatal_cuda_worker_error(error_text):
                    _log_reward_worker_memory(
                        "error",
                        request=request,
                        assigned_gpu=assigned_gpu,
                        worker_device=worker_device,
                        reward_batch_index=reward_batch_index,
                        completion_index=completion_index,
                        error=error_text,
                    )
                result = _empty_eval_result(
                    reward=-1.0,
                    error=error_text,
                    seed_accuracy_baseline=request.get("seed_accuracy_baseline"),
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=_extract_backbone_model_names_from_code(request.get("code", "")),
                )
                result["components"]["reward"] = -1.0
            worker_restart_requested = _is_fatal_cuda_worker_error(result.get("error"))
            result["assigned_gpu"] = assigned_gpu
            result["worker_device"] = worker_device
            result["worker_slot"] = request.get("worker_slot", None)
            result["worker_restart_requested"] = worker_restart_requested
            conn.send(result)
            if worker_restart_requested:
                break
    finally:
        loader_cache.clear()
        _clear_reward_cuda_state()
        try:
            conn.close()
        except Exception:
            pass

def _evaluate_code_and_reward_direct(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """
    The original evaluation logic (renamed).
    """
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9}
    defaults = {"lr": 1e-2, "momentum": 0.9, "batch": 32, "epoch": 1, "transform": None}
    prm = {**defaults, **prm}
    backbone_model_names = _extract_backbone_model_names_from_code(code)
    cfg = _build_effective_eval_cfg(
        cfg=cfg,
        prm=prm,
        device=device,
        in_shape=in_shape,
        out_shape=out_shape,
    )

    if _is_static_only_eval_enabled(cfg):
        try:
            builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            failure_result = _empty_eval_result(
                error=error_msg,
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=cfg.eval_limit_seconds,
                backbone_model_names=backbone_model_names,
            )
            failure_result["frozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="frozen",
                backbone_frozen=True,
            )
            failure_result["static_only"] = True
            return failure_result
        try:
            static_result = evaluate_static_build_and_forward(
                build_fn=builder,
                cfg=cfg,
                seed_accuracy_baseline=seed_accuracy_baseline,
                backbone_model_names=backbone_model_names,
            )
            static_result["frozen_eval"] = _nested_eval_payload(
                static_result,
                eval_mode="frozen",
                backbone_frozen=True,
            )
            return static_result
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            failure_result = _empty_eval_result(
                error=error_msg,
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=cfg.eval_limit_seconds,
                backbone_model_names=backbone_model_names,
            )
            failure_result["frozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="frozen",
                backbone_frozen=True,
            )
            failure_result["static_only"] = True
            return failure_result

    if _is_formal_nn_eval_enabled(cfg):
        try:
            frozen_result, frozen_cfg, _frozen_prm = _run_formal_eval_with_backoff(
                code=code,
                prm=prm,
                cfg=cfg,
                freeze_backbones=True,
                seed_accuracy_baseline=seed_accuracy_baseline,
                backbone_model_names=backbone_model_names,
            )
            if frozen_result.get("error"):
                return frozen_result

            return _merge_dual_eval_results(
                frozen_result=frozen_result,
                unfrozen_result=None,
                cfg=frozen_cfg,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            error_stage = None
            error_context = None
            error_hint = None
            if isinstance(e, FormalEvalTraceError):
                error_type = e.error_type
                error_msg = e.error_message
                error_stage = e.stage
                error_context = dict(e.context)
                error_hint = e.hint
            failure_result = _empty_eval_result(
                error=error_msg,
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=_effective_eval_limit_seconds(cfg),
                backbone_model_names=backbone_model_names,
            )
            _apply_error_trace(
                failure_result,
                error_type=error_type,
                error_stage=error_stage,
                error_context=error_context,
                error_hint=error_hint,
            )
            failure_result["frozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="frozen",
                backbone_frozen=True,
            )
            return failure_result

    try:
        builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as e:
        # Pass through error type so reward_fn can assign layered partial rewards
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        failure_result = _empty_eval_result(
            error=error_msg,
            seed_accuracy_baseline=seed_accuracy_baseline,
            eval_limit_seconds=cfg.eval_limit_seconds,
            backbone_model_names=backbone_model_names,
        )
        failure_result["frozen_eval"] = _nested_eval_payload(
            failure_result,
            eval_mode="frozen",
            backbone_frozen=True,
        )
        return failure_result
    try:
        frozen_result = evaluate_and_reward(
            build_fn=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
            backbone_model_names=backbone_model_names,
            freeze_backbones=True,
        )
        return _merge_dual_eval_results(
            frozen_result=frozen_result,
            unfrozen_result=None,
            cfg=cfg,
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        failure_result = _empty_eval_result(
            error=error_msg,
            seed_accuracy_baseline=seed_accuracy_baseline,
            eval_limit_seconds=cfg.eval_limit_seconds,
            backbone_model_names=backbone_model_names,
        )
        failure_result["frozen_eval"] = _nested_eval_payload(
            failure_result,
            eval_mode="frozen",
            backbone_frozen=True,
        )
        return failure_result


atexit.register(shutdown_eval_worker)

if __name__ == "__main__":
    demo_code = r'''
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
                           
        self.conv1 = nn.Conv2d(in_shape[0], prm['conv1_filters'], kernel_size=prm['conv1_kernel'], stride=prm['conv1_stride'])
        self.bn1 = nn.BatchNorm2d(prm['conv1_filters']) if prm['use_batchnorm'] else None
        self.conv2 = nn.Conv2d(prm['conv1_filters'], prm['conv2_filters'], kernel_size=prm['conv2_kernel'])
        self.bn2 = nn.BatchNorm2d(prm['conv2_filters']) if prm['use_batchnorm'] else None
        self.conv3 = nn.Conv2d(prm['conv2_filters'], prm['conv3_filters'], kernel_size=prm['conv3_kernel'])
        self.bn3 = nn.BatchNorm2d(prm['conv3_filters']) if prm['use_batchnorm'] else None
        self.conv4 = nn.Conv2d(prm['conv3_filters'], prm['conv4_filters'], kernel_size=prm['conv4_kernel'])
        self.bn4 = nn.BatchNorm2d(prm['conv4_filters']) if prm['use_batchnorm'] else None
        self.conv5 = nn.Conv2d(prm['conv4_filters'], prm['conv5_filters'], kernel_size=prm['conv5_kernel'])
        self.bn5 = nn.BatchNorm2d(prm['conv5_filters']) if prm['use_batchnorm'] else None
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(prm['fc1_neurons'], prm['fc2_neurons'])
        self.fc2 = nn.Linear(prm['fc2_neurons'], out_shape)
        
                       
        self.dropout = nn.Dropout(p=prm['dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x'''
    res = evaluate_code_and_reward(
        demo_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={"lr": 1e-2, "momentum": 0.9, "dropout": 0.5},
        device="cpu",
        val_metric_baseline=0.10,
    )
    print(res)

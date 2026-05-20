import gc
import os
import subprocess
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


_AUX_GPU_TOKENS_ENV = "NNGPT_AUX_GPU_TOKENS"
_TRAIN_GPU_TOKENS_ENV = "NNGPT_TRAIN_GPU_TOKENS"
_LEGACY_REWARD_GPU_TOKENS_ENV = "NNGPT_REWARD_GPU_TOKENS"
_NNEVAL_GPU_TOKENS_ENV = "NNGPT_NNEVAL_GPU_TOKENS"

_NNEVAL_WORKER_POOL: Optional["_NNEvalWorkerPool"] = None


class PersistentNNEvalWorkerError(RuntimeError):
    pass


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_is_set(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw != ""


def _configured_cuda_device_tokens(env_name: str) -> Optional[List[str]]:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _visible_cuda_device_tokens() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        raw = raw.strip()
        if raw in {"", "-1"}:
            return []
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if tokens:
            return tokens
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(int(torch.cuda.device_count()))]


def _cuda_device_memory_snapshot_gib(device_index: int, device_token: str) -> Dict[str, Any]:
    info = {
        "device_index": int(device_index),
        "device_token": str(device_token),
        "free_gib": None,
        "used_gib": None,
        "total_gib": None,
        "gpu_name": "",
    }
    if not torch.cuda.is_available():
        return info
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(int(device_index))
        info["free_gib"] = float(free_bytes) / float(1024 ** 3)
        info["total_gib"] = float(total_bytes) / float(1024 ** 3)
        info["used_gib"] = float(total_bytes - free_bytes) / float(1024 ** 3)
    except Exception:
        pass
    try:
        info["gpu_name"] = str(torch.cuda.get_device_name(int(device_index)))
    except Exception:
        pass
    return info


def _dynamic_workers_for_snapshot(
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
        return max(0, min(int(max_workers_per_gpu), max(1, int(min_workers_per_gpu))))
    available_gib = max(0.0, float(free_gib) - float(reserved_headroom_gib))
    worker_budget_gib = max(0.5, float(worker_budget_gib))
    worker_count = int(available_gib // worker_budget_gib)
    if worker_count <= 0 and float(free_gib) >= float(reserved_headroom_gib) + (worker_budget_gib * 0.75):
        worker_count = 1
    if worker_count > 0:
        worker_count = max(worker_count, int(min_workers_per_gpu))
    return max(0, min(int(max_workers_per_gpu), int(worker_count)))


def _expand_gpu_assignments(
    visible_gpu_tokens: Sequence[str],
    per_gpu_worker_counts: Sequence[int],
) -> Tuple[List[int], List[str]]:
    assigned_indices: List[int] = []
    assigned_tokens: List[str] = []
    active_device_indices = [
        int(device_index)
        for device_index, worker_count in enumerate(per_gpu_worker_counts)
        if int(worker_count) > 0
    ]
    max_workers = max([int(count) for count in per_gpu_worker_counts] or [0])
    for round_index in range(max_workers):
        for device_index in active_device_indices:
            worker_count = int(per_gpu_worker_counts[device_index])
            if round_index >= worker_count:
                continue
            assigned_indices.append(int(device_index))
            assigned_tokens.append(str(visible_gpu_tokens[device_index]))
    return assigned_indices, assigned_tokens


def _cpu_worker_plan(*, reason: str, use_all_visible_gpus: bool) -> Dict[str, Any]:
    visible_gpu_tokens = _visible_cuda_device_tokens()
    visible_gpu_count = int(len(visible_gpu_tokens))
    return {
        "mode": "cpu_fallback",
        "reason": str(reason),
        "use_all_visible_gpus": bool(use_all_visible_gpus),
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_tokens": list(visible_gpu_tokens),
        "target_gpu_indices": [],
        "target_gpu_tokens": [],
        "eval_gpu_indices": [],
        "eval_gpu_tokens": [],
        "per_gpu_worker_counts": [0] * visible_gpu_count,
        "pool_size": 1,
        "workers_per_gpu": 0,
        "dynamic_scaling": False,
        "gpu_memory_snapshots": [],
        "worker_budget_gib": None,
        "reserved_headroom_gib": None,
    }


def _resolve_target_gpu_tokens(
    visible_gpu_tokens: Sequence[str],
    *,
    use_all_visible_gpus: bool,
) -> Tuple[List[str], str]:
    explicit_eval_tokens = _configured_cuda_device_tokens(_NNEVAL_GPU_TOKENS_ENV)
    if explicit_eval_tokens is not None:
        tokens = [token for token in explicit_eval_tokens if token in visible_gpu_tokens]
        return tokens, "explicit_nneval_gpu_tokens"

    if use_all_visible_gpus:
        return list(visible_gpu_tokens), "all_visible_gpus"

    aux_gpu_tokens = _configured_cuda_device_tokens(_AUX_GPU_TOKENS_ENV)
    if aux_gpu_tokens is None:
        aux_gpu_tokens = _configured_cuda_device_tokens(_LEGACY_REWARD_GPU_TOKENS_ENV)
    if aux_gpu_tokens:
        tokens = [token for token in aux_gpu_tokens if token in visible_gpu_tokens]
        return tokens, "aux_gpu_tokens"

    train_gpu_tokens = set(_configured_cuda_device_tokens(_TRAIN_GPU_TOKENS_ENV) or [])
    tokens = [token for token in visible_gpu_tokens if token not in train_gpu_tokens]
    if tokens:
        return tokens, "visible_minus_train_gpu_tokens"
    return list(visible_gpu_tokens), "visible_fallback"


def build_nneval_worker_plan(*, use_all_visible_gpus: bool = True) -> Dict[str, Any]:
    visible_gpu_tokens = _visible_cuda_device_tokens()
    visible_gpu_count = int(len(visible_gpu_tokens))
    if visible_gpu_count <= 0 or (not torch.cuda.is_available()):
        return _cpu_worker_plan(reason="no_visible_cuda_devices", use_all_visible_gpus=use_all_visible_gpus)

    target_gpu_tokens, source_reason = _resolve_target_gpu_tokens(
        visible_gpu_tokens,
        use_all_visible_gpus=use_all_visible_gpus,
    )
    if not target_gpu_tokens:
        return _cpu_worker_plan(reason="no_target_eval_gpus", use_all_visible_gpus=use_all_visible_gpus)

    target_gpu_indices = [
        int(index)
        for index, token in enumerate(visible_gpu_tokens)
        if str(token) in set(str(candidate) for candidate in target_gpu_tokens)
    ]
    if not target_gpu_indices:
        return _cpu_worker_plan(reason="configured_eval_gpus_unavailable", use_all_visible_gpus=use_all_visible_gpus)

    fixed_workers_override = _env_is_set("NNGPT_NNEVAL_WORKERS_PER_GPU")
    gpu_memory_snapshots = [
        _cuda_device_memory_snapshot_gib(index, visible_gpu_tokens[index])
        for index in target_gpu_indices
    ]

    per_gpu_worker_counts = [0] * visible_gpu_count
    if fixed_workers_override:
        workers_per_gpu = max(1, _safe_int_env("NNGPT_NNEVAL_WORKERS_PER_GPU", 1))
        for index in target_gpu_indices:
            per_gpu_worker_counts[int(index)] = int(workers_per_gpu)
        eval_gpu_indices, eval_gpu_tokens = _expand_gpu_assignments(visible_gpu_tokens, per_gpu_worker_counts)
        return {
            "mode": "fixed_worker_pool",
            "reason": "fixed_workers_override",
            "use_all_visible_gpus": bool(use_all_visible_gpus),
            "visible_gpu_count": visible_gpu_count,
            "visible_gpu_tokens": list(visible_gpu_tokens),
            "target_gpu_indices": list(target_gpu_indices),
            "target_gpu_tokens": list(target_gpu_tokens),
            "eval_gpu_indices": list(eval_gpu_indices),
            "eval_gpu_tokens": list(eval_gpu_tokens),
            "per_gpu_worker_counts": list(per_gpu_worker_counts),
            "pool_size": max(1, len(eval_gpu_indices)),
            "workers_per_gpu": int(workers_per_gpu),
            "dynamic_scaling": False,
            "gpu_memory_snapshots": list(gpu_memory_snapshots),
            "worker_budget_gib": None,
            "reserved_headroom_gib": None,
        }

    dynamic_scaling = _safe_bool_env("NNGPT_NNEVAL_ENABLE_DYNAMIC_SCALING", True)
    if not dynamic_scaling:
        for index in target_gpu_indices:
            per_gpu_worker_counts[int(index)] = 1
        eval_gpu_indices, eval_gpu_tokens = _expand_gpu_assignments(visible_gpu_tokens, per_gpu_worker_counts)
        return {
            "mode": "static_worker_pool",
            "reason": source_reason,
            "use_all_visible_gpus": bool(use_all_visible_gpus),
            "visible_gpu_count": visible_gpu_count,
            "visible_gpu_tokens": list(visible_gpu_tokens),
            "target_gpu_indices": list(target_gpu_indices),
            "target_gpu_tokens": list(target_gpu_tokens),
            "eval_gpu_indices": list(eval_gpu_indices),
            "eval_gpu_tokens": list(eval_gpu_tokens),
            "per_gpu_worker_counts": list(per_gpu_worker_counts),
            "pool_size": max(1, len(eval_gpu_indices)),
            "workers_per_gpu": 1,
            "dynamic_scaling": False,
            "gpu_memory_snapshots": list(gpu_memory_snapshots),
            "worker_budget_gib": None,
            "reserved_headroom_gib": None,
        }

    reserved_headroom_gib = max(0.0, _safe_float_env("NNGPT_NNEVAL_RESERVED_HEADROOM_GIB", 2.0))
    worker_budget_gib = max(0.5, _safe_float_env("NNGPT_NNEVAL_WORKER_BUDGET_GIB", 3.0))
    min_workers_per_gpu = max(0, _safe_int_env("NNGPT_NNEVAL_MIN_WORKERS_PER_GPU", 1))
    max_workers_per_gpu = max(
        min_workers_per_gpu,
        _safe_int_env("NNGPT_NNEVAL_MAX_WORKERS_PER_GPU", 2),
    )
    local_worker_counts = [
        _dynamic_workers_for_snapshot(
            snapshot,
            reserved_headroom_gib=reserved_headroom_gib,
            worker_budget_gib=worker_budget_gib,
            min_workers_per_gpu=min_workers_per_gpu,
            max_workers_per_gpu=max_workers_per_gpu,
        )
        for snapshot in gpu_memory_snapshots
    ]
    for target_index, worker_count in zip(target_gpu_indices, local_worker_counts):
        per_gpu_worker_counts[int(target_index)] = int(worker_count)
    if sum(per_gpu_worker_counts) <= 0:
        return {
            "mode": "gpu_wait",
            "reason": "awaiting_gpu_headroom",
            "use_all_visible_gpus": bool(use_all_visible_gpus),
            "visible_gpu_count": visible_gpu_count,
            "visible_gpu_tokens": list(visible_gpu_tokens),
            "target_gpu_indices": list(target_gpu_indices),
            "target_gpu_tokens": list(target_gpu_tokens),
            "eval_gpu_indices": [],
            "eval_gpu_tokens": [],
            "per_gpu_worker_counts": list(per_gpu_worker_counts),
            "pool_size": 0,
            "workers_per_gpu": 0,
            "dynamic_scaling": True,
            "gpu_memory_snapshots": list(gpu_memory_snapshots),
            "worker_budget_gib": float(worker_budget_gib),
            "reserved_headroom_gib": float(reserved_headroom_gib),
        }

    eval_gpu_indices, eval_gpu_tokens = _expand_gpu_assignments(visible_gpu_tokens, per_gpu_worker_counts)
    return {
        "mode": "dynamic_worker_pool",
        "reason": source_reason,
        "use_all_visible_gpus": bool(use_all_visible_gpus),
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_tokens": list(visible_gpu_tokens),
        "target_gpu_indices": list(target_gpu_indices),
        "target_gpu_tokens": list(target_gpu_tokens),
        "eval_gpu_indices": list(eval_gpu_indices),
        "eval_gpu_tokens": list(eval_gpu_tokens),
        "per_gpu_worker_counts": list(per_gpu_worker_counts),
        "pool_size": max(1, len(eval_gpu_indices)),
        "workers_per_gpu": max([int(count) for count in per_gpu_worker_counts] or [0]),
        "dynamic_scaling": True,
        "gpu_memory_snapshots": list(gpu_memory_snapshots),
        "worker_budget_gib": float(worker_budget_gib),
        "reserved_headroom_gib": float(reserved_headroom_gib),
    }


def _nngpt_worker_plan_signature(plan: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(plan.get("mode", "")),
        bool(plan.get("use_all_visible_gpus", True)),
        tuple(str(token) for token in plan.get("visible_gpu_tokens", []) or []),
        tuple(int(index) for index in plan.get("eval_gpu_indices", []) or []),
        tuple(str(token) for token in plan.get("eval_gpu_tokens", []) or []),
        tuple(int(count) for count in plan.get("per_gpu_worker_counts", []) or []),
    )


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


def _clear_cuda_state() -> None:
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


def _extract_accuracy(eval_results: Any) -> Tuple[Optional[str], Optional[float]]:
    checksum = None
    accuracy = None
    if isinstance(eval_results, tuple):
        if len(eval_results) >= 2:
            checksum = eval_results[0]
            accuracy = eval_results[1]
    elif isinstance(eval_results, dict):
        accuracy = eval_results.get("accuracy", eval_results.get("acc"))
        if accuracy is None:
            epochs_data = eval_results.get("epochs", [])
            if epochs_data:
                first_epoch = epochs_data[0] or {}
                accuracy = first_epoch.get("accuracy", first_epoch.get("acc"))
    if accuracy is not None:
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError):
            accuracy = None
    return (None if checksum is None else str(checksum), accuracy)


def _execute_nneval_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    from ab.gpt.util.Eval import Eval

    model_dir = str(payload["model_dir"])
    code_file = str(payload["code_file"])
    evaluator = Eval(
        model_source_package=model_dir,
        task=str(payload["task"]),
        dataset=str(payload["dataset"]),
        metric=str(payload["metric"]),
        prm=dict(payload["prm"] or {}),
        save_to_db=bool(payload.get("save_to_db", False)),
        prefix=payload.get("prefix"),
        save_path=payload.get("save_path"),
        use_ast_validation=payload.get("use_ast_validation"),
    )
    epoch_limit_minutes = payload.get("epoch_limit_minutes")
    if epoch_limit_minutes not in (None, "", 0):
        evaluator.epoch_limit_minutes = epoch_limit_minutes
    eval_results = evaluator.evaluate(code_file)
    checksum, accuracy = _extract_accuracy(eval_results)
    if accuracy is None:
        raise ValueError(f"Could not extract accuracy from evaluation results: {type(eval_results).__name__}")
    return {
        "success": True,
        "model_id": str(payload["model_id"]),
        "accuracy": float(accuracy),
        "checksum": checksum,
        "eval_args": evaluator.get_args(),
        "full_result": str(eval_results),
    }


def _request_timeout_seconds(payload: Dict[str, Any]) -> float:
    explicit_timeout = payload.get("request_timeout_seconds")
    if explicit_timeout not in (None, "", 0):
        return max(300.0, float(explicit_timeout))
    epoch_limit_minutes = payload.get("epoch_limit_minutes")
    if epoch_limit_minutes not in (None, "", 0):
        return max(300.0, float(epoch_limit_minutes) * 60.0 + 600.0)
    return float(max(300, _safe_int_env("NNGPT_NNEVAL_REQUEST_TIMEOUT_SECONDS", 3600)))


class _PersistentNNEvalWorkerSession:
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
            "ab.gpt.util.nneval_worker_bootstrap",
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
            "[NNEval Worker] Ready "
            f"slot={self._worker_slot} "
            f"pid={self._worker_info['pid']} "
            f"assigned_gpu={self._worker_info['assigned_gpu']} "
            f"assigned_cuda_visible_device={self._worker_info['assigned_cuda_visible_device']!r} "
            f"worker_device={self._worker_info['worker_device']} "
            f"cuda_visible_devices={self._worker_info['cuda_visible_devices']!r} "
            f"cuda_available={self._worker_info['cuda_available']} "
            f"cuda_device_count={self._worker_info['cuda_device_count']} "
            f"rss_gib={self._worker_info['rss_gib']:.2f}"
        )

    def request(self, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        if self._process.poll() is not None:
            raise PersistentNNEvalWorkerError("Persistent NNEval worker exited before handling a request")
        try:
            self._parent_conn.send(payload)
        except Exception as exc:
            raise PersistentNNEvalWorkerError(f"Persistent NNEval worker send failed: {exc}") from exc

        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentNNEvalWorkerError(f"Persistent NNEval worker timed out after {timeout:.0f} seconds")

        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentNNEvalWorkerError("Persistent NNEval worker closed its pipe unexpectedly") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentNNEvalWorkerError(f"Persistent NNEval worker recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentNNEvalWorkerError("Persistent NNEval worker returned a non-dict response")
        return response

    def _wait_for_ready(self, *, timeout: float) -> Dict[str, Any]:
        if self._process.poll() is not None:
            raise PersistentNNEvalWorkerError("Persistent NNEval worker exited during startup")
        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentNNEvalWorkerError(
                f"Persistent NNEval worker did not send a startup handshake within {timeout:.0f} seconds"
            )
        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentNNEvalWorkerError("Persistent NNEval worker closed its pipe during startup") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentNNEvalWorkerError(f"Persistent NNEval worker startup recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentNNEvalWorkerError("Persistent NNEval worker startup handshake was not a dict")
        if response.get("cmd") == "worker_init_error":
            self.close(force=True)
            raise PersistentNNEvalWorkerError(
                f"Persistent NNEval worker failed to initialize: {response.get('error', 'unknown error')}"
            )
        if response.get("cmd") != "worker_ready":
            self.close(force=True)
            raise PersistentNNEvalWorkerError(
                f"Persistent NNEval worker returned unexpected startup message: {response!r}"
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
        }
        missing = sorted(required_keys.difference(response))
        if missing:
            self.close(force=True)
            raise PersistentNNEvalWorkerError(
                f"Persistent NNEval worker startup handshake missing fields: {', '.join(missing)}"
            )
        cuda_visible_devices = str(response["cuda_visible_devices"]).strip()
        cuda_device_count = int(response["cuda_device_count"])
        if self._assigned_gpu is None:
            if str(response["worker_device"]) != "cpu":
                self.close(force=True)
                raise PersistentNNEvalWorkerError(
                    f"CPU NNEval worker returned unexpected device {response['worker_device']!r}"
                )
            if cuda_visible_devices not in {"", "-1"} or cuda_device_count != 0:
                self.close(force=True)
                raise PersistentNNEvalWorkerError(
                    "CPU NNEval worker unexpectedly reported CUDA visibility "
                    f"(CUDA_VISIBLE_DEVICES={response['cuda_visible_devices']!r}, count={response['cuda_device_count']})"
                )
        else:
            expected_device = None if self._assigned_cuda_visible_device is None else str(self._assigned_cuda_visible_device)
            if str(response["worker_device"]) != "cuda:0":
                self.close(force=True)
                raise PersistentNNEvalWorkerError(
                    f"GPU NNEval worker returned unexpected device {response['worker_device']!r}"
                )
            if not bool(response["cuda_available"]) or cuda_device_count != 1:
                self.close(force=True)
                raise PersistentNNEvalWorkerError(
                    "GPU NNEval worker must expose exactly one visible CUDA device, but reported "
                    f"(available={response['cuda_available']}, count={response['cuda_device_count']})"
                )
            if expected_device is not None and cuda_visible_devices != expected_device:
                self.close(force=True)
                raise PersistentNNEvalWorkerError(
                    f"GPU NNEval worker bound unexpected CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}, "
                    f"expected {expected_device!r}"
                )
        return {
            "pid": int(response["pid"]),
            "assigned_gpu": response["assigned_gpu"],
            "assigned_cuda_visible_device": (
                None
                if response["assigned_cuda_visible_device"] is None
                else str(response["assigned_cuda_visible_device"])
            ),
            "worker_device": str(response["worker_device"]),
            "cuda_visible_devices": str(response["cuda_visible_devices"]),
            "cuda_available": bool(response["cuda_available"]),
            "cuda_device_count": int(response["cuda_device_count"]),
            "rss_gib": float(response["rss_gib"]),
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


class _NNEvalWorkerPool:
    def __init__(self, *, plan: Optional[Dict[str, Any]] = None) -> None:
        self._plan = dict(plan or build_nneval_worker_plan())
        self._sessions: List[_PersistentNNEvalWorkerSession] = []
        try:
            if self._plan["eval_gpu_indices"]:
                eval_gpu_tokens = list(self._plan.get("eval_gpu_tokens") or [])
                for slot, assigned_gpu in enumerate(self._plan["eval_gpu_indices"]):
                    assigned_cuda_visible_device = (
                        eval_gpu_tokens[slot]
                        if slot < len(eval_gpu_tokens)
                        else str(self._plan["visible_gpu_tokens"][int(assigned_gpu)])
                    )
                    self._sessions.append(
                        _PersistentNNEvalWorkerSession(
                            assigned_gpu=int(assigned_gpu),
                            assigned_cuda_visible_device=assigned_cuda_visible_device,
                            worker_slot=slot,
                        )
                    )
            else:
                self._sessions.append(
                    _PersistentNNEvalWorkerSession(
                        assigned_gpu=None,
                        assigned_cuda_visible_device=None,
                        worker_slot=0,
                    )
                )
        except Exception:
            for session in self._sessions:
                session.close(force=True)
            self._sessions = [
                _PersistentNNEvalWorkerSession(
                    assigned_gpu=None,
                    assigned_cuda_visible_device=None,
                    worker_slot=0,
                )
            ]
            self._plan = {
                **self._plan,
                "mode": "cpu_fallback",
                "eval_gpu_indices": [],
                "eval_gpu_tokens": [],
                "per_gpu_worker_counts": [0] * max(0, int(self._plan.get("visible_gpu_count", 0))),
                "pool_size": 1,
                "workers_per_gpu": 0,
            }
        print(
            "[NNEval Worker Pool] "
            f"mode={self._plan['mode']} "
            f"use_all_visible_gpus={self._plan.get('use_all_visible_gpus', True)} "
            f"visible_gpu_count={self._plan['visible_gpu_count']} "
            f"target_gpu_tokens={self._plan.get('target_gpu_tokens', [])} "
            f"workers_per_gpu={self._plan.get('workers_per_gpu', 1)} "
            f"per_gpu_worker_counts={self._plan.get('per_gpu_worker_counts', [])} "
            f"eval_gpu_indices={self._plan['eval_gpu_indices']} "
            f"eval_gpu_tokens={self._plan.get('eval_gpu_tokens', [])} "
            f"reason={self._plan.get('reason', '')!r}"
        )

    def worker_count(self) -> int:
        return len(self._sessions)

    def plan_signature(self) -> Tuple[Any, ...]:
        return _nngpt_worker_plan_signature(self._plan)

    def diagnostics(self) -> Dict[str, Any]:
        workers = [session.diagnostics() for session in self._sessions]
        total_rss_gib = sum(float(worker.get("rss_gib") or 0.0) for worker in workers)
        return {
            **self._plan,
            "pool_size": len(workers),
            "workers": workers,
            "worker_pids": [worker.get("pid") for worker in workers],
            "total_rss_gib": total_rss_gib,
        }

    def _replace_session(self, slot: int) -> _PersistentNNEvalWorkerSession:
        old_session = self._sessions[slot]
        old_info = old_session.diagnostics()
        assigned_gpu = old_info.get("assigned_gpu")
        assigned_cuda_visible_device = old_info.get("assigned_cuda_visible_device")
        old_session.close(force=True)
        try:
            new_session = _PersistentNNEvalWorkerSession(
                assigned_gpu=assigned_gpu if assigned_gpu is not None else None,
                assigned_cuda_visible_device=assigned_cuda_visible_device,
                worker_slot=slot,
            )
        except Exception:
            new_session = _PersistentNNEvalWorkerSession(
                assigned_gpu=None,
                assigned_cuda_visible_device=None,
                worker_slot=slot,
            )
        self._sessions[slot] = new_session
        return new_session

    def _failure_result_for_entry(
        self,
        entry: Dict[str, Any],
        *,
        error: str,
        worker_slot: int,
        assigned_gpu: Optional[int],
        worker_device: str,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "model_id": str(entry["payload"].get("model_id", "")),
            "error": str(error),
            "traceback": "",
            "is_oom": "out of memory" in str(error).lower(),
            "worker_slot": int(worker_slot),
            "assigned_gpu": assigned_gpu,
            "worker_device": str(worker_device),
        }

    def _request_entry(self, slot: int, entry: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        session = self._sessions[slot]
        session_info = session.diagnostics()
        assigned_gpu = session_info.get("assigned_gpu")
        worker_device = str(session_info.get("worker_device", "cpu"))
        worker_slot = int(session_info.get("slot", slot))
        entry["payload"]["worker_slot"] = worker_slot
        model_id = str(entry["payload"].get("model_id", ""))
        request_started_at = time.time()
        print(
            "[NNEval Assign] "
            f"model_id={model_id} "
            f"worker_slot={worker_slot} "
            f"assigned_gpu={assigned_gpu} "
            f"worker_device={worker_device} "
            f"timeout_seconds={timeout:.0f}"
        )
        try:
            result = session.request(entry["payload"], timeout=timeout)
        except PersistentNNEvalWorkerError as exc:
            elapsed_seconds = max(0.0, time.time() - request_started_at)
            print(
                "[NNEval Assign] error "
                f"model_id={model_id} "
                f"worker_slot={worker_slot} "
                f"assigned_gpu={assigned_gpu} "
                f"worker_device={worker_device} "
                f"elapsed_seconds={elapsed_seconds:.2f} "
                f"error={type(exc).__name__}: {exc}"
            )
            self._replace_session(slot)
            return self._failure_result_for_entry(
                entry,
                error=f"{type(exc).__name__}: {exc}",
                worker_slot=worker_slot,
                assigned_gpu=assigned_gpu,
                worker_device=worker_device,
            )
        result["worker_slot"] = worker_slot
        result["assigned_gpu"] = assigned_gpu
        result["worker_device"] = worker_device
        if bool(result.get("worker_restart_requested", False)):
            print(
                "[NNEval Worker Pool] Restart "
                f"slot={worker_slot} "
                f"assigned_gpu={assigned_gpu} "
                f"worker_device={worker_device} "
                f"reason={result.get('error', 'unknown')!r}"
            )
            self._replace_session(slot)
        return result

    def map_entries(self, entries: List[Dict[str, Any]], *, timeout: float) -> List[Dict[str, Any]]:
        if not entries:
            return []
        if self.worker_count() <= 1:
            return [self._request_entry(0, entry, timeout=timeout) for entry in entries]

        indexed_results: List[Optional[Dict[str, Any]]] = [None] * len(entries)
        assignments: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in self._sessions]
        for index, entry in enumerate(entries):
            slot = index % self.worker_count()
            assignments[slot].append((index, entry))

        def _process_worker_tasks(slot: int, tasks: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
            return [
                (index, self._request_entry(slot, entry, timeout=timeout))
                for index, entry in tasks
            ]

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
            else self._failure_result_for_entry(
                entries[index],
                error="PersistentNNEvalWorkerError: missing result from worker pool dispatch",
                worker_slot=-1,
                assigned_gpu=None,
                worker_device="cpu",
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


def _get_or_create_nneval_worker_pool(*, use_all_visible_gpus: bool) -> _NNEvalWorkerPool:
    global _NNEVAL_WORKER_POOL
    desired_plan = build_nneval_worker_plan(use_all_visible_gpus=use_all_visible_gpus)
    desired_signature = _nngpt_worker_plan_signature(desired_plan)
    if _NNEVAL_WORKER_POOL is not None:
        diagnostics = _NNEVAL_WORKER_POOL.diagnostics()
        workers = diagnostics.get("workers", [])
        current_signature = _NNEVAL_WORKER_POOL.plan_signature()
        if current_signature != desired_signature:
            print(
                "[NNEval Worker Pool] Reconfigure "
                f"old_mode={diagnostics.get('mode')} "
                f"new_mode={desired_plan.get('mode')} "
                f"old_eval_gpu_indices={diagnostics.get('eval_gpu_indices')} "
                f"new_eval_gpu_indices={desired_plan.get('eval_gpu_indices')}"
            )
            _NNEVAL_WORKER_POOL.close()
            _NNEVAL_WORKER_POOL = None
        elif not workers or not all(bool(worker.get("alive", False)) for worker in workers):
            _NNEVAL_WORKER_POOL.close()
            _NNEVAL_WORKER_POOL = None
    if _NNEVAL_WORKER_POOL is None:
        _NNEVAL_WORKER_POOL = _NNEvalWorkerPool(plan=desired_plan)
    return _NNEVAL_WORKER_POOL


def _await_nneval_worker_pool(*, use_all_visible_gpus: bool, require_gpu: bool, timeout: float) -> Optional[_NNEvalWorkerPool]:
    if not require_gpu:
        return _get_or_create_nneval_worker_pool(use_all_visible_gpus=use_all_visible_gpus)
    if not torch.cuda.is_available():
        return None

    deadline = time.time() + max(0.0, float(timeout))
    logged_wait = False
    while True:
        pool = _get_or_create_nneval_worker_pool(use_all_visible_gpus=use_all_visible_gpus)
        diagnostics = pool.diagnostics()
        has_gpu_worker = any(worker.get("assigned_gpu") is not None for worker in diagnostics.get("workers", []))
        if has_gpu_worker:
            return pool

        remaining = max(0.0, deadline - time.time())
        if remaining <= 0.0:
            return None
        if not logged_wait:
            print(
                "[NNEval Worker Pool] Waiting "
                f"mode={diagnostics.get('mode')} "
                f"reason={diagnostics.get('reason', '')!r} "
                f"timeout_seconds={timeout:.0f}"
            )
            logged_wait = True
        shutdown_nneval_workers()
        time.sleep(min(5.0, remaining))


def prewarm_nneval_workers(*, use_all_visible_gpus: bool, timeout_seconds: float = 60.0) -> Dict[str, Any]:
    desired_plan = build_nneval_worker_plan(use_all_visible_gpus=use_all_visible_gpus)
    planned_gpu_workers = int(sum(max(0, int(count)) for count in desired_plan.get("per_gpu_worker_counts", []) or []))
    should_require_gpu = bool(planned_gpu_workers > 0 or desired_plan.get("mode") == "gpu_wait")
    print(
        "[NNEval Worker Warmup] start "
        f"mode={desired_plan.get('mode')} "
        f"use_all_visible_gpus={use_all_visible_gpus} "
        f"require_gpu={should_require_gpu} "
        f"planned_gpu_workers={planned_gpu_workers} "
        f"per_gpu_worker_counts={desired_plan.get('per_gpu_worker_counts', [])} "
        f"eval_gpu_tokens={desired_plan.get('eval_gpu_tokens', [])} "
        f"timeout_seconds={timeout_seconds:.0f}"
    )
    pool = _await_nneval_worker_pool(
        use_all_visible_gpus=use_all_visible_gpus,
        require_gpu=should_require_gpu,
        timeout=float(timeout_seconds),
    )
    if pool is None:
        raise PersistentNNEvalWorkerError(
            "NNEval worker warmup timed out while waiting for GPU workers "
            f"(timeout_seconds={timeout_seconds:.0f}, mode={desired_plan.get('mode')!r})"
        )
    diagnostics = pool.diagnostics()
    print(
        "[NNEval Worker Warmup] end "
        f"mode={diagnostics.get('mode')} "
        f"pool_size={diagnostics.get('pool_size')} "
        f"worker_pids={diagnostics.get('worker_pids', [])}"
    )
    return diagnostics


def shutdown_nneval_workers() -> None:
    global _NNEVAL_WORKER_POOL
    if _NNEVAL_WORKER_POOL is None:
        return
    info = _NNEVAL_WORKER_POOL.diagnostics()
    print(
        "[NNEval Worker Pool] Shutdown "
        f"mode={info.get('mode')} "
        f"pool_size={info.get('pool_size')} "
        f"per_gpu_worker_counts={info.get('per_gpu_worker_counts', [])} "
        f"worker_pids={info.get('worker_pids', [])} "
        f"total_rss_gib={_format_mem_value(info.get('total_rss_gib'))}"
    )
    _NNEVAL_WORKER_POOL.close()
    _NNEVAL_WORKER_POOL = None


def get_nneval_worker_diagnostics() -> Optional[Dict[str, Any]]:
    if _NNEVAL_WORKER_POOL is None:
        return None
    return _NNEVAL_WORKER_POOL.diagnostics()


def evaluate_model_entries(
    entries: List[Dict[str, Any]],
    *,
    use_all_visible_gpus: bool,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    normalized_entries = [
        {
            **entry,
            "payload": {
                "cmd": "evaluate_model",
                **dict(entry.get("payload") or {}),
            },
        }
        for entry in entries
    ]
    max_timeout = max(_request_timeout_seconds(entry["payload"]) for entry in entries)
    require_gpu = bool(torch.cuda.is_available())
    pool = _await_nneval_worker_pool(
        use_all_visible_gpus=use_all_visible_gpus,
        require_gpu=require_gpu,
        timeout=max_timeout,
    )
    if pool is None:
        return [
            {
                "success": False,
                "model_id": str(entry["payload"].get("model_id", "")),
                "error": "PersistentNNEvalWorkerError: timed out waiting for available GPU worker",
                "traceback": "",
                "is_oom": False,
                "assigned_gpu": None,
                "worker_device": "cpu",
                "worker_slot": -1,
            }
            for entry in entries
        ]
    return pool.map_entries(normalized_entries, timeout=max_timeout)


def _persistent_nneval_worker_entry(conn, assigned_gpu: Optional[int], assigned_cuda_visible_device: Optional[str]) -> None:
    worker_device = "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)
            worker_device = "cuda:0"
        conn.send(
            {
                "cmd": "worker_ready",
                "pid": os.getpid(),
                "assigned_gpu": assigned_gpu,
                "assigned_cuda_visible_device": assigned_cuda_visible_device,
                "worker_device": worker_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "rss_gib": float(_read_process_rss_gib() or 0.0),
            }
        )
    except Exception as exc:
        conn.send(
            {
                "cmd": "worker_init_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return

    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break
            if not isinstance(request, dict):
                conn.send(
                    {
                        "success": False,
                        "error": "Persistent NNEval worker received a non-dict request",
                        "traceback": "",
                        "is_oom": False,
                    }
                )
                continue
            if request.get("cmd") == "shutdown":
                break
            if request.get("cmd") != "evaluate_model":
                conn.send(
                    {
                        "success": False,
                        "error": f"Persistent NNEval worker received unknown command: {request.get('cmd')!r}",
                        "traceback": "",
                        "is_oom": False,
                    }
                )
                continue

            try:
                if worker_device.startswith("cuda"):
                    _clear_cuda_state()
                result = _execute_nneval_task(request)
            except Exception as exc:
                result = {
                    "success": False,
                    "model_id": str(request.get("model_id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "is_oom": "out of memory" in str(exc).lower(),
                }
            finally:
                _clear_cuda_state()
            result["worker_restart_requested"] = _is_fatal_cuda_worker_error(result.get("error"))
            result["assigned_gpu"] = assigned_gpu
            result["worker_device"] = worker_device
            result["worker_slot"] = request.get("worker_slot")
            conn.send(result)
            if bool(result.get("worker_restart_requested", False)):
                break
    finally:
        _clear_cuda_state()
        try:
            conn.close()
        except Exception:
            pass

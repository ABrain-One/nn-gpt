import ast
import csv
from datetime import timedelta
import inspect
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings


_RL_FILTERED_LOG_PATTERNS = (
    "Skipping import of cpp extensions due to incompatible torch version",
    "github.com/pytorch/ao/issues/2919",
)


class _RLFilteredStream:
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._buffer = ""

    def write(self, text):
        text = str(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        return self._wrapped.flush()

    def _write_line(self, text: str) -> None:
        normalized = " ".join(text.split())
        if all(pattern in normalized for pattern in _RL_FILTERED_LOG_PATTERNS):
            return
        self._wrapped.write(text)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _install_rl_runtime_noise_filters() -> None:
    if getattr(_install_rl_runtime_noise_filters, "_installed", False):
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\. Use `HF_HOME` instead\..*",
        category=FutureWarning,
    )
    sys.stdout = _RLFilteredStream(sys.stdout)
    sys.stderr = _RLFilteredStream(sys.stderr)
    _install_rl_runtime_noise_filters._installed = True


_install_rl_runtime_noise_filters()

import torch
from peft import prepare_model_for_kbit_training
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
import ab.gpt.rl_pipeline.trainer_runtime as TrainerRuntime
import ab.gpt.rl_pipeline.stage_state as StageState
import ab.gpt.rl_pipeline.reward_payload as RewardPayload
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.util.ArchDiscovery import (
    ensure_pattern_name,
    extract_graph_info,
    normalize_pattern_name,
)
from ab.gpt.util.Util import extract_str
from ab.gpt.util.Const import conf_train_dir, conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
from ab.gpt.util.Reward import (
    EvalConfig,
    FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC,
    PersistentEvalWorkerError,
    evaluate_code_and_reward,
    evaluate_code_and_reward_batch,
    get_distributed_runtime_info,
    get_eval_worker_diagnostics,
    prewarm_eval_workers,
    shutdown_eval_worker,
)
import ab.nn.api as api

import textwrap
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ab.gpt.util.simple_logger import SimpleCodeLogger
import ab.gpt.util.training_runtime as TrainingRuntime
from typing import Tuple, Any, List, Dict, Optional, Set
from collections import Counter, deque

# Open-architecture archives are keyed by canonical graph structure, not prompt labels.
graph_archive_counts = Counter()
family_archive_counts = Counter()
family_hash_archive_counts = Counter()
descriptor_archive_counts = Counter()
backbone_signature_archive_counts = Counter()
cnn_signature_archive_counts = Counter()
backbone_cnn_pair_archive_counts = Counter()
family_metric_best: Dict[str, float] = {}
motif_name_counts = Counter()
saved_graph_counts = Counter()
saved_family_hash_counts = Counter()
saved_backbone_signature_counts = Counter()
saved_cnn_signature_counts = Counter()
saved_backbone_cnn_pair_counts = Counter()
goal_graph_archive_counts: Dict[str, Counter] = {}
goal_family_hash_archive_counts: Dict[str, Counter] = {}
saved_goal_family_hash_counts: Dict[str, Counter] = {}
train_graph_hashes: Set[str] = set()
train_family_hashes: Set[str] = set()
train_descriptor_keys: Set[str] = set()
train_reference_stats: Dict[str, int] = {}
current_group_reward_target_sum_by_backbone: Dict[str, float] = {}
current_group_reward_target_count_by_backbone = Counter()
prev_closed_group_mean_reward_target_by_backbone: Dict[str, float] = {}
best_closed_group_mean_reward_target_by_backbone: Dict[str, float] = {}
saved_best_reward_target_by_backbone_cnn: Dict[str, float] = {}


# ===== Configuration Options =====
base_model = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct" # 使用新的 Backbone 模型
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
B_index = 0
GROUP_BATCH_SIZE = 20
GROUP_IMPROVEMENT_DELTA = 0.003
BEST_GROUP_REFRESH_DELTA = 0.0015
GOAL_REFRESH_DELTA = 0.0015
NON_IMPROVING_REWARD_CAP = 0.04
FORMAL_REWARD_TRANSFORM = "norm_128_flip"
BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES = 3
SAVE_DUPLICATE_BACKBONE_CNN_DELTA = 0.002
BATCH_ELITE_SOFT_BONUSES = (0.10, 0.07, 0.05, 0.03, 0.02)
BATCH_ELITE_IMPROVING_BONUSES = (0.18, 0.13, 0.09, 0.06, 0.04)
STRUCTURE_MACRO_BONUS = 0.04
STRUCTURE_MULTI_STAGE_BONUS = 0.03
STRUCTURE_MOTIF_BONUS = 0.02
STRUCTURE_BATCH_DIVERSITY_BONUS = 0.03
STRUCTURE_NON_DOMINANT_FAMILY_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS = 0.03
STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS = 0.01
REPEAT_FAMILY_PENALTY = -0.10
PLAIN_FUSE_PENALTY = -0.10
NO_PROGRESS_PENALTY = -0.06
GOAL_REFRESH_BONUS = 0.30
GOAL_MATCH_REWARD_SCALE = 0.12
TRAINSET_NOVEL_FAMILY_BONUS = 0.04
TRAINSET_NOVEL_GRAPH_BONUS = 0.02
GENERALIZATION_GAP_TOLERANCE = 0.02
GENERALIZATION_PENALTY_SCALE = 2.0
GENERALIZATION_PENALTY_CAP = -0.20
REWARD_TARGET_METRIC = "frozen_test_acc"
FEEDBACK_GRAPH_EXPR_MAX_CHARS = 160
FEEDBACK_SUMMARY_MAX_CHARS = 240
FEEDBACK_SUMMARY_LIMIT = 2
RL_DEEPSPEED_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "conf" / "DeepSpeedSftGrpo.json")
STAGE1_STRUCTURE_EXPLORE = "stage1_structure_explore"
STAGE2_FORMAL_EXPLORE = "stage2_formal_explore"
STAGE3_FORMAL_OPTIMIZE = "stage3_formal_optimize"
RL_STAGE_ORDER = (
    STAGE1_STRUCTURE_EXPLORE,
    STAGE2_FORMAL_EXPLORE,
    STAGE3_FORMAL_OPTIMIZE,
)
RL_STAGE_TO_INDEX = {
    stage_name: index
    for index, stage_name in enumerate(RL_STAGE_ORDER, start=1)
}
STAGE_REFERENCE_MIN_GROUPS = {
    STAGE1_STRUCTURE_EXPLORE: 4,
    STAGE2_FORMAL_EXPLORE: 5,
    STAGE3_FORMAL_OPTIMIZE: 0,
}
STAGE1_GATE_WINDOW_GENERATIONS = 1600
STAGE2_GATE_WINDOW_GENERATIONS = 4000
RECOVERY_GATE_WINDOW_GENERATIONS = 2000
STAGE1_PROMOTION_MIN_GROUPS = 20
STAGE1_GATE_EXECUTABLE_MIN = 96
STAGE1_GATE_DISCOVERY_MIN = 8
STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE1_FORCE_PROMOTION_EXECUTABLE_MIN = 800
STAGE1_FORCE_PROMOTION_DISCOVERY_MIN = 8
STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE2_GATE_MIN_REWARD_TARGET = 0.90
STAGE2_GATE_MIN_TARGET_COUNT = 16
STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES = 6
STAGE2_GATE_IMPROVING_GROUPS_REQUIRED = 2
STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE = 0.50
STAGE_RECOVERY_DOMINANT_SHARE_THRESHOLD = 0.55
STAGE_RECOVERY_NEW_DISCOVERY_FAMILIES_MAX = 1
STAGE_RECOVERY_RELEASE_GENERATIONS = 2000
STAGE_RECOVERY_RELEASE_DISCOVERY_FAMILIES = 4
MAX_STAGE_SAMPLE_HISTORY = 24000
MAX_STAGE_GROUP_HISTORY = 512
TRAINING_CONTEXT_WINDOW = 50
TRAINING_CONTEXT_MIN_POINTS = 8
STATIC_STAGE_REWARD_TARGET_METRIC = "stage1_static_score"
FORMAL_STAGE_REWARD_TARGET_METRIC = FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
FORMAL_SUCCESS_SIGNAL_BONUS = 0.02
STAGE1_EXECUTABLE_BONUS = 0.10
STAGE1_DISCOVERY_FAMILY_BONUS = 0.42
STAGE1_DISCOVERY_GRAPH_BONUS = 0.20
STAGE1_STATIC_BASE_SCORE = 0.04
STAGE1_GOAL_MATCH_SCALE = 0.10
STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE = 1.0 / 3.0
STAGE1_ZERO_GOAL_HIT_PENALTY = 0.0
STAGE1_LOW_GOAL_HIT_PENALTY = 0.0
STAGE1_STRUCTURE_GROUP_SCALE = 1.45
STAGE1_STRUCTURE_ARCHIVE_SCALE = 1.85
STAGE1_NON_DISCOVERY_EXECUTABLE_PENALTY = 0.0
STAGE1_ARCHIVE_REPEAT_STEP_PENALTY = -0.05
STAGE1_ARCHIVE_REPEAT_MAX_PENALTY = -0.30
STAGE1_BATCH_REPEAT_STEP_PENALTY = -0.05
STAGE1_BATCH_REPEAT_MAX_PENALTY = -0.24
STAGE1_DOMINANT_FAMILY_PENALTY = -0.08
STAGE1_PLAIN_PARALLEL_PENALTY = -0.10
STAGE1_PLAIN_PARALLEL_WARMUP_PENALTY = -0.03
STAGE1_DESCRIPTOR_BATCH_UNIQUE_BONUS = 0.12
STAGE1_GRAPH_BATCH_UNIQUE_BONUS = 0.05
STAGE1_DESCRIPTOR_ARCHIVE_NOVEL_BONUS = 0.03
STAGE1_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY = -0.04
STAGE1_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY = -0.12
STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY = -0.02
STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY = -0.08
STAGE1_GRAPH_BATCH_REPEAT_STEP_PENALTY = -0.06
STAGE1_GRAPH_BATCH_REPEAT_MAX_PENALTY = -0.18
STAGE1_ZERO_GOAL_HIT_REWARD_CAP = 1.0
STAGE1_LOW_GOAL_HIT_REWARD_CAP = 1.0
STAGE1_PLAIN_PARALLEL_REWARD_CAP = 1.0
STAGE1_OFF_TARGET_PLAIN_PARALLEL_REWARD_CAP = 1.0
STAGE23_DESCRIPTOR_BATCH_UNIQUE_BONUS = 0.03
STAGE23_DESCRIPTOR_ARCHIVE_NOVEL_BONUS = 0.02
STAGE23_NON_DOMINANT_DESCRIPTOR_BONUS = 0.06
STAGE23_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY = -0.05
STAGE23_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY = -0.20
STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY = -0.025
STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY = -0.14
STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE = 0.45
STAGE23_DOMINANT_DESCRIPTOR_STRONG_SHARE = 0.60
STAGE23_DOMINANT_DESCRIPTOR_REPEAT_PENALTY = -0.12
STAGE23_DOMINANT_DESCRIPTOR_REPEAT_STRONG_PENALTY = -0.20
STAGE23_CNN_BATCH_UNIQUE_BONUS = 0.07
STAGE23_CNN_ARCHIVE_NOVEL_BONUS = 0.05
STAGE23_CNN_BATCH_REPEAT_STEP_PENALTY = -0.08
STAGE23_CNN_BATCH_REPEAT_MAX_PENALTY = -0.30
STAGE23_CNN_ARCHIVE_REPEAT_STEP_PENALTY = -0.03
STAGE23_CNN_ARCHIVE_REPEAT_MAX_PENALTY = -0.18
STAGE23_NON_DOMINANT_CNN_BONUS = 0.08
STAGE23_DOMINANT_CNN_SOFT_SHARE = 0.45
STAGE23_DOMINANT_CNN_STRONG_SHARE = 0.65
STAGE23_DOMINANT_CNN_REPEAT_PENALTY = -0.16
STAGE23_DOMINANT_CNN_REPEAT_STRONG_PENALTY = -0.24
STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP = 0.03
STAGE2_DENSE_SCALE = 0.50
STAGE2_PREV_GROUP_SCALE = 0.70
STAGE2_BEST_GROUP_SCALE = 0.70
STAGE2_GLOBAL_BASELINE_BLEND = 0.20
STAGE2_BACKBONE_PREV_GROUP_SCALE = 0.95
STAGE2_BACKBONE_BEST_GROUP_SCALE = 0.95
STAGE2_GOAL_BEST_SCALE = 0.70
STAGE2_GOAL_MATCH_SCALE = 0.85
STAGE2_STRUCTURE_SCALE = 1.40
STAGE2_REPEAT_FAMILY_SCALE = 1.10
STAGE2_PLAIN_FUSE_SCALE = 1.10
STAGE2_NO_PROGRESS_SCALE = 0.50
STAGE2_NON_IMPROVING_CAP = 0.10
STAGE2_DESCRIPTOR_NON_IMPROVING_CAP = 0.03
STAGE3_DENSE_SCALE = 0.70
STAGE3_PREV_GROUP_SCALE = 1.10
STAGE3_BEST_GROUP_SCALE = 1.10
STAGE3_GLOBAL_BASELINE_BLEND = 0.25
STAGE3_BACKBONE_PREV_GROUP_SCALE = 1.20
STAGE3_BACKBONE_BEST_GROUP_SCALE = 1.15
STAGE3_GOAL_BEST_SCALE = 1.00
STAGE3_GOAL_MATCH_SCALE = 1.00
STAGE3_STRUCTURE_SCALE = 0.85
STAGE3_REPEAT_FAMILY_SCALE = 1.00
STAGE3_PLAIN_FUSE_SCALE = 1.00
STAGE3_NO_PROGRESS_SCALE = 1.15
STAGE3_NON_IMPROVING_CAP = NON_IMPROVING_REWARD_CAP
STAGE3_DESCRIPTOR_NON_IMPROVING_CAP = 0.00
RL_STAGE_KL_COEF = 0.005
reward_batch_index = 0
current_group_id = 0
current_group_reward_target_sum = 0.0
current_group_reward_target_count = 0
current_group_frozen_train_acc_sum = 0.0
current_group_frozen_train_acc_count = 0
current_group_frozen_test_acc_sum = 0.0
current_group_frozen_test_acc_count = 0
current_group_unfrozen_train_acc_sum = 0.0
current_group_unfrozen_train_acc_count = 0
current_group_unfrozen_test_acc_sum = 0.0
current_group_unfrozen_test_acc_count = 0
prev_closed_group_mean_reward_target_acc: Optional[float] = None
best_closed_group_mean_reward_target_acc: Optional[float] = None
prev_closed_group_train_acc_mean: Optional[float] = None
best_closed_group_mean_train_acc: Optional[float] = None
prev_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_id: Optional[int] = None
best_reward_target_by_goal: Dict[str, float] = {}
dominant_family_hash: Optional[str] = None
dominant_family_share: float = 0.0
dominant_descriptor_key: Optional[str] = None
dominant_descriptor_share: float = 0.0
dominant_backbone_signature: Optional[str] = None
dominant_backbone_share: float = 0.0
dominant_cnn_signature: Optional[str] = None
dominant_cnn_share: float = 0.0
dominant_backbone_cnn_pair: Optional[str] = None
dominant_backbone_cnn_share: float = 0.0
prev_group_feedback: List["GroupFeedbackSummary"] = []
best_group_feedback: List["GroupFeedbackSummary"] = []
current_group_top_feedback: List["GroupFeedbackSummary"] = []
current_group_goal_best_candidates: Dict[str, float] = {}
current_stage_name = STAGE1_STRUCTURE_EXPLORE
stage_closed_group_counts = Counter()
stage_best_group_mean_reward_target: Dict[str, float] = {}
stage_entry_generation_totals: Dict[str, int] = {}
stage_entry_reward_batches: Dict[str, int] = {}
generation_history: List[Dict[str, Any]] = []
closed_group_history: List[Dict[str, Any]] = []
stage_event_history: List[Dict[str, Any]] = []
discovery_family_hashes_seen: Set[str] = set()
recovery_active = False
recovery_start_generation_total = 0
recovery_start_discovery_family_count = 0
# ==================================


class NullCodeLogger:
    def log_to_file(self, message: str) -> None:
        return

    def log_generation(self, prompt: str, completion: str, reward: float, api_result: Any = None) -> None:
        return

    def save_log(self) -> None:
        return


code_logger: Any = NullCodeLogger()
active_rl_model: Any = None
active_rl_tokenizer: Any = None
_registered_signal_handlers: Dict[int, Any] = {}
_signal_checkpoint_in_progress = False


def clear_extraction_meta_cache() -> None:
    return

SHALLOW_COLLAPSE_FAMILIES = {
    "ParallelTriple_Shallow",
    "DualBackboneFuse_Shallow",
    "TripleBackboneFuse_Shallow",
}


@dataclass
class GroupFeedbackSummary:
    goal_key: str
    pattern_name: str
    graph_expr_short: str
    reward_target_value: float
    frozen_train_acc: float
    frozen_test_acc: float
    unfrozen_train_acc: Optional[float]
    unfrozen_test_acc: Optional[float]
    backbone_model_names: List[str]
    stats_short: str
    summary: str
    family_hash: str
    signature: str
    reward_group_id: int
    backbone_signature: str = ""
    cnn_signature: str = ""
    cnn_expr_short: str = ""


def _current_generation_total() -> int:
    return StageState.current_generation_total(sys.modules[__name__])


def _normalize_backbone_signature_names(backbone_model_names: Optional[List[str]]) -> List[str]:
    normalized = [
        str(name).strip()
        for name in list(backbone_model_names or [])
        if str(name).strip()
    ]
    normalized.sort()
    return normalized


def build_backbone_signature(backbone_model_names: Optional[List[str]]) -> str:
    normalized = _normalize_backbone_signature_names(backbone_model_names)
    return " + ".join(normalized) if normalized else "unknown_backbone_pair"


def _backbone_cnn_pair_key(backbone_signature: str, cnn_signature: str) -> str:
    return f"{str(backbone_signature or 'unknown_backbone_pair')}::{str(cnn_signature or 'incomplete_cnn')}"


def _result_backbone_signature(res: Dict[str, Any]) -> str:
    signature = str(res.get("backbone_signature") or "").strip()
    if signature:
        return signature
    return build_backbone_signature(res.get("backbone_model_names"))


def _result_cnn_signature(res: Dict[str, Any], graph_info) -> str:
    signature = str(res.get("cnn_signature") or "").strip()
    if signature:
        return signature
    if graph_info is not None:
        signature = str(getattr(graph_info, "cnn_signature", "") or "").strip()
        if signature:
            return signature
    return "incomplete_cnn"


def capture_reward_runtime_state() -> Dict[str, Any]:
    return StageState.capture_reward_runtime_state(
        globals(),
        max_stage_sample_history=MAX_STAGE_SAMPLE_HISTORY,
        max_stage_group_history=MAX_STAGE_GROUP_HISTORY,
        feedback_summary_payload=_feedback_summary_payload,
        current_group_top_feedback_payload=_current_group_top_feedback_payload,
    )


def restore_reward_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    StageState.restore_reward_runtime_state(
        globals(),
        state,
        max_stage_sample_history=MAX_STAGE_SAMPLE_HISTORY,
        max_stage_group_history=MAX_STAGE_GROUP_HISTORY,
        stage1_structure_explore=STAGE1_STRUCTURE_EXPLORE,
        feedback_summary_cls=GroupFeedbackSummary,
    )

def _distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


_OBJECT_SYNC_GROUP = None
_OBJECT_SYNC_GROUP_WORLD_SIZE = None
_OBJECT_SYNC_GROUP_DISABLED = False


def _distributed_world_size() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_world_size())
    return max(1, env_int("WORLD_SIZE", 1))


def _distributed_rank() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_rank())
    return env_int("RANK", 0)


def is_main_process() -> bool:
    return _distributed_rank() == 0


def _object_sync_timeout_seconds() -> int:
    return max(600, env_int("NNGPT_RL_OBJECT_SYNC_TIMEOUT_SECONDS", 3600))


def _default_process_group_backend() -> str:
    if not _distributed_initialized():
        return ""
    try:
        backend = torch.distributed.get_backend()
    except Exception:
        return ""
    backend_text = str(backend).lower()
    if backend_text.startswith("backend."):
        backend_text = backend_text.split(".", 1)[1]
    return backend_text


def _get_object_sync_group():
    global _OBJECT_SYNC_GROUP
    global _OBJECT_SYNC_GROUP_WORLD_SIZE
    global _OBJECT_SYNC_GROUP_DISABLED

    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return None
    if _default_process_group_backend() == "gloo":
        return None
    current_world_size = _distributed_world_size()
    if _OBJECT_SYNC_GROUP is not None and _OBJECT_SYNC_GROUP_WORLD_SIZE == current_world_size:
        return _OBJECT_SYNC_GROUP
    if _OBJECT_SYNC_GROUP_DISABLED:
        return None

    timeout_seconds = _object_sync_timeout_seconds()
    try:
        _OBJECT_SYNC_GROUP = torch.distributed.new_group(
            backend="gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
        _OBJECT_SYNC_GROUP_WORLD_SIZE = current_world_size
        print(
            "[Reward Sync Group] initialized "
            f"rank={_distributed_rank()} "
            f"world_size={current_world_size} "
            f"backend=gloo "
            f"timeout_seconds={timeout_seconds}"
        )
    except Exception as exc:
        _OBJECT_SYNC_GROUP = None
        _OBJECT_SYNC_GROUP_WORLD_SIZE = None
        _OBJECT_SYNC_GROUP_DISABLED = True
        print(
            "[Reward Sync Group] fallback "
            f"rank={_distributed_rank()} "
            f"backend={_default_process_group_backend() or 'unknown'} "
            f"error={type(exc).__name__}: {exc}"
        )
    return _OBJECT_SYNC_GROUP


def _all_gather_object(payload: Any) -> List[Any]:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return [payload]
    gathered: List[Any] = [None] * _distributed_world_size()
    torch.distributed.all_gather_object(gathered, payload, group=_get_object_sync_group())
    return gathered


def _broadcast_object(payload: Any, *, src: int = 0) -> Any:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return payload
    objects = [payload if _distributed_rank() == src else None]
    torch.distributed.broadcast_object_list(objects, src=src, group=_get_object_sync_group())
    return objects[0]


def has_structural_motif(graph_info) -> bool:
    return bool(graph_info and (graph_info.project_calls or graph_info.stem_calls or graph_info.fractal_calls))


def is_multi_stage_architecture(graph_info) -> bool:
    return bool(graph_info and (graph_info.depth >= 5 or graph_info.merges >= 2 or graph_info.fractal_calls >= 2))


def passes_macro_structure_gate(graph_info) -> bool:
    if not graph_info or not graph_info.parse_ok or graph_info.is_plain_parallel_triple:
        return False
    if graph_info.project_calls or graph_info.stem_calls:
        return True
    return is_multi_stage_architecture(graph_info)


def is_shallow_one_shot_fuse(graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and not graph_info.is_plain_parallel_triple
        and graph_info.fuse_calls >= 1
        and graph_info.merges <= 1
        and graph_info.depth <= 4
        and graph_info.project_calls == 0
        and graph_info.stem_calls == 0
        and graph_info.fractal_calls <= 1
        and graph_info.backbone_calls >= 1
    )


def family_save_cap(graph_info) -> int:
    return 4


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _stage1_only_enabled() -> bool:
    return env_flag("NNGPT_RL_STAGE1_ONLY", False)


def resolve_generation_plan(
    runtime: Dict[str, Any],
    *,
    env_name: str,
    default: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, int]:
    world_size = max(1, int(runtime.get("world_size", 1)))
    requested_global_num_generations = max(1, env_int(env_name, default))
    effective_train_batch_size = max(
        1,
        int(world_size) * max(1, int(per_device_train_batch_size)) * max(1, int(gradient_accumulation_steps)),
    )
    valid_generation_values = [
        value
        for value in range(2, effective_train_batch_size + 1)
        if effective_train_batch_size % value == 0
    ]
    if not valid_generation_values:
        raise ValueError(
            f"{env_name} cannot be resolved because effective_train_batch_size={effective_train_batch_size} "
            "does not permit GRPO's minimum 2 generations per prompt. Increase gradient accumulation or batch size."
        )

    if requested_global_num_generations in valid_generation_values:
        resolved_global_num_generations = requested_global_num_generations
    else:
        lower_or_equal = [value for value in valid_generation_values if value <= requested_global_num_generations]
        resolved_global_num_generations = (
            max(lower_or_equal)
            if lower_or_equal
            else min(valid_generation_values)
        )
    return {
        "world_size": world_size,
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_train_batch_size": int(effective_train_batch_size),
        "requested_global_num_generations": requested_global_num_generations,
        "global_num_generations": int(resolved_global_num_generations),
        "effective_global_num_generations": int(resolved_global_num_generations),
        "global_num_generations_adapted": int(resolved_global_num_generations != requested_global_num_generations),
        "valid_generation_values": list(valid_generation_values),
    }


def resolve_rl_runtime_settings(runtime: Dict[str, Any]) -> Dict[str, int]:
    grad_accum = env_int("NNGPT_RL_GRAD_ACCUM", 16)
    fixed_num_generations = 8
    os.environ["NNGPT_RL_NUM_GENERATIONS"] = str(fixed_num_generations)
    generation_plan = resolve_generation_plan(
        runtime,
        env_name="NNGPT_RL_NUM_GENERATIONS",
        default=fixed_num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
    )
    return {
        "dataset_limit": env_int("NNGPT_RL_DATASET_LIMIT", 500),
        "grad_accum": grad_accum,
        "max_completion_length": env_int("NNGPT_RL_MAX_COMPLETION_LENGTH", 1024),
        "effective_train_batch_size": generation_plan["effective_train_batch_size"],
        "requested_global_num_generations": generation_plan["requested_global_num_generations"],
        "global_num_generations": generation_plan["global_num_generations"],
        "effective_global_num_generations": generation_plan["effective_global_num_generations"],
        "global_num_generations_adapted": generation_plan["global_num_generations_adapted"],
        "valid_generation_values": generation_plan["valid_generation_values"],
    }


def _resolve_rl_deepspeed_enabled(runtime: Dict[str, Any]) -> bool:
    raw = os.getenv("NNGPT_RL_USE_DEEPSPEED")
    if raw is None or raw == "":
        return int(runtime.get("world_size", 1)) > 1
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_rl_deepspeed_config_path() -> str:
    config_path = Path(os.getenv("NNGPT_RL_DEEPSPEED_CONFIG", RL_DEEPSPEED_DEFAULT_CONFIG)).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"RL DeepSpeed config not found: {config_path}")
    return str(config_path)


def _maybe_init_hf_deepspeed_config(config_path: str) -> Any:
    last_error: Optional[Exception] = None
    for module_name in ("transformers.integrations", "transformers.deepspeed"):
        try:
            module = __import__(module_name, fromlist=["HfDeepSpeedConfig"])
            config_cls = getattr(module, "HfDeepSpeedConfig", None)
            if config_cls is not None:
                return config_cls(config_path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "DeepSpeed ZeRO-3 requested for RL GRPO, but HfDeepSpeedConfig could not be imported"
    ) from last_error


def _build_rl_grpo_config(
    *,
    precision: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: Optional[str],
    runtime_settings: Dict[str, int],
) -> Any:
    config_signature = inspect.signature(GRPOConfig.__init__)
    config_kwargs: Dict[str, Any] = {
        "temperature": env_float("NNGPT_RL_TEMPERATURE", 1.0),
        "learning_rate": env_float("NNGPT_RL_LR", 5e-5),
        "max_completion_length": runtime_settings["max_completion_length"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": runtime_settings["grad_accum"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": env_int("NNGPT_RL_NUM_EPOCHS", 5),
        "remove_unused_columns": False,
        "logging_steps": 1,
        "output_dir": os.getenv("NNGPT_RL_TRAINER_OUT", "./grpo_backbone_outputs"),
        "eval_strategy": "no",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": True,
        "num_generations": runtime_settings["global_num_generations"],
    }
    if "gradient_checkpointing_kwargs" in config_signature.parameters:
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    explicit_kl_coef = env_float("NNGPT_RL_KL_COEF", RL_STAGE_KL_COEF)
    if "beta" in config_signature.parameters:
        config_kwargs["beta"] = explicit_kl_coef
    elif "kl_coef" in config_signature.parameters:
        config_kwargs["kl_coef"] = explicit_kl_coef
    else:
        raise RuntimeError("Installed GRPOConfig does not expose `beta` or `kl_coef`; cannot set explicit KL control")
    if use_deepspeed:
        if "deepspeed" not in config_signature.parameters:
            raise RuntimeError("Installed GRPOConfig does not support the `deepspeed` argument")
        config_kwargs["deepspeed"] = deepspeed_config_path
        if "ds3_gather_for_generation" in config_signature.parameters:
            config_kwargs["ds3_gather_for_generation"] = False
    return GRPOConfig(**config_kwargs)


def best_mixed_precision() -> Dict[str, Any]:
    bf16_requested = os.getenv("NNGPT_RL_USE_BF16", "").strip().lower() in {"1", "true", "yes", "on"}
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(bf16_requested and bf16_ok)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "torch_dtype": torch_dtype,
        "label": "bf16" if use_bf16 else "fp16",
    }


class DTypeSafeLinearWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    @property
    def weight(self):
        return getattr(self.module, "weight", None)

    @property
    def bias(self):
        return getattr(self.module, "bias", None)

    def forward(self, inputs, *args, **kwargs):
        weight = getattr(self.module, "weight", None)
        if weight is not None and hasattr(inputs, "dtype") and inputs.dtype != weight.dtype:
            inputs = inputs.to(weight.dtype)
        return self.module(inputs, *args, **kwargs)


def align_generation_head_dtype(model, torch_dtype: torch.dtype) -> None:
    aligned_modules = []
    wrapped_modules = []
    visited_models = set()
    visited_modules = set()
    wrapper_cache: Dict[int, DTypeSafeLinearWrapper] = {}

    def _cast_module(module, label: str) -> None:
        if module is None or not hasattr(module, "weight"):
            return
        if isinstance(module, DTypeSafeLinearWrapper):
            module = module.module
        module_id = id(module)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)
        weight = getattr(module, "weight", None)
        if weight is None:
            return
        before_dtype = weight.dtype
        if before_dtype == torch_dtype:
            return
        module.to(dtype=torch_dtype)
        aligned_modules.append(f"{label}:{before_dtype}->{torch_dtype}")

    def _ensure_wrapper(module, label: str):
        if module is None or not hasattr(module, "weight"):
            return module
        if isinstance(module, DTypeSafeLinearWrapper):
            return module
        module_id = id(module)
        wrapped = wrapper_cache.get(module_id)
        if wrapped is None:
            wrapped = DTypeSafeLinearWrapper(module)
            wrapper_cache[module_id] = wrapped
            wrapped_modules.append(label)
        return wrapped

    def _walk_model_tree(current_model, prefix: str) -> None:
        if current_model is None:
            return
        model_id = id(current_model)
        if model_id in visited_models:
            return
        visited_models.add(model_id)

        _cast_module(getattr(current_model, "lm_head", None), f"{prefix}.lm_head")
        try:
            _cast_module(current_model.get_output_embeddings(), f"{prefix}.output_embeddings")
        except Exception:
            pass

        head_module = getattr(current_model, "lm_head", None)
        wrapped_head = _ensure_wrapper(head_module, f"{prefix}.lm_head")
        if wrapped_head is not head_module:
            try:
                setattr(current_model, "lm_head", wrapped_head)
            except Exception:
                pass

        try:
            output_module = current_model.get_output_embeddings()
        except Exception:
            output_module = None
        wrapped_output = _ensure_wrapper(output_module, f"{prefix}.output_embeddings")
        if wrapped_output is not output_module and hasattr(current_model, "set_output_embeddings"):
            try:
                current_model.set_output_embeddings(wrapped_output)
            except Exception:
                pass

        for attr_name in ("base_model", "model", "module"):
            nested_model = getattr(current_model, attr_name, None)
            if nested_model is not None and nested_model is not current_model:
                _walk_model_tree(nested_model, f"{prefix}.{attr_name}")

    _walk_model_tree(model, "model")

    config = getattr(model, "config", None)
    if config is not None:
        try:
            config.torch_dtype = torch_dtype
        except Exception:
            pass

    if aligned_modules:
        print(f"[RL] Output dtype alignment: {', '.join(aligned_modules)}")
    if wrapped_modules:
        print(f"[RL] Output dtype safety wrappers: {', '.join(wrapped_modules)}")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_reward_target_value(res: Dict[str, Any]) -> Optional[float]:
    reward_target_value = _optional_float(res.get("reward_target_value"))
    if reward_target_value is not None:
        return reward_target_value
    return _optional_float(res.get("frozen_test_acc", res.get("val_metric")))


def _increment_optional_metric(sum_name: str, count_name: str, value: Optional[float]) -> None:
    if value is None:
        return
    globals()[sum_name] += float(value)
    globals()[count_name] += 1


def _mean_from_accumulator(sum_value: float, count_value: int) -> Optional[float]:
    if count_value <= 0:
        return None
    return float(sum_value) / float(count_value)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _feedback_stats_short(open_discovery: Dict[str, Any]) -> str:
    structure_progress = float(open_discovery.get("r_structure_group", 0.0) or 0.0) + float(
        open_discovery.get("r_structure_archive", 0.0) or 0.0
    )
    return (
        f"depth:{int(open_discovery.get('depth', 0))} "
        f"merges:{int(open_discovery.get('merges', 0))} "
        f"stem:{int(open_discovery.get('stem_calls', 0))} "
        f"project:{int(open_discovery.get('project_calls', 0))} "
        f"fuse:{int(open_discovery.get('fuse_calls', 0))} "
        f"struct:{structure_progress:.2f}"
    )


def _group_feedback_paths() -> Tuple[Path, Path, Path]:
    log_dir = Path(run_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)
    return (
        log_dir / "group_progress.jsonl",
        log_dir / "group_feedback_samples.jsonl",
        log_dir / "best_group_feedback.json",
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _reward_runtime_hooks() -> TrainingRuntime.RuntimeStateHooks:
    # Reward bookkeeping is pipeline-owned, but the save/restore contract is shared.
    return TrainingRuntime.RuntimeStateHooks(
        capture=capture_reward_runtime_state,
        restore=restore_reward_runtime_state,
        reset=reset_reward_runtime_state,
    )


def _current_group_top_feedback_payload() -> List[Dict[str, Any]]:
    return [asdict(item) for item in current_group_top_feedback[:FEEDBACK_SUMMARY_LIMIT]]


def _feedback_summary_payload(items: List[GroupFeedbackSummary]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in items[:FEEDBACK_SUMMARY_LIMIT]]


def _build_group_feedback_summary(
    *,
    goal_key: str,
    res: Dict[str, Any],
    graph_info,
    reward_group_id: int,
) -> GroupFeedbackSummary:
    graph_expr_short = _truncate_text(str(res.get("graph_expr") or ""), FEEDBACK_GRAPH_EXPR_MAX_CHARS)
    pattern_name = str(res.get("pattern_name") or res.get("suggested_pattern_name") or "unknown")
    reward_target_value = float(_result_reward_target_value(res) or 0.0)
    frozen_train_acc = float(_optional_float(res.get("frozen_train_acc", res.get("train_acc"))) or 0.0)
    frozen_test_acc = float(_optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))) or 0.0)
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    backbone_names = list(res.get("backbone_model_names") or [])
    backbone_signature = str(res.get("backbone_signature") or build_backbone_signature(backbone_names))
    cnn_signature = str(res.get("cnn_signature") or getattr(graph_info, "cnn_signature", "") or "")
    cnn_expr_short = _truncate_text(str(res.get("cnn_expr") or getattr(graph_info, "cnn_expr", "") or ""), 96)
    open_discovery = dict(res.get("open_discovery") or {})
    stats_short = _feedback_stats_short(open_discovery)
    summary = (
        f"pattern={pattern_name}; "
        f"target={reward_target_value:.4f}; "
        f"frozen_train={frozen_train_acc:.4f}; "
        f"frozen_test={frozen_test_acc:.4f}; "
        f"backbones=[{', '.join(backbone_names)}]; "
        f"backbone_bucket={backbone_signature}; "
        f"cnn={cnn_expr_short or cnn_signature or 'n/a'}; "
        f"graph={graph_expr_short}; "
        f"stats={stats_short}"
    )
    summary = _truncate_text(summary, FEEDBACK_SUMMARY_MAX_CHARS)
    return GroupFeedbackSummary(
        goal_key=goal_key,
        pattern_name=pattern_name,
        graph_expr_short=graph_expr_short,
        reward_target_value=reward_target_value,
        frozen_train_acc=frozen_train_acc,
        frozen_test_acc=frozen_test_acc,
        unfrozen_train_acc=unfrozen_train_acc,
        unfrozen_test_acc=unfrozen_test_acc,
        backbone_model_names=backbone_names,
        stats_short=stats_short,
        summary=summary,
        family_hash=str(getattr(graph_info, "family_hash", "") or res.get("family_hash") or ""),
        signature=str(res.get("signature") or ""),
        reward_group_id=reward_group_id,
        backbone_signature=backbone_signature,
        cnn_signature=cnn_signature,
        cnn_expr_short=cnn_expr_short,
    )


def _update_top_feedback(summary: GroupFeedbackSummary) -> None:
    current_group_top_feedback.append(summary)
    current_group_top_feedback.sort(key=lambda item: item.reward_target_value, reverse=True)
    del current_group_top_feedback[FEEDBACK_SUMMARY_LIMIT:]


def _record_current_group_trainable_sample(goal_key: str, res: Dict[str, Any], graph_info) -> None:
    reward_target_value = _result_reward_target_value(res)
    if reward_target_value is None:
        return
    current_best = current_group_goal_best_candidates.get(goal_key)
    if current_best is None or float(reward_target_value) > current_best:
        current_group_goal_best_candidates[goal_key] = float(reward_target_value)
    summary = _build_group_feedback_summary(
        goal_key=goal_key,
        res=res,
        graph_info=graph_info,
        reward_group_id=current_group_id,
    )
    _update_top_feedback(summary)


def _reset_current_group_feedback_state() -> None:
    current_group_top_feedback.clear()
    current_group_goal_best_candidates.clear()


def get_prompt_feedback_state() -> Dict[str, Any]:
    training_context = summarize_stage_training_context(current_stage_name)
    return {
        "prev_closed_group_mean_reward_target_acc": prev_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "prev_closed_group_mean_train_acc": prev_closed_group_train_acc_mean,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "prev_closed_group_mean_test_acc": prev_closed_group_mean_test_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "dominant_backbone_signature": dominant_backbone_signature,
        "dominant_backbone_share": dominant_backbone_share,
        "dominant_cnn_signature": dominant_cnn_signature,
        "dominant_cnn_share": dominant_cnn_share,
        "dominant_backbone_cnn_pair": dominant_backbone_cnn_pair,
        "dominant_backbone_cnn_share": dominant_backbone_cnn_share,
        "prev_group_feedback": _feedback_summary_payload(prev_group_feedback),
        "best_group_feedback": _feedback_summary_payload(best_group_feedback),
        "training_context": training_context,
    }


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_optional_signed_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def _format_target_metric(base_value: Optional[float], delta: float) -> str:
    if base_value is None:
        return "n/a"
    return f"{float(base_value) + float(delta):.4f}"


def render_prompt_feedback_text(*, feedback_char_budget: int = 1200) -> str:
    state = get_prompt_feedback_state()
    current_metric = _stage_reward_target_metric(current_stage_name)
    header_lines = [
        f"- Current Stage: {current_stage_name}",
        f"- Reward Target Metric: {current_metric}",
        f"- Previous Closed Group Mean Target Acc: {_format_optional_metric(state['prev_closed_group_mean_reward_target_acc'])}",
        f"- Current Best Closed Group Mean Target Acc: {_format_optional_metric(state['best_closed_group_mean_reward_target_acc'])}",
        f"- Previous Closed Group Mean Frozen Train Acc: {_format_optional_metric(state['prev_closed_group_mean_train_acc'])}",
        f"- Previous Closed Group Mean Frozen Test Acc: {_format_optional_metric(state['prev_closed_group_mean_test_acc'])}",
        (
            "- Current Dominant Family To Avoid When Not Improving: "
            f"{state['dominant_family_hash'] or 'n/a'} "
            f"(share={float(state['dominant_family_share'] or 0.0):.2%})"
        ),
        (
            "- Rule: same backbone pair is acceptable; compare new models mainly against that pair's own recent baseline"
        ),
        (
            "- Rule: within the same backbone pair, change stem/project/fuse CNN layout, not just widths, ordering, or formatting"
        ),
    ]
    if current_stage_name != STAGE1_STRUCTURE_EXPLORE:
        header_lines.extend(
            [
                f"- Meaningful Reward Target: >= {_format_target_metric(state['prev_closed_group_mean_reward_target_acc'], GROUP_IMPROVEMENT_DELTA)}",
                f"- Stretch Target To Refresh Best: >= {_format_target_metric(state['best_closed_group_mean_reward_target_acc'], BEST_GROUP_REFRESH_DELTA)}",
                "- Rule: prioritize higher frozen test accuracy, not just easier train accuracy",
                "- Rule: dominant-family reuse or plain classifier-only fuse below target is penalized",
            ]
        )
    if current_stage_name == STAGE2_FORMAL_EXPLORE:
        training_context = dict(state.get("training_context") or {})
        context_guidance = _training_context_guidance(training_context)
        header_lines.extend(
            [
                (
                    "- Current Training Context: "
                    f"last50 best_loss={_format_optional_metric(training_context.get('recent_best_loss'))}, "
                    f"delta_best={_format_optional_signed_metric(training_context.get('delta_best_loss'))}; "
                    f"last50 avg_loss={_format_optional_metric(training_context.get('recent_avg_loss'))}, "
                    f"delta_avg={_format_optional_signed_metric(training_context.get('delta_avg_loss'))}"
                ),
                (
                    "- Training Trend: "
                    f"slope={_format_optional_signed_metric(training_context.get('loss_slope_recent'))}/epoch, "
                    f"variance={_format_optional_metric(training_context.get('loss_variance_recent'))}, "
                    f"since_best={training_context.get('epochs_since_last_improvement', 'n/a')}, "
                    f"plateau={float(training_context.get('plateau_score') or 0.0):.2f}, "
                    f"oscillation={float(training_context.get('oscillation_score') or 0.0):.2f}"
                ),
                f"- Training Guidance: {context_guidance}",
            ]
        )

    prev_lines = [
        f"  - {item['summary']}"
        for item in state.get("prev_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]
    best_lines = [
        f"  - {item['summary']}"
        for item in state.get("best_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]

    def _compose_lines(current_prev_lines: List[str], current_best_lines: List[str]) -> str:
        lines = list(header_lines)
        if current_prev_lines:
            lines.append("- Previous Group Strong Examples:")
            lines.extend(current_prev_lines)
        else:
            lines.append("- Previous Group Strong Examples: none yet")

        if current_best_lines:
            lines.append("- Current Best Group Strong Examples:")
            lines.extend(current_best_lines)
        else:
            lines.append("- Current Best Group Strong Examples: none yet")
        return "\n".join(lines)

    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(best_lines) >= 2:
        best_lines = best_lines[:1]
    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(prev_lines) >= 2:
        prev_lines = prev_lines[:1]
        text = _compose_lines(prev_lines, best_lines)
    return _truncate_text(text, feedback_char_budget)


def reset_reward_runtime_state() -> None:
    StageState.reset_reward_runtime_state(
        globals(),
        stage1_structure_explore=STAGE1_STRUCTURE_EXPLORE,
        reset_current_group_feedback_state=_reset_current_group_feedback_state,
    )

def current_reward_group_context() -> Dict[str, Any]:
    return StageState.current_reward_group_context(sys.modules[__name__])


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


def _cuda_memory_gib() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated() / float(1024 ** 3)
        reserved = torch.cuda.memory_reserved() / float(1024 ** 3)
        return allocated, reserved
    except RuntimeError:
        return None, None


def _format_mem_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _visible_cuda_device_tokens() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return []
    raw = raw.strip()
    if raw in {"", "-1"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolved_train_gpu_index() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return None
    if _distributed_world_size() > 1:
        raw_local_rank = env_int("LOCAL_RANK", 0)
        if visible_gpu_count == 1:
            return 0
        if 0 <= raw_local_rank < visible_gpu_count:
            return raw_local_rank
    try:
        current_device = int(torch.cuda.current_device())
        if 0 <= current_device < visible_gpu_count:
            return current_device
    except Exception:
        pass
    raw_local_rank = env_int("LOCAL_RANK", 0)
    if visible_gpu_count == 1:
        return 0
    if 0 <= raw_local_rank < visible_gpu_count:
        return raw_local_rank
    return 0


def _visible_cuda_memory_snapshots(*, include_all_visible_gpus: bool) -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return []
    device_tokens = _visible_cuda_device_tokens()
    train_gpu_index = _resolved_train_gpu_index()
    if include_all_visible_gpus:
        device_indices = list(range(visible_gpu_count))
    elif train_gpu_index is not None:
        device_indices = [int(train_gpu_index)]
    else:
        device_indices = [0]

    snapshots: List[Dict[str, Any]] = []
    for device_index in device_indices:
        total_gib = None
        free_gib = None
        used_gib = None
        allocated_gib = None
        reserved_gib = None
        device_name = ""
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_gib = float(props.total_memory) / float(1024 ** 3)
            device_name = str(getattr(props, "name", "") or "")
        except Exception:
            pass
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

        other_used_gib = None
        if used_gib is not None and allocated_gib is not None:
            other_used_gib = max(0.0, float(used_gib) - float(allocated_gib))

        device_token = (
            device_tokens[device_index]
            if 0 <= device_index < len(device_tokens)
            else str(int(device_index))
        )
        snapshots.append(
            {
                "device_index": int(device_index),
                "device_token": str(device_token),
                "device_name": device_name,
                "total_gib": total_gib,
                "free_gib": free_gib,
                "used_gib": used_gib,
                "allocated_gib": allocated_gib,
                "reserved_gib": reserved_gib,
                "other_used_gib": other_used_gib,
                "is_train_gpu": bool(train_gpu_index is not None and int(train_gpu_index) == int(device_index)),
            }
        )
    return snapshots


def _current_cuda_allocator_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = _resolved_train_gpu_index()
    if current_device is None:
        return {}
    try:
        stats = torch.cuda.memory_stats(current_device)
    except Exception:
        return {}
    return {
        "current_device": int(current_device),
        "active_gib": float(stats.get("active_bytes.all.current", 0.0)) / float(1024 ** 3),
        "reserved_gib": float(stats.get("reserved_bytes.all.current", 0.0)) / float(1024 ** 3),
        "inactive_split_gib": float(stats.get("inactive_split_bytes.all.current", 0.0)) / float(1024 ** 3),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
    }


def _query_nvidia_smi_csv(query_kind: str, columns: List[str]) -> List[List[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        f"--query-{query_kind}={','.join(columns)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    reader = csv.reader(completed.stdout.splitlines())
    return [[cell.strip() for cell in row] for row in reader if row]


def _log_nvidia_smi_snapshot(stage: str) -> None:
    gpu_rows = _query_nvidia_smi_csv(
        "gpu",
        ["index", "uuid", "name", "memory.total", "memory.used", "memory.free"],
    )
    if not gpu_rows:
        print(f"[OOM nvidia-smi] stage={stage} unavailable=True")
        return

    visible_tokens = set(_visible_cuda_device_tokens())
    filter_visible = bool(visible_tokens)
    gpu_rows_by_uuid: Dict[str, Dict[str, str]] = {}
    for row in gpu_rows:
        if len(row) < 6:
            continue
        gpu_index, gpu_uuid, gpu_name, total_mib, used_mib, free_mib = row[:6]
        if filter_visible and gpu_index not in visible_tokens and gpu_uuid not in visible_tokens:
            continue
        gpu_rows_by_uuid[gpu_uuid] = {
            "index": gpu_index,
            "uuid": gpu_uuid,
            "name": gpu_name,
            "total_mib": total_mib,
            "used_mib": used_mib,
            "free_mib": free_mib,
        }
        print(
            "[OOM nvidia-smi GPU] "
            f"stage={stage} "
            f"gpu={gpu_index} "
            f"name={gpu_name!r} "
            f"used_mib={used_mib} "
            f"free_mib={free_mib} "
            f"total_mib={total_mib}"
        )

    process_rows = _query_nvidia_smi_csv(
        "compute-apps",
        ["gpu_uuid", "pid", "process_name", "used_memory"],
    )
    process_snapshots: List[Dict[str, Any]] = []
    for row in process_rows:
        if len(row) < 4:
            continue
        gpu_uuid, pid, process_name, used_mib = row[:4]
        gpu_info = gpu_rows_by_uuid.get(gpu_uuid)
        if gpu_info is None:
            continue
        try:
            used_mib_value = int(float(used_mib))
        except (TypeError, ValueError):
            used_mib_value = 0
        process_snapshots.append(
            {
                "gpu": gpu_info["index"],
                "pid": pid,
                "process_name": process_name,
                "used_mib": used_mib_value,
            }
        )
    process_snapshots.sort(key=lambda item: int(item["used_mib"]), reverse=True)
    for snapshot in process_snapshots[:24]:
        print(
            "[OOM nvidia-smi Proc] "
            f"stage={stage} "
            f"gpu={snapshot['gpu']} "
            f"pid={snapshot['pid']} "
            f"used_mib={snapshot['used_mib']} "
            f"process={snapshot['process_name']!r}"
        )


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    normalized = " ".join(str(exc).split()).lower()
    return "out of memory" in normalized and "cuda" in normalized


def log_cuda_oom_diagnostics(
    stage: str,
    exc: BaseException,
    *,
    group_context: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"[OOM] stage={stage} error={type(exc).__name__}: {exc}")
    log_memory_snapshot(stage, group_context=group_context, include_all_visible_gpus=True)
    allocator_snapshot = _current_cuda_allocator_snapshot()
    if allocator_snapshot:
        print(
            "[OOM Allocator] "
            f"stage={stage} "
            f"current_device={allocator_snapshot['current_device']} "
            f"active_gib={_format_mem_value(allocator_snapshot['active_gib'])} "
            f"reserved_gib={_format_mem_value(allocator_snapshot['reserved_gib'])} "
            f"inactive_split_gib={_format_mem_value(allocator_snapshot['inactive_split_gib'])} "
            f"num_ooms={allocator_snapshot['num_ooms']} "
            f"num_alloc_retries={allocator_snapshot['num_alloc_retries']}"
        )
    _log_nvidia_smi_snapshot(stage)


class _CudaMemoryMonitor:
    def __init__(self, stage_prefix: str) -> None:
        self._stage_prefix = str(stage_prefix)
        self._enabled = bool(torch.cuda.is_available()) and env_int("NNGPT_CUDA_MEMORY_MONITOR", 1) > 0
        self._interval_seconds = max(1.0, env_float("NNGPT_CUDA_MEMORY_MONITOR_INTERVAL_SECONDS", 30.0))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional["_CudaMemoryMonitor"]:
        if not self._enabled:
            return None
        print(
            "[Memory Monitor] "
            f"stage_prefix={self._stage_prefix} "
            f"interval_seconds={self._interval_seconds:.1f}"
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"nngpt-cuda-memory-monitor-{self._stage_prefix}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                log_memory_snapshot(f"{self._stage_prefix}:tick")
            except Exception as exc:
                print(
                    "[Memory Monitor] "
                    f"stage_prefix={self._stage_prefix} "
                    f"error={type(exc).__name__}: {exc}"
                )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self._interval_seconds + 1.0))
        self._thread = None


def start_cuda_memory_monitor(stage_prefix: str) -> Optional[_CudaMemoryMonitor]:
    monitor = _CudaMemoryMonitor(stage_prefix)
    return monitor.start()


def log_memory_snapshot(
    stage: str,
    *,
    group_context: Optional[Dict[str, Any]] = None,
    include_all_visible_gpus: Optional[bool] = None,
) -> None:
    effective_group_context = group_context or current_reward_group_context()
    cuda_allocated_gib, cuda_reserved_gib = _cuda_memory_gib()
    worker_info = get_eval_worker_diagnostics()
    worker_pid = worker_info.get("worker_pids", [worker_info.get("pid")]) if worker_info else None
    rank = _distributed_rank()
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = _distributed_world_size()
    train_gpu = _resolved_train_gpu_index()
    if include_all_visible_gpus is None:
        # In single-process SFT/RL runs, touching every visible GPU here creates
        # extra CUDA contexts on reward GPUs. Default to the training GPU only
        # unless a caller explicitly asks for a full visible-device snapshot.
        include_all_visible_gpus = bool(world_size > 1 and is_main_process())
    visible_cuda_snapshots = _visible_cuda_memory_snapshots(
        include_all_visible_gpus=bool(include_all_visible_gpus)
    )
    print(
        "[Memory] "
        f"stage={stage} "
        f"pid={os.getpid()} "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"world_size={world_size} "
        f"train_gpu={train_gpu} "
        f"reward_batch_index={effective_group_context.get('reward_batch_index')} "
        f"reward_group_id={effective_group_context.get('reward_group_id')} "
        f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
        f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
        f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
        f"worker_pid={worker_pid}"
    )
    for snapshot in visible_cuda_snapshots:
        train_gpu_marker = "*" if snapshot.get("is_train_gpu") else ""
        print(
            "[Memory GPU] "
            f"stage={stage} "
            f"gpu={snapshot['device_index']}{train_gpu_marker} "
            f"token={snapshot['device_token']} "
            f"name={snapshot['device_name']!r} "
            f"free_gib={_format_mem_value(snapshot['free_gib'])} "
            f"used_gib={_format_mem_value(snapshot['used_gib'])} "
            f"total_gib={_format_mem_value(snapshot['total_gib'])} "
            f"proc_allocated_gib={_format_mem_value(snapshot['allocated_gib'])} "
            f"proc_reserved_gib={_format_mem_value(snapshot['reserved_gib'])} "
            f"other_used_gib={_format_mem_value(snapshot['other_used_gib'])}"
        )
def update_current_group_metrics(results: List[Dict[str, Any]]) -> None:
    for res in results:
        reward_target_value = _result_reward_target_value(res)
        _increment_optional_metric(
            "current_group_reward_target_sum",
            "current_group_reward_target_count",
            reward_target_value,
        )
        backbone_signature = _result_backbone_signature(res)
        if reward_target_value is not None and backbone_signature:
            current_group_reward_target_sum_by_backbone[backbone_signature] = (
                float(current_group_reward_target_sum_by_backbone.get(backbone_signature, 0.0))
                + float(reward_target_value)
            )
            current_group_reward_target_count_by_backbone[backbone_signature] += 1
        _increment_optional_metric(
            "current_group_frozen_train_acc_sum",
            "current_group_frozen_train_acc_count",
            _optional_float(res.get("frozen_train_acc", res.get("train_acc"))),
        )
        _increment_optional_metric(
            "current_group_frozen_test_acc_sum",
            "current_group_frozen_test_acc_count",
            _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))),
        )
        _increment_optional_metric(
            "current_group_unfrozen_train_acc_sum",
            "current_group_unfrozen_train_acc_count",
            _optional_float(res.get("unfrozen_train_acc")),
        )
        _increment_optional_metric(
            "current_group_unfrozen_test_acc_sum",
            "current_group_unfrozen_test_acc_count",
            _optional_float(res.get("unfrozen_test_acc")),
        )


def _reset_stage_comparison_state() -> None:
    global prev_closed_group_mean_reward_target_acc
    global best_closed_group_mean_reward_target_acc
    global prev_closed_group_train_acc_mean
    global best_closed_group_mean_train_acc
    global prev_closed_group_mean_test_acc
    global best_closed_group_mean_test_acc
    global best_closed_group_id

    prev_closed_group_mean_reward_target_acc = None
    best_closed_group_mean_reward_target_acc = None
    prev_closed_group_train_acc_mean = None
    best_closed_group_mean_train_acc = None
    prev_closed_group_mean_test_acc = None
    best_closed_group_mean_test_acc = None
    best_closed_group_id = None
    prev_closed_group_mean_reward_target_by_backbone.clear()
    best_closed_group_mean_reward_target_by_backbone.clear()
    best_reward_target_by_goal.clear()
    prev_group_feedback.clear()
    best_group_feedback.clear()
    _reset_current_group_feedback_state()


def _stage_checkpoint_root() -> Path:
    return StageState.stage_checkpoint_root(sys.modules[__name__])


def _stage_checkpoint_dir(stage_name: str) -> Path:
    return StageState.stage_checkpoint_dir(sys.modules[__name__], stage_name)


def _stage_group_snapshot_payload(current_group_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return StageState.stage_group_snapshot_payload(sys.modules[__name__], current_group_payload)


def _save_stage_plot_snapshot(output_path: Path) -> None:
    return StageState.save_stage_plot_snapshot(sys.modules[__name__], output_path)


def _save_stage_checkpoint(
    event: str,
    *,
    stage_name: Optional[str] = None,
    group_progress_payload: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    save_plot_snapshot: bool = True,
) -> Optional[Path]:
    return StageState.save_stage_checkpoint(
        sys.modules[__name__],
        event,
        stage_name=stage_name,
        group_progress_payload=group_progress_payload,
        reason=reason,
        save_plot_snapshot=save_plot_snapshot,
    )


def _handle_checkpoint_signal(signum: int, _frame) -> None:
    global _signal_checkpoint_in_progress

    if _signal_checkpoint_in_progress:
        raise SystemExit(128 + int(signum))

    _signal_checkpoint_in_progress = True
    signal_name = signal.Signals(signum).name.lower()
    try:
        _save_stage_checkpoint(
            "signal",
            stage_name=current_stage_name,
            reason=f"signal_{signal_name}",
            save_plot_snapshot=False,
        )
        try:
            code_logger.save_log()
        except Exception as exc:
            code_logger.log_to_file(f"[Signal Save] save_log failed: {type(exc).__name__}: {exc}")
    finally:
        signal.signal(signum, signal.SIG_DFL)
        _signal_checkpoint_in_progress = False

    raise SystemExit(128 + int(signum))


def register_stage_checkpoint_signal_handlers() -> None:
    if not is_main_process():
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        if signum in _registered_signal_handlers:
            continue
        _registered_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_checkpoint_signal)


def _stage1_gate_ready() -> bool:
    recent_generations = _recent_stage_generation_window(STAGE1_STRUCTURE_EXPLORE, STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(_recent_stage_group_window(STAGE1_STRUCTURE_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE1_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < STAGE1_PROMOTION_MIN_GROUPS:
        return False
    executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    unique_discovery_families = len(_family_hash_set(discovery_rows, key="family_hash"))
    return bool(
        executable_count >= STAGE1_GATE_EXECUTABLE_MIN
        and len(discovery_rows) >= STAGE1_GATE_DISCOVERY_MIN
        and unique_discovery_families >= STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN
    )


def _stage1_force_promotion_ready() -> Optional[Dict[str, int]]:
    recent_generations = _recent_stage_generation_window(STAGE1_STRUCTURE_EXPLORE, STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(_recent_stage_group_window(STAGE1_STRUCTURE_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE1_GATE_WINDOW_GENERATIONS:
        return None
    if current_entry_group_count < STAGE1_PROMOTION_MIN_GROUPS:
        return None
    recent_executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    recent_discovery_count = len(discovery_rows)
    recent_unique_discovery_families = len(_family_hash_set(discovery_rows, key="family_hash"))
    if recent_executable_count < STAGE1_FORCE_PROMOTION_EXECUTABLE_MIN:
        return None
    if recent_discovery_count < STAGE1_FORCE_PROMOTION_DISCOVERY_MIN:
        return None
    if recent_unique_discovery_families < STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN:
        return None
    return {
        "stage_group_count": current_entry_group_count,
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": recent_executable_count,
        "recent_discovery_count": recent_discovery_count,
        "recent_unique_discovery_families": recent_unique_discovery_families,
    }


def _stage2_gate_ready() -> bool:
    recent_generations = _recent_stage_generation_window(STAGE2_FORMAL_EXPLORE, STAGE2_GATE_WINDOW_GENERATIONS)
    recent_groups = _recent_stage_group_window(STAGE2_FORMAL_EXPLORE, 5)
    recent_improvement_groups = _recent_stage_group_window(STAGE2_FORMAL_EXPLORE, 4)
    current_entry_group_count = len(_recent_stage_group_window(STAGE2_FORMAL_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE2_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < STAGE_REFERENCE_MIN_GROUPS[STAGE2_FORMAL_EXPLORE]:
        return False
    formal_rows = [item for item in recent_generations if bool(item.get("formal_success_candidate"))]
    qualified_rows = _stage2_target_qualified_rows(recent_generations)
    unique_target_families = len(_family_hash_set(qualified_rows, key="family_hash"))
    mean_dominant_share = _mean_dominant_share(recent_groups)
    mean_dominant_descriptor_share = _mean_dominant_descriptor_share(recent_groups)
    improving_groups = _count_group_improvements(recent_improvement_groups)
    return bool(
        len(qualified_rows) >= STAGE2_GATE_MIN_TARGET_COUNT
        and unique_target_families >= STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES
        and improving_groups >= STAGE2_GATE_IMPROVING_GROUPS_REQUIRED
        and mean_dominant_share is not None
        and mean_dominant_share <= 0.45
        and mean_dominant_descriptor_share is not None
        and mean_dominant_descriptor_share <= STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE
    )


def _stage_gate_snapshot() -> Dict[str, Any]:
    stage_name = str(current_stage_name)
    recent_generations = _recent_stage_generation_window(
        stage_name,
        STAGE1_GATE_WINDOW_GENERATIONS if stage_name == STAGE1_STRUCTURE_EXPLORE else STAGE2_GATE_WINDOW_GENERATIONS,
    )
    recent_groups = _recent_stage_group_window(stage_name, 5)
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    formal_rows = [item for item in recent_generations if bool(item.get("formal_success_candidate"))]
    qualified_target_rows = _stage2_target_qualified_rows(recent_generations)
    return {
        "stage_name": stage_name,
        "stage_index": RL_STAGE_TO_INDEX.get(stage_name, 0),
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": sum(1 for item in recent_generations if bool(item.get("executable_candidate"))),
        "recent_discovery_count": len(discovery_rows),
        "recent_unique_discovery_families": len(_family_hash_set(discovery_rows, key="family_hash")),
        "recent_formal_success_count": len(formal_rows),
        "recent_unique_formal_families": len(_family_hash_set(formal_rows, key="family_hash")),
        "recent_target_qualified_count": len(qualified_target_rows),
        "recent_unique_target_families": len(_family_hash_set(qualified_target_rows, key="family_hash")),
        "recent_unique_backbone_signatures": len(_family_hash_set(recent_generations, key="backbone_signature")),
        "recent_unique_cnn_signatures": len(_family_hash_set(recent_generations, key="cnn_signature")),
        "recent_unique_backbone_cnn_pairs": len(_family_hash_set(recent_generations, key="backbone_cnn_pair_key")),
        "recent_mean_dominant_family_share": _mean_dominant_share(recent_groups),
        "recent_mean_dominant_descriptor_share": _mean_dominant_descriptor_share(recent_groups),
        "recent_mean_dominant_backbone_share": _mean_dominant_backbone_share(recent_groups),
        "recent_mean_dominant_cnn_share": _mean_dominant_cnn_share(recent_groups),
        "recent_mean_dominant_backbone_cnn_share": _mean_dominant_backbone_cnn_share(recent_groups),
        "recent_improving_groups": _count_group_improvements(_recent_stage_group_window(stage_name, 4)),
        "recovery_active": bool(recovery_active),
    }


def _transition_to_stage(
    new_stage_name: str,
    *,
    event: str,
    reason: str,
    group_progress_payload: Optional[Dict[str, Any]] = None,
) -> None:
    return StageState.transition_to_stage(
        sys.modules[__name__],
        new_stage_name,
        event=event,
        reason=reason,
        group_progress_payload=group_progress_payload,
    )


def _maybe_update_stage_best_checkpoint(group_progress_payload: Dict[str, Any]) -> None:
    return StageState.maybe_update_stage_best_checkpoint(sys.modules[__name__], group_progress_payload)


def _evaluate_stage_transitions(group_progress_payload: Dict[str, Any]) -> None:
    return StageState.evaluate_stage_transitions(sys.modules[__name__], group_progress_payload)


def close_reward_group_if_needed() -> Optional[Dict[str, Any]]:
    return StageState.close_reward_group_if_needed(sys.modules[__name__])


def _coerce_accuracy_baseline(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context}: missing required sample accuracy baseline")
    if isinstance(value, bool):
        raise ValueError(f"{context}: accuracy baseline must be numeric, got bool")
    try:
        baseline = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: accuracy baseline must be numeric, got {value!r}") from exc
    if baseline != baseline or baseline in {float("inf"), float("-inf")}:
        raise ValueError(f"{context}: accuracy baseline must be finite, got {value!r}")
    return baseline


def require_sample_accuracy_baselines(kwargs: Dict[str, Any], expected_count: int) -> List[float]:
    if "accuracy" not in kwargs:
        raise ValueError("compute_reward requires kwargs['accuracy'] for every sample")
    raw_values = kwargs["accuracy"]
    if len(raw_values) != expected_count:
        raise ValueError(
            f"compute_reward expected {expected_count} accuracy baselines, got {len(raw_values)}"
        )
    return [
        _coerce_accuracy_baseline(value, context=f"completion[{idx}]")
        for idx, value in enumerate(raw_values)
    ]


def run_epoch_dir(*args):
    root_override = os.getenv("NNGPT_RL_EPOCH_ROOT")
    if root_override:
        e_dir = Path(root_override)
        for d in args:
            e_dir = e_dir / f"A{d}"
        return e_dir
    return epoch_dir(*args)


def run_log_dir() -> str:
    return os.getenv("NNGPT_RL_LOG_DIR", "rl_output")


def run_model_out() -> str:
    return os.getenv("NNGPT_RL_MODEL_OUT", SAVED_MODEL_PATH)


def _resolve_resume_checkpoint_dir() -> Optional[Path]:
    explicit_dir = os.getenv("NNGPT_RL_RESUME_CHECKPOINT_DIR", "").strip()
    resume_stage = os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()
    if resume_stage:
        return _stage_checkpoint_dir(resume_stage)
    return None


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _current_stage_index() -> int:
    return StageState.current_stage_index(sys.modules[__name__])


def _history_trim_in_place(items: List[Dict[str, Any]], *, limit: int) -> None:
    return StageState.history_trim_in_place(sys.modules[__name__], items, limit=limit)


def _append_stage_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.append_stage_event(sys.modules[__name__], payload)


def _record_generation_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.record_generation_event(sys.modules[__name__], payload)


def _record_closed_group_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.record_closed_group_event(sys.modules[__name__], payload)


def _recent_stage_generation_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    return StageState.recent_stage_generation_window(sys.modules[__name__], stage_name, max_items)


def _recent_stage_group_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    return StageState.recent_stage_group_window(sys.modules[__name__], stage_name, max_items)


def _family_hash_set(items: List[Dict[str, Any]], *, key: str) -> Set[str]:
    return {
        str(item.get(key))
        for item in items
        if item.get(key)
    }


def _mean_dominant_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_family_share"))
        for item in items
        if item.get("dominant_family_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_descriptor_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_descriptor_share"))
        for item in items
        if item.get("dominant_descriptor_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_backbone_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_backbone_share"))
        for item in items
        if item.get("dominant_backbone_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_cnn_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_cnn_share"))
        for item in items
        if item.get("dominant_cnn_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_backbone_cnn_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_backbone_cnn_share"))
        for item in items
        if item.get("dominant_backbone_cnn_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _count_group_improvements(items: List[Dict[str, Any]]) -> int:
    count = 0
    for item in items:
        improvement_vs_prev = item.get("improvement_vs_prev")
        if improvement_vs_prev is None:
            continue
        if float(improvement_vs_prev) >= GROUP_IMPROVEMENT_DELTA:
            count += 1
    return count


def _training_context_metric_from_event(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    best_epoch_loss = _optional_float(item.get("best_epoch_loss"))
    if best_epoch_loss is not None:
        return "best_epoch_loss", best_epoch_loss
    loss_end = _optional_float(item.get("loss_end"))
    if loss_end is not None:
        return "loss_end", loss_end
    metric_name = str(item.get("training_context_metric_name") or "").strip()
    metric_value = _optional_float(item.get("training_context_metric_value"))
    if metric_name and metric_value is not None:
        return metric_name, metric_value
    return None, None


def _recent_stage_trainable_metric_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in _recent_stage_generation_window(stage_name, max_items):
        metric_name, metric_value = _training_context_metric_from_event(item)
        if metric_value is None:
            continue
        if not bool(item.get("backward_ok") or item.get("trained_step_ok") or item.get("loss_drop_ok")):
            continue
        epochs_completed = max(1, int(item.get("epochs_completed", 0) or 1))
        records.append(
            {
                "generation_total": int(item.get("generation_total", 0) or 0),
                "metric_name": metric_name or "best_epoch_loss",
                "metric_value": float(metric_value),
                "epochs_completed": epochs_completed,
            }
        )
    return records


def _series_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _series_variance(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mean_value = _series_mean(values)
    if mean_value is None:
        return None
    return float(sum((float(value) - mean_value) ** 2 for value in values) / len(values))


def _series_slope(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    n = len(values)
    x_mean = float(n - 1) / 2.0
    y_mean = _series_mean(values)
    if y_mean is None:
        return None
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        x_delta = float(index) - x_mean
        numerator += x_delta * (float(value) - y_mean)
        denominator += x_delta * x_delta
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _epochs_since_last_best(records: List[Dict[str, Any]]) -> Optional[int]:
    if not records:
        return None
    best_value = None
    total_epochs = 0
    last_best_epoch = 0
    for item in records:
        epochs_completed = max(1, int(item.get("epochs_completed", 0) or 1))
        total_epochs += epochs_completed
        metric_value = float(item["metric_value"])
        if best_value is None or metric_value < best_value - 1e-8:
            best_value = metric_value
            last_best_epoch = total_epochs
    return max(0, total_epochs - last_best_epoch)


def _history_exploration_pressure_from_summary(summary: Dict[str, Any]) -> float:
    if not summary.get("has_recent_window"):
        return 0.0
    pressure = 0.0
    delta_avg_loss = _optional_float(summary.get("delta_avg_loss"))
    if delta_avg_loss is not None:
        if delta_avg_loss >= 0.0:
            pressure += 0.35
        elif delta_avg_loss >= -0.01:
            pressure += 0.20
    loss_slope_recent = _optional_float(summary.get("loss_slope_recent"))
    if loss_slope_recent is not None:
        if loss_slope_recent >= 0.0:
            pressure += 0.25
        elif loss_slope_recent >= -5e-4:
            pressure += 0.12
    pressure += 0.30 * float(summary.get("plateau_score") or 0.0)
    pressure += 0.18 * float(summary.get("oscillation_score") or 0.0)
    epochs_since_last_improvement = summary.get("epochs_since_last_improvement")
    recent_window_epochs = max(1, int(summary.get("recent_window_epochs", 0) or 1))
    if epochs_since_last_improvement is not None:
        pressure += 0.25 * min(1.0, float(epochs_since_last_improvement) / float(recent_window_epochs))
    return _clip(pressure, 0.0, 1.0)


def summarize_stage_training_context(
    stage_name: str,
    *,
    window_size: int = TRAINING_CONTEXT_WINDOW,
) -> Dict[str, Any]:
    effective_window = max(1, int(window_size))
    records = _recent_stage_trainable_metric_window(stage_name, max_items=max(effective_window * 4, effective_window))
    recent_records = records[-effective_window:]
    prev_records = records[-(effective_window * 2):-effective_window] if len(records) > effective_window else []
    recent_values = [float(item["metric_value"]) for item in recent_records]
    prev_values = [float(item["metric_value"]) for item in prev_records]
    recent_window_epochs = sum(max(1, int(item.get("epochs_completed", 0) or 1)) for item in recent_records)
    prev_window_epochs = sum(max(1, int(item.get("epochs_completed", 0) or 1)) for item in prev_records)
    recent_best_loss = min(recent_values) if recent_values else None
    prev_best_loss = min(prev_values) if prev_values else None
    recent_avg_loss = _series_mean(recent_values)
    prev_avg_loss = _series_mean(prev_values)
    delta_best_loss = (
        float(recent_best_loss - prev_best_loss)
        if recent_best_loss is not None and prev_best_loss is not None
        else None
    )
    delta_avg_loss = (
        float(recent_avg_loss - prev_avg_loss)
        if recent_avg_loss is not None and prev_avg_loss is not None
        else None
    )
    improvement_rate = (
        float((prev_avg_loss - recent_avg_loss) / float(max(1, recent_window_epochs)))
        if recent_avg_loss is not None and prev_avg_loss is not None
        else None
    )
    loss_slope_recent = _series_slope(recent_values)
    loss_variance_recent = _series_variance(recent_values)
    epochs_since_last_improvement = _epochs_since_last_best(records)
    recent_range = (max(recent_values) - min(recent_values)) if len(recent_values) >= 2 else 0.0
    recent_scale = max(1e-6, abs(recent_avg_loss if recent_avg_loss is not None else (recent_best_loss or 1.0)))
    normalized_slope = 0.0
    if loss_slope_recent is not None:
        normalized_slope = min(1.0, abs(float(loss_slope_recent)) * float(max(1, len(recent_values))) / recent_scale)
    normalized_range = min(1.0, float(recent_range) / recent_scale)
    plateau_score = _clip(1.0 - min(1.0, 0.65 * normalized_slope + 0.35 * normalized_range), 0.0, 1.0)
    diffs = [recent_values[index + 1] - recent_values[index] for index in range(max(0, len(recent_values) - 1))]
    nontrivial_diffs = [float(value) for value in diffs if abs(float(value)) > 1e-8]
    diff_signs = [1 if value > 0.0 else -1 for value in nontrivial_diffs]
    oscillation_score = 0.0
    if len(diff_signs) >= 2:
        sign_changes = sum(1 for left, right in zip(diff_signs, diff_signs[1:]) if left != right)
        oscillation_score = float(sign_changes) / float(len(diff_signs) - 1)
    monotonic_improving = bool(nontrivial_diffs) and all(value < 0.0 for value in nontrivial_diffs)
    metric_name = recent_records[-1]["metric_name"] if recent_records else "best_epoch_loss"
    summary = {
        "stage_name": str(stage_name),
        "metric_name": metric_name,
        "sample_count": len(records),
        "recent_window_size": len(recent_records),
        "compare_window_size": len(prev_records),
        "recent_window_epochs": recent_window_epochs,
        "compare_window_epochs": prev_window_epochs,
        "has_recent_window": len(recent_records) >= max(1, TRAINING_CONTEXT_MIN_POINTS),
        "has_compare_window": len(prev_records) >= max(1, TRAINING_CONTEXT_MIN_POINTS),
        "recent_best_loss": recent_best_loss,
        "prev_best_loss": prev_best_loss,
        "delta_best_loss": delta_best_loss,
        "recent_avg_loss": recent_avg_loss,
        "prev_avg_loss": prev_avg_loss,
        "delta_avg_loss": delta_avg_loss,
        "improvement_rate": improvement_rate,
        "loss_slope_recent": loss_slope_recent,
        "loss_variance_recent": loss_variance_recent,
        "epochs_since_last_improvement": epochs_since_last_improvement,
        "plateau_score": plateau_score,
        "oscillation_score": oscillation_score,
        "monotonic_improving": monotonic_improving,
    }
    summary["exploration_pressure"] = _history_exploration_pressure_from_summary(summary)
    return summary


def _training_context_guidance(summary: Dict[str, Any]) -> str:
    if not summary.get("has_recent_window"):
        return "no train-loss window yet"
    pressure = float(summary.get("exploration_pressure") or 0.0)
    if pressure >= 0.60:
        return "loss has plateaued or oscillated; favor structurally new candidates and avoid dominant templates"
    if bool(summary.get("monotonic_improving")) and pressure <= 0.25:
        return "loss is still improving; keep local mutations and avoid collapsing to one family"
    return "improvement is slowing; bias toward descriptor and family novelty over shallow repeats"


def _stage_reward_target_metric(stage_name: str) -> str:
    if str(stage_name) == STAGE1_STRUCTURE_EXPLORE:
        return STATIC_STAGE_REWARD_TARGET_METRIC
    return FORMAL_STAGE_REWARD_TARGET_METRIC


def _stage_uses_formal_eval(stage_name: str) -> bool:
    return str(stage_name) in {STAGE2_FORMAL_EXPLORE, STAGE3_FORMAL_OPTIMIZE}


def _stage_uses_static_only(stage_name: str) -> bool:
    return str(stage_name) == STAGE1_STRUCTURE_EXPLORE


def _iter_text_candidates(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_iter_text_candidates(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_iter_text_candidates(item))
        return out
    return []


def _score_seed_source_candidate(field_name: str, text: str) -> int:
    lowered_field = (field_name or "").lower()
    lowered_text = text.lower()
    score = 0
    if "<init>" in lowered_text and "<forward>" in lowered_text:
        score += 100
    if "class net" in lowered_text and "def forward" in lowered_text:
        score += 80
    if "def __init__" in lowered_text and "def forward" in lowered_text:
        score += 60
    if any(token in lowered_field for token in ("completion", "response", "output", "assistant", "xml")):
        score += 25
    if any(token in lowered_field for token in ("code", "nn", "model", "content", "text")):
        score += 10
    if len(text) > 200:
        score += 5
    return score


def _extract_method_from_module_text(source_text: str, class_name: str, method_name: str) -> str:
    try:
        tree = ast.parse(source_text)
    except Exception:
        return ""

    lines = source_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    if item.end_lineno is None:
                        return ""
                    snippet = "\n".join(lines[item.lineno - 1:item.end_lineno])
                    return textwrap.dedent(snippet).strip()
    return ""


def _extract_seed_init_forward_from_text(text: str) -> Tuple[str, str]:
    candidate = clean_block(text)
    if not candidate:
        return "", ""

    _, init_code, forward_code = extract_completion_blocks(candidate)
    if init_code and forward_code:
        return init_code, forward_code

    stripped = candidate.replace("```python", "").replace("```", "").strip()
    init_code = _extract_method_from_module_text(stripped, "Net", "__init__")
    forward_code = _extract_method_from_module_text(stripped, "Net", "forward")
    return init_code, forward_code


def _extract_seed_candidates_from_row(row: Any) -> List[str]:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    ranked: List[Tuple[int, str]] = []
    seen: Set[str] = set()

    for key, value in row_dict.items():
        for text in _iter_text_candidates(value):
            stripped = text.strip()
            if not stripped or stripped in seen:
                continue
            score = _score_seed_source_candidate(str(key), stripped)
            if score <= 0:
                continue
            ranked.append((score, stripped))
            seen.add(stripped)

    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    return [text for _, text in ranked]


def bootstrap_trainset_reference_library(data) -> None:
    train_graph_hashes.clear()
    train_family_hashes.clear()
    train_descriptor_keys.clear()

    stats = {
        "rows_seen": 0,
        "rows_parsed": 0,
        "rows_skipped": 0,
        "candidate_texts": 0,
    }

    for _, row in data.iterrows():
        stats["rows_seen"] += 1
        candidates = _extract_seed_candidates_from_row(row)
        stats["candidate_texts"] += len(candidates)
        parsed_ok = False

        for candidate in candidates:
            init_code, forward_code = _extract_seed_init_forward_from_text(candidate)
            if not init_code or not forward_code:
                continue
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
            if not graph_info.parse_ok:
                continue
            train_graph_hashes.add(graph_info.graph_hash)
            train_family_hashes.add(graph_info.family_hash)
            train_descriptor_keys.add(graph_info.descriptor_key)
            parsed_ok = True
            break

        if parsed_ok:
            stats["rows_parsed"] += 1
        else:
            stats["rows_skipped"] += 1

    train_reference_stats.clear()
    train_reference_stats.update(stats)
    print(
        "[Trainset Reference] "
        f"rows={stats['rows_seen']}, parsed={stats['rows_parsed']}, skipped={stats['rows_skipped']}, "
        f"graph_hashes={len(train_graph_hashes)}, family_hashes={len(train_family_hashes)}, "
        f"descriptor_keys={len(train_descriptor_keys)}"
    )


def extract_prompt_goal_tags(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    match = re.search(
        r"(?:^|\n)\s*(?:-\s*)?(?:(?:Discovery|Optimization)\s+)?Target Tags:\s*([A-Za-z0-9_, \-]+)",
        prompt_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [tag.strip() for tag in match.group(1).split(",") if tag.strip()]


def prompt_goal_satisfied(graph_info, tag: str) -> bool:
    if not graph_info or not graph_info.parse_ok:
        return False
    if tag == "stem":
        return graph_info.stem_calls > 0
    if tag == "project":
        return graph_info.project_calls > 0
    if tag == "multi_stage":
        return is_multi_stage_architecture(graph_info)
    if tag == "fractal_deep":
        return graph_info.fractal_calls >= 2 or (graph_info.fractal_calls >= 1 and graph_info.depth >= 5)
    if tag == "branch_reuse":
        return graph_info.merges >= 2 or (graph_info.project_calls > 0 and graph_info.fuse_calls >= 2)
    if tag == "single_backbone":
        return graph_info.backbone_calls == 1
    if tag == "wide_fuse":
        return graph_info.max_fan_in >= 3 and graph_info.fuse_calls >= 1
    return False


def primary_goal_key(prompt_goal_tags: List[str]) -> str:
    return "__".join(prompt_goal_tags or ["open"])


def goal_family_save_cap(graph_info) -> int:
    return 2


def get_goal_counter(store: Dict[str, Counter], goal_key: str) -> Counter:
    if goal_key not in store:
        store[goal_key] = Counter()
    return store[goal_key]


def clean_block(text: str) -> str:
    """Remove common LLM artifacts like markdown code blocks."""
    if not text: return ""
    text = text.strip()
    # Remove ```python ... ```
    text = re.sub(r'^```python\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

def extract_completion_blocks(completion: str) -> Tuple[str, str, str]:
    """Extract the three XML code blocks and normalize their formatting."""
    block_code = clean_block(extract_str(completion, '<block>', '</block>'))
    init_code = clean_block(extract_str(completion, '<init>', '</init>'))
    forward_code = clean_block(extract_str(completion, '<forward>', '</forward>'))
    return block_code, init_code, forward_code


def render_completion_xml(block_code: str, init_code: str, forward_code: str) -> str:
    return "\n".join(
        [
            "<block>",
            textwrap.dedent(block_code).strip(),
            "</block>",
            "<init>",
            textwrap.dedent(init_code).strip(),
            "</init>",
            "<forward>",
            textwrap.dedent(forward_code).strip(),
            "</forward>",
        ]
    )


def _extract_backbone_model_names(init_code: str) -> List[str]:
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in ("backbone_a", "backbone_b") if name in matches]


def _entry_backbone_model_names(entry: Dict[str, Any]) -> List[str]:
    backbone_names = list(entry.get("backbone_model_names") or [])
    if backbone_names:
        return backbone_names
    _, init_code, _ = extract_completion_blocks(str(entry.get("completion") or ""))
    return _extract_backbone_model_names(init_code)


def _entry_backbone_signature(entry: Dict[str, Any]) -> str:
    signature = str(entry.get("backbone_signature") or "").strip()
    if signature:
        return signature
    return build_backbone_signature(_entry_backbone_model_names(entry))


def _entry_cnn_signature(entry: Dict[str, Any]) -> str:
    signature = str(entry.get("cnn_signature") or "").strip()
    if signature:
        return signature
    graph_info = entry.get("graph_info")
    if graph_info is not None:
        signature = str(getattr(graph_info, "cnn_signature", "") or "").strip()
        if signature:
            return signature
    return "incomplete_cnn"


def reconstruct_code(
    completion: str,
    *,
    pattern_name_override: str = "",
) -> str:
    """Rebuild a runnable Python module from the XML blocks."""
    block_code, init_code, forward_code = extract_completion_blocks(completion)
    if not block_code or not init_code or not forward_code:
        return ""

    if pattern_name_override:
        init_code = ensure_pattern_name(init_code, pattern_name_override)

    code = SFTUtil.open_discovery_skeleton_code
    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
    code = code.replace(sig_block, textwrap.dedent(block_code))

    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
    code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))

    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
    code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
    return code


def _compute_build_partial_reward(res: Dict[str, Any]) -> float:
    error_str = str(res.get('error', ''))
    error_lower = error_str.lower()
    error_stage = str(res.get("error_stage") or "")
    error_context = dict(res.get("error_context") or {})
    code_trace = dict(error_context.get("code_trace") or {})
    raw_extraction = dict(res.get("raw_extraction") or {})
    build_partial = 0.0

    if error_stage == "cpu_prevalidate":
        if "must call self.infer_dimensions_dynamically" in error_str:
            build_partial = 0.00
        elif "infer_dimensions_dynamically() takes 2 positional arguments but 3 were given" in error_str:
            build_partial = -0.04
        elif "has no attribute '_input_spec'" in error_lower:
            build_partial = -0.12
        elif "has no attribute '_output_dim'" in error_lower or "has no attribute '_input_dim'" in error_lower:
            build_partial = -0.09
        elif "has no attribute 'infer_dimensions'" in error_lower:
            build_partial = -0.08
        elif "nameerror" in error_lower and any(
            token in error_str for token in ("dropout_prob", "in_channels", "features", "out_channels")
        ):
            build_partial = -0.06
        elif "keyerror" in error_lower and "out_channels" in error_lower:
            build_partial = -0.06
        elif "runtimeerror" in error_lower and "expected input" in error_lower and "to have" in error_lower:
            build_partial = -0.05
        else:
            build_partial = -0.10

        if bool(raw_extraction.get("dual_backbone_ok")):
            build_partial += 0.02
        if bool(raw_extraction.get("xml_tag_exact")):
            build_partial += 0.01
        if bool(raw_extraction.get("exact_init_signature")):
            build_partial += 0.02
        if bool(raw_extraction.get("exact_forward_signature")):
            build_partial += 0.01
        if bool(code_trace.get("assigns_input_spec")):
            build_partial += 0.03
        elif bool(code_trace.get("references_input_spec")):
            build_partial -= 0.02

        return _clip(build_partial, -0.12, 0.12)

    if 'SyntaxError' in error_str:
        build_partial = -0.3
    elif 'NameError' in error_str or 'ImportError' in error_str:
        build_partial = -0.2
    elif 'TypeError' in error_str:
        build_partial = -0.1
    elif 'RuntimeError' in error_str and 'shape' in error_str.lower():
        build_partial = 0.05
    elif error_str:
        build_partial = -0.15
    return build_partial


def _compute_warmup_dense_reward(test_acc: Optional[float]) -> Optional[float]:
    if test_acc is None:
        return None
    return max(0.05, min(0.30, 0.08 + 0.55 * float(test_acc)))


def _is_minimal_backbone_classifier_template(init_code: str) -> bool:
    significant_lines = []
    for raw_line in textwrap.dedent(init_code or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(
            (
                "def __init__",
                "super().__init__",
                "self.device",
                "self.use_amp",
                "self._input_spec",
                "self.pattern",
                "self.infer_dimensions",
            )
        ):
            continue
        significant_lines.append(line)
    assignment_lines = [line for line in significant_lines if line.startswith("self.")]
    if len(assignment_lines) > 3:
        return False
    has_backbone_a = any("self.backbone_a" in line for line in assignment_lines)
    has_backbone_b = any("self.backbone_b" in line for line in assignment_lines)
    has_classifier = any("self.classifier" in line for line in assignment_lines)
    if not (has_backbone_a and has_backbone_b and has_classifier):
        return False
    non_core_assignments = [
        line
        for line in assignment_lines
        if all(token not in line for token in ("self.backbone_a", "self.backbone_b", "self.classifier"))
    ]
    return not non_core_assignments


def _stage1_validity_scale(res: Dict[str, Any]) -> float:
    if bool(res.get("loss_drop_ok")):
        return 1.0
    if bool(res.get("backward_ok")):
        return 0.85
    if bool(res.get("forward_shape_ok")):
        return 0.55
    return 0.0


def _stage1_validity_reward(res: Dict[str, Any], graph_info) -> float:
    if not graph_info or not graph_info.parse_ok:
        return -0.25
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0) or 0.0)
        return min(-0.25, -0.50 + build_partial)
    if not res.get("forward_ok"):
        return -0.18
    if not res.get("forward_shape_ok"):
        return -0.08
    if not res.get("backward_ok"):
        return -0.02
    if not res.get("loss_drop_ok"):
        loss_drop = _optional_float(res.get("loss_drop"))
        if loss_drop is None:
            return 0.02
        return _clip(0.01 + 0.25 * float(loss_drop), -0.01, 0.05)
    return STAGE1_EXECUTABLE_BONUS


def _template_penalty(
    *,
    stage_name: str,
    shallow_one_shot: bool,
    minimal_init_template: bool,
) -> float:
    penalty = 0.0
    if shallow_one_shot:
        penalty += -0.08 if stage_name == STAGE1_STRUCTURE_EXPLORE else -0.05
    if minimal_init_template:
        penalty += -0.10 if stage_name == STAGE1_STRUCTURE_EXPLORE else -0.08
    return penalty


def _history_context_reward(
    *,
    stage_name: str,
    training_context: Dict[str, Any],
    executable_candidate: bool,
    formal_success_candidate: bool,
    discovery_candidate: bool,
    novel_vs_trainset_family: bool,
    novel_vs_trainset_graph: bool,
    dominant_family_repeat: bool,
    dominant_descriptor_repeat: bool,
    shallow_one_shot: bool,
    plain_parallel_repeat: bool,
    minimal_init_template: bool,
    batch_same_descriptor_count: int,
    validity_scale: float = 1.0,
) -> float:
    return 0.0


def _goal_tag_match_stats(graph_info, prompt_goal_tags: Optional[List[str]]) -> Tuple[int, int, float]:
    tags = list(prompt_goal_tags or [])
    if not tags:
        return 0, 0, 0.0
    hit_count = sum(1 for tag in tags if prompt_goal_satisfied(graph_info, tag))
    total_count = len(tags)
    hit_rate = float(hit_count) / float(total_count) if total_count > 0 else 0.0
    return hit_count, total_count, hit_rate


def _discovery_failure_result(
    reward: float,
    error: str,
    *,
    seed_accuracy_baseline: float,
    backbone_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "reward": reward,
        "built_ok": False,
        "forward_ok": False,
        "forward_shape_ok": False,
        "trained_step_ok": False,
        "backward_ok": False,
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
        "test_acc": None,
        "train_acc": None,
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
        "group_baseline_reward_target_acc": None,
        "group_reward_target_gain": None,
        "group_reward_target_improved": False,
        "reward_batch_index": None,
        "reward_group_id": None,
        "group_warmup": False,
        "val_metric": None,
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": list(backbone_model_names or []),
        "reward_target_metric": _stage_reward_target_metric(current_stage_name),
        "reward_target_value": None,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_reward_target_for_goal": None,
        "r_dense": 0.0,
        "r_prev_group": 0.0,
        "r_best_group": 0.0,
        "r_goal_best": 0.0,
        "r_goal_match": 0.0,
        "r_trainset_novelty": 0.0,
        "r_generalization": 0.0,
        "r_structure_group": 0.0,
        "r_structure_archive": 0.0,
        "r_descriptor_diversity": 0.0,
        "r_batch_elite": 0.0,
        "r_repeat_family": 0.0,
        "r_plain_fuse_penalty": 0.0,
        "r_template_penalty": 0.0,
        "r_history_context": 0.0,
        "r_no_progress_penalty": 0.0,
        "batch_elite_rank": None,
        "batch_elite_tier": "none",
        "batch_elite_threshold_passed": False,
        "goal_tag_hit_count": 0,
        "goal_tag_total_count": 0,
        "goal_tag_hit_rate": 0.0,
        "prev_target_reward_target_acc": None,
        "best_target_reward_target_acc": None,
        "open_discovery": {
            "r_primary": 0.0,
            "r_tiebreak": 0.0,
            "r_trainset_novelty": 0.0,
            "r_dense": 0.0,
            "r_prev_group": 0.0,
            "r_best_group": 0.0,
            "r_goal_best": 0.0,
            "r_goal_match": 0.0,
            "r_generalization": 0.0,
            "r_structure_group": 0.0,
            "r_structure_archive": 0.0,
            "r_descriptor_diversity": 0.0,
            "r_batch_elite": 0.0,
            "r_repeat_family": 0.0,
            "r_plain_fuse_penalty": 0.0,
            "r_template_penalty": 0.0,
            "r_history_context": 0.0,
            "r_no_progress_penalty": 0.0,
            "batch_elite_rank": None,
            "batch_elite_tier": "none",
            "batch_elite_threshold_passed": False,
            "novel_vs_trainset_family": False,
            "novel_vs_trainset_graph": False,
            "archive_snapshot_family_freq": 0,
            "batch_same_family_count": 0,
            "reward_target_metric": _stage_reward_target_metric(current_stage_name),
            "reward_target_value": None,
            "goal_tag_hit_count": 0,
            "goal_tag_total_count": 0,
            "goal_tag_hit_rate": 0.0,
        },
        "error": error,
        "current_stage_name": current_stage_name,
        "current_stage_index": _current_stage_index(),
        "stage_uses_formal_eval": _stage_uses_formal_eval(current_stage_name),
        "stage_uses_static_only": _stage_uses_static_only(current_stage_name),
    }


def _is_trainable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and res.get("backward_ok")
        and res.get("loss_drop_ok")
    )


def _has_completed_formal_epoch(res: Dict[str, Any]) -> bool:
    try:
        return int(res.get("epochs_completed", 0) or 0) >= 1
    except (TypeError, ValueError):
        return False


def _is_executable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
    )


def _stage2_target_qualified_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qualified_rows: List[Dict[str, Any]] = []
    for item in rows:
        if not bool(item.get("executable_candidate")):
            continue
        if not _has_completed_formal_epoch(item):
            continue
        reward_target_value = _optional_float(item.get("reward_target_value"))
        if reward_target_value is None or float(reward_target_value) < STAGE2_GATE_MIN_REWARD_TARGET:
            continue
        qualified_rows.append(item)
    return qualified_rows


def _apply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.30)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        return min(reward_value, -0.70 + build_partial)
    if not res.get("forward_ok"):
        return min(reward_value, -0.30)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.20)
    if not res.get("backward_ok"):
        loss_drop = _optional_float(res.get("loss_drop"))
        partial_progress = _clip(0.25 * float(loss_drop or 0.0), -0.04, 0.04)
        return min(reward_value, -0.12 + partial_progress)
    if not res.get("loss_drop_ok"):
        loss_drop = _optional_float(res.get("loss_drop"))
        if loss_drop is None:
            return min(reward_value, -0.02)
        return min(reward_value, _clip(-0.02 + 0.20 * float(loss_drop), -0.08, 0.04))
    return reward_value


def _apply_executability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.35)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        return min(reward_value, -0.70 + build_partial)
    if not res.get("forward_ok"):
        return min(reward_value, -0.28)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.16)
    return reward_value


def _stage_reward_profile(stage_name: str) -> Dict[str, float]:
    if stage_name == STAGE2_FORMAL_EXPLORE:
        return {
            "dense_scale": STAGE2_DENSE_SCALE,
            "prev_group_scale": STAGE2_PREV_GROUP_SCALE,
            "best_group_scale": STAGE2_BEST_GROUP_SCALE,
            "global_baseline_blend": STAGE2_GLOBAL_BASELINE_BLEND,
            "backbone_prev_group_scale": STAGE2_BACKBONE_PREV_GROUP_SCALE,
            "backbone_best_group_scale": STAGE2_BACKBONE_BEST_GROUP_SCALE,
            "goal_best_scale": STAGE2_GOAL_BEST_SCALE,
            "goal_match_scale": STAGE2_GOAL_MATCH_SCALE,
            "structure_scale": STAGE2_STRUCTURE_SCALE,
            "repeat_family_scale": STAGE2_REPEAT_FAMILY_SCALE,
            "plain_fuse_scale": STAGE2_PLAIN_FUSE_SCALE,
            "no_progress_scale": STAGE2_NO_PROGRESS_SCALE,
            "non_improving_cap": STAGE2_NON_IMPROVING_CAP,
            "descriptor_non_improving_cap": STAGE2_DESCRIPTOR_NON_IMPROVING_CAP,
        }
    return {
        "dense_scale": STAGE3_DENSE_SCALE,
        "prev_group_scale": STAGE3_PREV_GROUP_SCALE,
        "best_group_scale": STAGE3_BEST_GROUP_SCALE,
        "global_baseline_blend": STAGE3_GLOBAL_BASELINE_BLEND,
        "backbone_prev_group_scale": STAGE3_BACKBONE_PREV_GROUP_SCALE,
        "backbone_best_group_scale": STAGE3_BACKBONE_BEST_GROUP_SCALE,
        "goal_best_scale": STAGE3_GOAL_BEST_SCALE,
        "goal_match_scale": STAGE3_GOAL_MATCH_SCALE,
        "structure_scale": STAGE3_STRUCTURE_SCALE,
        "repeat_family_scale": STAGE3_REPEAT_FAMILY_SCALE,
        "plain_fuse_scale": STAGE3_PLAIN_FUSE_SCALE,
        "no_progress_scale": STAGE3_NO_PROGRESS_SCALE,
        "non_improving_cap": STAGE3_NON_IMPROVING_CAP,
        "descriptor_non_improving_cap": STAGE3_DESCRIPTOR_NON_IMPROVING_CAP,
    }


def _archive_rarity_bonus_stage1(archive_snapshot_family_freq: int) -> float:
    if archive_snapshot_family_freq <= 0:
        return STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS
    if archive_snapshot_family_freq == 1:
        return STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS
    if archive_snapshot_family_freq <= 3:
        return STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS
    return 0.0


def _archive_rarity_bonus_formal(archive_snapshot_family_freq: int) -> float:
    return min(
        STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP,
        STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP / math.sqrt(float(archive_snapshot_family_freq) + 1.0),
    )


def _structure_progress_components(
    graph_info,
    *,
    batch_same_family_count: int,
    archive_snapshot_family_freq: int,
    novel_vs_trainset_family: bool,
    novel_vs_trainset_graph: bool,
    shallow_one_shot: bool,
    use_formal_archive_bonus: bool = False,
) -> Tuple[float, float]:
    if not graph_info or not graph_info.parse_ok:
        return 0.0, 0.0

    r_structure_group = 0.0
    if passes_macro_structure_gate(graph_info):
        r_structure_group += STRUCTURE_MACRO_BONUS
    if is_multi_stage_architecture(graph_info):
        r_structure_group += STRUCTURE_MULTI_STAGE_BONUS
    if has_structural_motif(graph_info):
        r_structure_group += STRUCTURE_MOTIF_BONUS
    if batch_same_family_count <= 1:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS
    elif batch_same_family_count == 2:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS * 0.5
    if (
        dominant_family_hash
        and graph_info.family_hash != dominant_family_hash
        and float(dominant_family_share or 0.0) >= 0.20
    ):
        r_structure_group += STRUCTURE_NON_DOMINANT_FAMILY_BONUS
    if shallow_one_shot:
        r_structure_group = max(0.0, r_structure_group - 0.02)

    r_structure_archive = 0.0
    if novel_vs_trainset_family:
        r_structure_archive += TRAINSET_NOVEL_FAMILY_BONUS
    elif novel_vs_trainset_graph:
        r_structure_archive += TRAINSET_NOVEL_GRAPH_BONUS
    archive_bonus = _archive_rarity_bonus_formal if use_formal_archive_bonus else _archive_rarity_bonus_stage1
    r_structure_archive += archive_bonus(archive_snapshot_family_freq)

    return _clip(r_structure_group, 0.0, 0.14), _clip(r_structure_archive, 0.0, 0.08)


def _recompute_discovery_reward(
    res: Dict[str, Any],
    graph_info,
) -> Tuple[float, float, float]:
    stage_name = str(res.get("current_stage_name") or current_stage_name)
    r_primary = (
        float(res.get("r_dense", 0.0) or 0.0)
        + float(res.get("r_prev_group", 0.0) or 0.0)
        + float(res.get("r_best_group", 0.0) or 0.0)
        + float(res.get("r_prev_backbone_group", 0.0) or 0.0)
        + float(res.get("r_best_backbone_group", 0.0) or 0.0)
        + float(res.get("r_goal_best", 0.0) or 0.0)
        + float(res.get("r_generalization", 0.0) or 0.0)
        + float(res.get("r_structure_group", 0.0) or 0.0)
        + float(res.get("r_structure_archive", 0.0) or 0.0)
        + float(res.get("r_descriptor_diversity", 0.0) or 0.0)
        + float(res.get("r_cnn_diversity", 0.0) or 0.0)
        + float(res.get("r_batch_elite", 0.0) or 0.0)
        + float(res.get("r_repeat_family", 0.0) or 0.0)
        + float(res.get("r_plain_fuse_penalty", 0.0) or 0.0)
        + float(res.get("r_template_penalty", 0.0) or 0.0)
        + float(res.get("r_history_context", 0.0) or 0.0)
        + float(res.get("r_no_progress_penalty", 0.0) or 0.0)
    )
    r_tiebreak = float(res.get("r_goal_match", 0.0) or 0.0)
    total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
    if stage_name in {STAGE1_STRUCTURE_EXPLORE, STAGE2_FORMAL_EXPLORE}:
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)
    else:
        total_reward = _apply_trainability_clamp(res, total_reward, graph_info)
    return total_reward, r_primary, r_tiebreak


def build_stage_eval_cfg(
    *,
    stage_name: Optional[str] = None,
    in_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    out_shape: Tuple[int, ...] = (10,),
    prm: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    cfg: Optional[EvalConfig] = None,
) -> EvalConfig:
    del cfg
    requested_stage = str(stage_name or current_stage_name)
    resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if requested_stage == STAGE1_STRUCTURE_EXPLORE:
        eval_limit_seconds = env_int("NNGPT_RL_STAGE1_EVAL_LIMIT_SECONDS", 120)
        formal_epoch_limit_minutes = None
    else:
        eval_limit_seconds = env_int("NNGPT_RL_FORMAL_EVAL_LIMIT_SECONDS", 1800)
        configured_epoch_limit = env_float("NNGPT_RL_FORMAL_EPOCH_LIMIT_MINUTES", 0.0)
        formal_epoch_limit_minutes = configured_epoch_limit if configured_epoch_limit > 0.0 else None
    return EvalConfig(
        device=resolved_device,
        input_shape=tuple(in_shape),
        n_classes=int(out_shape[0]),
        train_epochs=int((prm or {}).get("epoch", 1) or 1),
        default_batch_size=int((prm or {}).get("batch", 32) or 32),
        eval_limit_seconds=eval_limit_seconds,
        reward_target_metric=_stage_reward_target_metric(requested_stage),
        formal_nn_eval=_stage_uses_formal_eval(requested_stage),
        static_only=_stage_uses_static_only(requested_stage),
        formal_task=os.getenv("NNGPT_RL_FORMAL_TASK", "img-classification"),
        formal_dataset=os.getenv("NNGPT_RL_FORMAL_DATASET", "cifar-10"),
        formal_metric=os.getenv("NNGPT_RL_FORMAL_METRIC", "acc"),
        formal_epoch_limit_minutes=formal_epoch_limit_minutes,
    )


def _invoke_eval_cfg_builder(eval_cfg_builder, **kwargs) -> EvalConfig:
    if not callable(eval_cfg_builder):
        raise TypeError("eval_cfg_builder must be callable")

    try:
        signature = inspect.signature(eval_cfg_builder)
    except (TypeError, ValueError):
        return eval_cfg_builder(**kwargs)

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return eval_cfg_builder(**kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return eval_cfg_builder(**supported_kwargs)


def _attach_group_context(
    res: Dict[str, Any],
    *,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    return RewardPayload.attach_group_context(
        sys.modules[__name__],
        res,
        seed_accuracy_baseline=seed_accuracy_baseline,
        group_context=group_context,
    )

def base_discovery_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Optional[Dict[str, Any]] = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_descriptor_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_cnn_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_cnn_pair_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
    group_baseline_reward_target_acc: Optional[float] = None,
    reward_batch_index: Optional[int] = None,
    reward_group_id: Optional[int] = None,
    group_warmup: bool = False,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    stage_name = str(current_stage_name)
    stage_profile = _stage_reward_profile(stage_name)
    stage_reward_metric = _stage_reward_target_metric(stage_name)
    prm = {
        'lr': 0.01,
        'batch': 64,
        'dropout': 0.3,
        'momentum': 0.9,
        'transform': FORMAL_REWARD_TRANSFORM,
        'epoch': 1,
    }
    block_code, init_code, forward_code = extract_completion_blocks(completion)
    backbone_model_names = _extract_backbone_model_names(init_code)
    if not block_code or not init_code or not forward_code:
        return _discovery_failure_result(
            -2.0,
            "Reconstruction failed (tags missing?)",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if "self.pattern" in forward_code:
        return _discovery_failure_result(
            -5.0,
            "CHEAT DETECTED: Accessed self.pattern inside forward block",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    graph_info = graph_info or extract_graph_info(
        init_code,
        forward_code,
        legacy_patterns=SFTUtil.legacy_patterns,
    )
    effective_pattern_name = (
        graph_info.pattern_name if graph_info.has_custom_pattern_name else graph_info.suggested_pattern_name
    )
    pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""

    final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
    if not final_code:
        return _discovery_failure_result(
            -2.0,
            "Code reconstruction failed",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if precomputed_eval_result is not None:
        res = dict(precomputed_eval_result)
    else:
        res = evaluate_code_and_reward(
            final_code,
            in_shape=(1, 3, 224, 224),
            out_shape=(10,),
            prm=prm,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=build_stage_eval_cfg(
                stage_name=stage_name,
                in_shape=(1, 3, 224, 224),
                out_shape=(10,),
                prm=prm,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )

    if not res.get("built_ok"):
        res["r_build_partial"] = _compute_build_partial_reward(res)
    res.setdefault("backbone_model_names", backbone_model_names)
    backbone_signature = build_backbone_signature(backbone_model_names)
    cnn_signature = str(getattr(graph_info, "cnn_signature", "") or "incomplete_cnn")
    cnn_expr = str(getattr(graph_info, "cnn_expr", "") or "IncompleteCNN")
    backbone_cnn_pair_key = _backbone_cnn_pair_key(backbone_signature, cnn_signature)

    training_context = summarize_stage_training_context(stage_name)
    shallow_one_shot = is_shallow_one_shot_fuse(graph_info)
    minimal_init_template = _is_minimal_backbone_classifier_template(init_code)
    batch_same_family_count = batch_family_hashes.count(graph_info.family_hash) if batch_family_hashes and graph_info.parse_ok else 0
    batch_same_graph_count = batch_graph_hashes.count(graph_info.graph_hash) if batch_graph_hashes and graph_info.parse_ok else 0
    batch_same_descriptor_count = (
        batch_descriptor_keys.count(graph_info.descriptor_key)
        if batch_descriptor_keys and graph_info.parse_ok
        else 0
    )
    batch_same_backbone_count = (
        batch_backbone_signatures.count(backbone_signature)
        if batch_backbone_signatures and graph_info.parse_ok
        else 0
    )
    batch_same_backbone_cnn_count = (
        sum(
            1
            for batch_backbone_signature, batch_cnn_signature in zip(batch_backbone_signatures or [], batch_cnn_signatures or [])
            if batch_backbone_signature == backbone_signature and batch_cnn_signature == cnn_signature
        )
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_family_freq = int((archive_snapshot_family_counts or {}).get(graph_info.family_hash, 0)) if graph_info.parse_ok else 0
    archive_snapshot_descriptor_freq = (
        int((archive_snapshot_descriptor_counts or {}).get(graph_info.descriptor_key, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_backbone_freq = (
        int((archive_snapshot_backbone_signature_counts or {}).get(backbone_signature, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_cnn_freq = (
        int((archive_snapshot_cnn_signature_counts or {}).get(cnn_signature, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_backbone_cnn_freq = (
        int((archive_snapshot_backbone_cnn_pair_counts or {}).get(backbone_cnn_pair_key, 0))
        if graph_info.parse_ok
        else 0
    )
    global_group_baseline_reward_target_acc = (
        group_baseline_reward_target_acc
        if group_baseline_reward_target_acc is not None
        else prev_closed_group_mean_reward_target_acc
    )
    use_backbone_baseline = bool(
        graph_info.parse_ok
        and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
        and backbone_signature in prev_closed_group_mean_reward_target_by_backbone
    )
    backbone_group_baseline_reward_target_acc = (
        prev_closed_group_mean_reward_target_by_backbone.get(backbone_signature)
        if use_backbone_baseline
        else None
    )
    best_backbone_group_mean_reward_target_acc = (
        best_closed_group_mean_reward_target_by_backbone.get(backbone_signature)
        if graph_info.parse_ok and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
        else None
    )
    effective_group_baseline_reward_target_acc = (
        backbone_group_baseline_reward_target_acc
        if backbone_group_baseline_reward_target_acc is not None
        else global_group_baseline_reward_target_acc
    )
    dominant_backbone_cnn_signature, dominant_backbone_cnn_share = StageState.dominant_counter_entry(
        {
            str(key): int(value)
            for key, value in (archive_snapshot_backbone_cnn_pair_counts or {}).items()
            if str(key).startswith(f"{backbone_signature}::")
        }
    )
    dominant_backbone_cnn_signature = (
        dominant_backbone_cnn_signature.split("::", 1)[1]
        if dominant_backbone_cnn_signature and "::" in dominant_backbone_cnn_signature
        else dominant_backbone_cnn_signature
    )

    novel_vs_trainset_family = False
    novel_vs_trainset_graph = False
    frozen_train_acc = _optional_float(res.get("frozen_train_acc", res.get("train_acc")))
    frozen_test_acc = _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric"))))
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    train_acc = frozen_train_acc
    test_acc = frozen_test_acc
    reward_target_value = _result_reward_target_value(res)
    if reward_target_value is None and stage_name != STAGE1_STRUCTURE_EXPLORE:
        reward_target_value = frozen_test_acc
    goal_key = primary_goal_key(prompt_goal_tags or [])
    best_reward_target_for_goal = best_reward_target_by_goal.get(goal_key)
    group_train_acc_gain = None
    group_train_acc_improved = False
    group_reward_target_gain = None
    group_reward_target_improved = False
    r_primary = 0.0
    r_tiebreak = 0.0
    r_dense = 0.0
    r_prev_group = 0.0
    r_best_group = 0.0
    r_prev_backbone_group = 0.0
    r_best_backbone_group = 0.0
    r_goal_best = 0.0
    r_goal_match = 0.0
    r_trainset_novelty = 0.0
    r_generalization = 0.0
    r_structure_group = 0.0
    r_structure_archive = 0.0
    r_batch_elite = 0.0
    r_repeat_family = 0.0
    r_plain_fuse_penalty = 0.0
    r_template_penalty = 0.0
    r_history_context = 0.0
    r_no_progress_penalty = 0.0
    r_descriptor_diversity = 0.0
    r_cnn_diversity = 0.0
    r_formal_success_signal = 0.0
    stage1_validity_scale = 0.0
    dominant_family_repeat = False
    dominant_descriptor_repeat = False
    dominant_cnn_repeat = False
    plain_parallel_repeat = False
    descriptor_reward_cap_applied = False
    cnn_reward_cap_applied = False
    executable_candidate = _is_executable_candidate(res, graph_info)
    formal_success_candidate = _is_trainable_candidate(res, graph_info)
    has_formal_epoch = _has_completed_formal_epoch(res)
    discovery_candidate = False
    goal_tag_hit_count, goal_tag_total_count, goal_tag_hit_rate = _goal_tag_match_stats(graph_info, prompt_goal_tags)
    prev_target_train_acc = None
    best_target_train_acc = None
    prev_target_reward_target_acc = None
    best_target_reward_target_acc = None
    backbone_prev_target_reward_target_acc = None
    backbone_best_target_reward_target_acc = None
    beat_prev_target = False
    beat_best_target = False
    beat_prev_backbone_target = False
    beat_best_backbone_target = False
    quality_diversity_eligible = False
    formal_progress_refresh = False
    backbone_reward_target_gain = None
    backbone_reward_target_improved = False

    if executable_candidate:
        novel_vs_trainset_family = graph_info.family_hash not in train_family_hashes
        novel_vs_trainset_graph = graph_info.graph_hash not in train_graph_hashes
        discovery_candidate = bool(graph_info.parse_ok and archive_snapshot_family_freq <= 0)
        if novel_vs_trainset_family:
            r_trainset_novelty = TRAINSET_NOVEL_FAMILY_BONUS
        elif novel_vs_trainset_graph:
            r_trainset_novelty = TRAINSET_NOVEL_GRAPH_BONUS
        r_structure_group, r_structure_archive = _structure_progress_components(
            graph_info,
            batch_same_family_count=batch_same_family_count,
            archive_snapshot_family_freq=archive_snapshot_family_freq,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            shallow_one_shot=shallow_one_shot,
            use_formal_archive_bonus=stage_name != STAGE1_STRUCTURE_EXPLORE,
        )
        if (
            (not group_warmup)
            and dominant_family_hash
            and graph_info.parse_ok
            and graph_info.family_hash == dominant_family_hash
            and not discovery_candidate
        ):
            dominant_family_repeat = True
            r_repeat_family = REPEAT_FAMILY_PENALTY
        if graph_info.is_plain_parallel_triple:
            plain_parallel_repeat = True
            if stage_name == STAGE1_STRUCTURE_EXPLORE:
                if goal_tag_hit_rate < STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                    r_plain_fuse_penalty = min(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_PENALTY)
                else:
                    r_plain_fuse_penalty = min(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_WARMUP_PENALTY)
            elif (not group_warmup) and not discovery_candidate:
                r_plain_fuse_penalty = PLAIN_FUSE_PENALTY

    if stage_name == STAGE1_STRUCTURE_EXPLORE:
        reward_target_value = None
        stage1_validity_scale = _stage1_validity_scale(res)
        r_dense = _stage1_validity_reward(res, graph_info)
        r_template_penalty = _template_penalty(
            stage_name=stage_name,
            shallow_one_shot=shallow_one_shot,
            minimal_init_template=minimal_init_template,
        )
        if executable_candidate:
            novelty_scale = max(0.35, float(stage1_validity_scale))
            goal_alignment_scale = float(goal_tag_hit_rate or 0.0)
            r_structure_group *= (
                STAGE1_STRUCTURE_GROUP_SCALE
                * float(stage1_validity_scale)
            )
            r_structure_archive *= (
                STAGE1_STRUCTURE_ARCHIVE_SCALE
                * float(stage1_validity_scale)
            )
            if batch_same_descriptor_count == 1:
                r_structure_group += (
                    STAGE1_DESCRIPTOR_BATCH_UNIQUE_BONUS
                    * novelty_scale
                )
                if batch_same_graph_count == 1:
                    r_structure_group += STAGE1_GRAPH_BATCH_UNIQUE_BONUS * novelty_scale
            elif batch_same_descriptor_count > 2:
                descriptor_batch_repeat_penalty = max(
                    STAGE1_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY,
                    STAGE1_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY * float(batch_same_descriptor_count - 2),
                )
                r_no_progress_penalty += descriptor_batch_repeat_penalty
                if batch_same_graph_count > 2:
                    graph_batch_repeat_penalty = max(
                        STAGE1_GRAPH_BATCH_REPEAT_MAX_PENALTY,
                        STAGE1_GRAPH_BATCH_REPEAT_STEP_PENALTY * float(batch_same_graph_count - 2),
                    )
                    r_no_progress_penalty += graph_batch_repeat_penalty
            if archive_snapshot_descriptor_freq <= 0:
                r_structure_archive += STAGE1_DESCRIPTOR_ARCHIVE_NOVEL_BONUS * novelty_scale
            elif archive_snapshot_descriptor_freq > 3:
                descriptor_archive_repeat_penalty = max(
                    STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_descriptor_freq - 3),
                )
                r_no_progress_penalty += descriptor_archive_repeat_penalty
            if discovery_candidate and goal_alignment_scale >= STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                r_goal_best = (
                    STAGE1_DISCOVERY_FAMILY_BONUS
                    * novelty_scale
                    * max(0.35, goal_alignment_scale)
                )
            elif novel_vs_trainset_graph and goal_alignment_scale >= STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                r_goal_best = (
                    STAGE1_DISCOVERY_GRAPH_BONUS
                    * novelty_scale
                    * max(0.35, goal_alignment_scale)
                )
            else:
                r_no_progress_penalty += STAGE1_NON_DISCOVERY_EXECUTABLE_PENALTY
            if goal_tag_total_count > 0:
                if goal_alignment_scale <= 0.0:
                    r_no_progress_penalty += STAGE1_ZERO_GOAL_HIT_PENALTY
                elif goal_alignment_scale < 0.5:
                    r_no_progress_penalty += STAGE1_LOW_GOAL_HIT_PENALTY
            if archive_snapshot_family_freq > 0:
                archive_repeat_penalty = max(
                    STAGE1_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE1_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_family_freq),
                )
                r_no_progress_penalty += archive_repeat_penalty
            if batch_same_family_count >= 3:
                batch_repeat_penalty = max(
                    STAGE1_BATCH_REPEAT_MAX_PENALTY,
                    STAGE1_BATCH_REPEAT_STEP_PENALTY * float(batch_same_family_count - 2),
                )
                r_no_progress_penalty += batch_repeat_penalty
            r_goal_match = STAGE1_GOAL_MATCH_SCALE * goal_tag_hit_rate * novelty_scale
            r_repeat_family = _clip(r_repeat_family, STAGE1_DOMINANT_FAMILY_PENALTY, 0.0)
            r_plain_fuse_penalty = _clip(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_PENALTY, 0.0)
            reward_target_value = _clip(
                STAGE1_STATIC_BASE_SCORE
                + max(0.0, r_dense)
                + max(0.0, r_goal_best)
                + max(0.0, r_structure_group)
                + max(0.0, r_structure_archive),
                0.0,
                1.0,
            )
            if goal_tag_total_count > 0:
                if goal_alignment_scale <= 0.0:
                    reward_target_value = min(float(reward_target_value), STAGE1_ZERO_GOAL_HIT_REWARD_CAP)
                elif goal_alignment_scale < 0.5:
                    reward_target_value = min(float(reward_target_value), STAGE1_LOW_GOAL_HIT_REWARD_CAP)
            if graph_info.is_plain_parallel_triple:
                reward_target_value = min(float(reward_target_value), STAGE1_PLAIN_PARALLEL_REWARD_CAP)
                if goal_alignment_scale < STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                    reward_target_value = min(float(reward_target_value), STAGE1_OFF_TARGET_PLAIN_PARALLEL_REWARD_CAP)
            shallow_pattern_repeat = bool(
                (not discovery_candidate)
                and shallow_one_shot
                and (archive_snapshot_family_freq > 0 or batch_same_family_count >= 3)
            )
            if (
                (not discovery_candidate and archive_snapshot_family_freq > 5)
                or shallow_pattern_repeat
                or dominant_family_repeat
                or minimal_init_template
            ):
                reward_target_value = min(float(reward_target_value), 0.26)
                if minimal_init_template:
                    reward_target_value = min(float(reward_target_value), 0.18)
        r_history_context = _history_context_reward(
            stage_name=stage_name,
            training_context=training_context,
            executable_candidate=executable_candidate,
            formal_success_candidate=formal_success_candidate,
            discovery_candidate=discovery_candidate,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            dominant_family_repeat=dominant_family_repeat,
            dominant_descriptor_repeat=False,
            shallow_one_shot=shallow_one_shot,
            plain_parallel_repeat=plain_parallel_repeat,
            minimal_init_template=minimal_init_template,
            batch_same_descriptor_count=batch_same_descriptor_count,
            validity_scale=stage1_validity_scale,
        )
        if (reward_target_value is not None) and (effective_group_baseline_reward_target_acc is not None) and (not group_warmup):
            group_reward_target_gain = float(reward_target_value - effective_group_baseline_reward_target_acc)
            group_reward_target_improved = bool(group_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)
        r_primary = (
            r_dense
            + r_goal_best
            + r_structure_group
            + r_structure_archive
            + r_repeat_family
            + r_plain_fuse_penalty
            + r_template_penalty
            + r_history_context
            + r_no_progress_penalty
        )
        r_tiebreak = r_goal_match
        total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)
    else:
        if (train_acc is not None) and (group_baseline_train_acc is not None) and (not group_warmup):
            group_train_acc_gain = float(train_acc - group_baseline_train_acc)
            group_train_acc_improved = bool(group_train_acc_gain >= GROUP_IMPROVEMENT_DELTA)
        if (reward_target_value is not None) and (effective_group_baseline_reward_target_acc is not None) and (not group_warmup):
            group_reward_target_gain = float(reward_target_value - effective_group_baseline_reward_target_acc)
            group_reward_target_improved = bool(group_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)
        if (reward_target_value is not None) and (backbone_group_baseline_reward_target_acc is not None) and (not group_warmup):
            backbone_reward_target_gain = float(reward_target_value - backbone_group_baseline_reward_target_acc)
            backbone_reward_target_improved = bool(backbone_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)

        if has_formal_epoch and reward_target_value is not None:
            train_acc_value = float(train_acc or 0.0)
            reward_target_float = float(reward_target_value)
            quality_diversity_eligible = bool(formal_success_candidate)
            r_dense = stage_profile["dense_scale"] * _clip(
                0.03 + 0.20 * reward_target_float + 0.04 * max(0.0, train_acc_value - 0.50),
                0.02,
                0.22,
            )
            if formal_success_candidate:
                r_formal_success_signal = FORMAL_SUCCESS_SIGNAL_BONUS
            if (not group_warmup) and (group_baseline_train_acc is not None):
                prev_target_train_acc = float(group_baseline_train_acc) + GROUP_IMPROVEMENT_DELTA
            if (not group_warmup) and (best_closed_group_mean_train_acc is not None):
                best_target_train_acc = float(best_closed_group_mean_train_acc) + BEST_GROUP_REFRESH_DELTA
            if (not group_warmup) and (global_group_baseline_reward_target_acc is not None):
                prev_target_reward_target_acc = float(global_group_baseline_reward_target_acc) + GROUP_IMPROVEMENT_DELTA
                beat_prev_target = reward_target_float >= prev_target_reward_target_acc
                global_prev_group_reward = stage_profile["prev_group_scale"] * _clip(
                    10.0 * (reward_target_float - prev_target_reward_target_acc),
                    -1.8,
                    1.8,
                )
                r_prev_group = global_prev_group_reward
                if backbone_group_baseline_reward_target_acc is not None:
                    r_prev_group *= float(stage_profile["global_baseline_blend"])
                    backbone_prev_target_reward_target_acc = (
                        float(backbone_group_baseline_reward_target_acc) + GROUP_IMPROVEMENT_DELTA
                    )
                    beat_prev_backbone_target = reward_target_float >= backbone_prev_target_reward_target_acc
                    r_prev_backbone_group = stage_profile["backbone_prev_group_scale"] * _clip(
                        10.0 * (reward_target_float - backbone_prev_target_reward_target_acc),
                        -1.8,
                        1.8,
                    )
            if (not group_warmup) and (best_closed_group_mean_reward_target_acc is not None):
                best_target_reward_target_acc = float(best_closed_group_mean_reward_target_acc) + BEST_GROUP_REFRESH_DELTA
                beat_best_target = reward_target_float >= best_target_reward_target_acc
                global_best_group_reward = stage_profile["best_group_scale"] * _clip(
                    12.0 * (reward_target_float - best_target_reward_target_acc),
                    -1.2,
                    1.2,
                )
                r_best_group = global_best_group_reward
                if best_backbone_group_mean_reward_target_acc is not None:
                    r_best_group *= float(stage_profile["global_baseline_blend"])
                    backbone_best_target_reward_target_acc = (
                        float(best_backbone_group_mean_reward_target_acc) + BEST_GROUP_REFRESH_DELTA
                    )
                    beat_best_backbone_target = reward_target_float >= backbone_best_target_reward_target_acc
                    r_best_backbone_group = stage_profile["backbone_best_group_scale"] * _clip(
                        12.0 * (reward_target_float - backbone_best_target_reward_target_acc),
                        -1.2,
                        1.2,
                    )
            if (
                (not group_warmup)
                and (best_reward_target_for_goal is not None)
                and reward_target_float >= float(best_reward_target_for_goal) + GOAL_REFRESH_DELTA
            ):
                r_goal_best = stage_profile["goal_best_scale"] * GOAL_REFRESH_BONUS
            effective_prev_target_reward_target_acc = (
                backbone_prev_target_reward_target_acc
                if backbone_prev_target_reward_target_acc is not None
                else prev_target_reward_target_acc
            )
            effective_beat_prev_target = (
                beat_prev_backbone_target
                if backbone_prev_target_reward_target_acc is not None
                else beat_prev_target
            )
            if (
                (not group_warmup)
                and effective_prev_target_reward_target_acc is not None
                and not effective_beat_prev_target
            ):
                r_no_progress_penalty = stage_profile["no_progress_scale"] * NO_PROGRESS_PENALTY
            if (frozen_train_acc is not None) and (frozen_test_acc is not None):
                overfit_gap = max(0.0, float(frozen_train_acc) - float(frozen_test_acc) - GENERALIZATION_GAP_TOLERANCE)
                r_generalization = _clip(
                    -GENERALIZATION_PENALTY_SCALE * overfit_gap,
                    GENERALIZATION_PENALTY_CAP,
                    0.0,
                )

        if backbone_prev_target_reward_target_acc is not None or backbone_best_target_reward_target_acc is not None:
            formal_progress_refresh = bool(
                beat_prev_backbone_target
                or beat_best_backbone_target
                or r_goal_best > 0.0
            )
        else:
            formal_progress_refresh = bool(
                beat_prev_target
                or beat_best_target
                or r_goal_best > 0.0
            )
        descriptor_progress_refresh = formal_progress_refresh
        if executable_candidate and graph_info.parse_ok and graph_info.descriptor_key:
            if quality_diversity_eligible and batch_same_descriptor_count == 1:
                r_descriptor_diversity += STAGE23_DESCRIPTOR_BATCH_UNIQUE_BONUS
            elif batch_same_descriptor_count > 1:
                r_descriptor_diversity += max(
                    STAGE23_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY,
                    STAGE23_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY * float(batch_same_descriptor_count - 1),
                )

            if quality_diversity_eligible and archive_snapshot_descriptor_freq <= 0:
                r_descriptor_diversity += STAGE23_DESCRIPTOR_ARCHIVE_NOVEL_BONUS
            elif archive_snapshot_descriptor_freq > 1:
                r_descriptor_diversity += max(
                    STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_descriptor_freq - 1),
                )

            if (
                (not group_warmup)
                and quality_diversity_eligible
                and dominant_descriptor_key
                and graph_info.descriptor_key != dominant_descriptor_key
                and float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE
            ):
                r_descriptor_diversity += STAGE23_NON_DOMINANT_DESCRIPTOR_BONUS
            elif (
                (not group_warmup)
                and dominant_descriptor_key
                and graph_info.descriptor_key == dominant_descriptor_key
                and float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE
                and not descriptor_progress_refresh
            ):
                dominant_descriptor_repeat = True
                if float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_STRONG_SHARE:
                    r_descriptor_diversity += STAGE23_DOMINANT_DESCRIPTOR_REPEAT_STRONG_PENALTY
                else:
                    r_descriptor_diversity += STAGE23_DOMINANT_DESCRIPTOR_REPEAT_PENALTY

        if executable_candidate and graph_info.parse_ok and cnn_signature:
            if quality_diversity_eligible and batch_same_backbone_cnn_count == 1:
                r_cnn_diversity += STAGE23_CNN_BATCH_UNIQUE_BONUS
            elif batch_same_backbone_cnn_count > 1:
                r_cnn_diversity += max(
                    STAGE23_CNN_BATCH_REPEAT_MAX_PENALTY,
                    STAGE23_CNN_BATCH_REPEAT_STEP_PENALTY * float(batch_same_backbone_cnn_count - 1),
                )

            if quality_diversity_eligible and archive_snapshot_backbone_cnn_freq <= 0:
                r_cnn_diversity += STAGE23_CNN_ARCHIVE_NOVEL_BONUS
            elif archive_snapshot_backbone_cnn_freq > 1:
                r_cnn_diversity += max(
                    STAGE23_CNN_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE23_CNN_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_backbone_cnn_freq - 1),
                )

            if (
                (not group_warmup)
                and quality_diversity_eligible
                and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
                and dominant_backbone_cnn_signature
                and cnn_signature != dominant_backbone_cnn_signature
                and float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_SOFT_SHARE
            ):
                r_cnn_diversity += STAGE23_NON_DOMINANT_CNN_BONUS
            elif (
                (not group_warmup)
                and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
                and dominant_backbone_cnn_signature
                and cnn_signature == dominant_backbone_cnn_signature
                and float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_SOFT_SHARE
                and not descriptor_progress_refresh
            ):
                dominant_cnn_repeat = True
                if float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_STRONG_SHARE:
                    r_cnn_diversity += STAGE23_DOMINANT_CNN_REPEAT_STRONG_PENALTY
                else:
                    r_cnn_diversity += STAGE23_DOMINANT_CNN_REPEAT_PENALTY

        r_goal_match = stage_profile["goal_match_scale"] * GOAL_MATCH_REWARD_SCALE * goal_tag_hit_rate
        r_template_penalty = _template_penalty(
            stage_name=stage_name,
            shallow_one_shot=shallow_one_shot,
            minimal_init_template=minimal_init_template,
        )
        r_history_context = _history_context_reward(
            stage_name=stage_name,
            training_context=training_context,
            executable_candidate=executable_candidate,
            formal_success_candidate=formal_success_candidate,
            discovery_candidate=discovery_candidate,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            dominant_family_repeat=dominant_family_repeat,
            dominant_descriptor_repeat=dominant_descriptor_repeat,
            shallow_one_shot=shallow_one_shot,
            plain_parallel_repeat=plain_parallel_repeat,
            minimal_init_template=minimal_init_template,
            batch_same_descriptor_count=batch_same_descriptor_count,
        )
        r_structure_group *= stage_profile["structure_scale"]
        r_structure_archive *= stage_profile["structure_scale"]
        r_repeat_family *= stage_profile["repeat_family_scale"]
        r_plain_fuse_penalty *= stage_profile["plain_fuse_scale"]

        r_primary = (
            r_dense
            + r_formal_success_signal
            + r_prev_group
            + r_best_group
            + r_prev_backbone_group
            + r_best_backbone_group
            + r_goal_best
            + r_generalization
            + r_structure_group
            + r_structure_archive
            + r_descriptor_diversity
            + r_cnn_diversity
            + r_batch_elite
            + r_repeat_family
            + r_plain_fuse_penalty
            + r_template_penalty
            + r_history_context
            + r_no_progress_penalty
        )
        r_tiebreak = r_goal_match
        total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
        effective_prev_target_reward_target_acc = (
            backbone_prev_target_reward_target_acc
            if backbone_prev_target_reward_target_acc is not None
            else prev_target_reward_target_acc
        )
        effective_beat_prev_target = (
            beat_prev_backbone_target
            if backbone_prev_target_reward_target_acc is not None
            else beat_prev_target
        )
        if has_formal_epoch and effective_prev_target_reward_target_acc is not None and not effective_beat_prev_target:
            total_reward = min(total_reward, stage_profile["non_improving_cap"])
        if has_formal_epoch and dominant_descriptor_repeat:
            total_reward = min(total_reward, stage_profile["descriptor_non_improving_cap"])
            descriptor_reward_cap_applied = True
        if has_formal_epoch and dominant_cnn_repeat:
            total_reward = min(total_reward, stage_profile["descriptor_non_improving_cap"])
            cnn_reward_cap_applied = True
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)

    reward_target_value_for_payload = reward_target_value
    reward_metric_for_payload = stage_reward_metric

    warmup_dense_reward = None
    if stage_name != STAGE1_STRUCTURE_EXPLORE and group_warmup and has_formal_epoch:
        warmup_dense_reward = _compute_warmup_dense_reward(reward_target_value)
        total_reward = float(warmup_dense_reward or 0.0)
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)

    res['reward'] = total_reward
    res['test_acc'] = test_acc
    res['train_acc'] = train_acc
    res['frozen_train_acc'] = frozen_train_acc
    res['frozen_test_acc'] = frozen_test_acc
    res['unfrozen_train_acc'] = unfrozen_train_acc
    res['unfrozen_test_acc'] = unfrozen_test_acc
    res['val_metric'] = frozen_test_acc
    res['seed_accuracy_baseline'] = seed_accuracy_baseline
    res['group_baseline_train_acc'] = group_baseline_train_acc
    res['group_train_acc_gain'] = group_train_acc_gain
    res['group_train_acc_improved'] = group_train_acc_improved
    res['reward_target_metric'] = reward_metric_for_payload
    res['reward_target_value'] = reward_target_value_for_payload
    res['global_group_baseline_reward_target_acc'] = global_group_baseline_reward_target_acc
    res['group_baseline_reward_target_acc'] = effective_group_baseline_reward_target_acc
    res['group_backbone_baseline_reward_target_acc'] = backbone_group_baseline_reward_target_acc
    res['group_reward_target_gain'] = group_reward_target_gain
    res['group_reward_target_improved'] = group_reward_target_improved
    res['backbone_reward_target_gain'] = backbone_reward_target_gain
    res['backbone_reward_target_improved'] = backbone_reward_target_improved
    res['reward_batch_index'] = reward_batch_index
    res['reward_group_id'] = reward_group_id
    res['group_warmup'] = group_warmup
    res['warmup_dense_reward'] = warmup_dense_reward
    res['current_stage_name'] = stage_name
    res['current_stage_index'] = RL_STAGE_TO_INDEX.get(stage_name, 0)
    res['stage_uses_formal_eval'] = _stage_uses_formal_eval(stage_name)
    res['stage_uses_static_only'] = _stage_uses_static_only(stage_name)
    res['best_closed_group_mean_reward_target_acc'] = best_closed_group_mean_reward_target_acc
    res['best_closed_group_mean_train_acc'] = best_closed_group_mean_train_acc
    res['best_closed_group_mean_test_acc'] = best_closed_group_mean_test_acc
    res['best_backbone_group_mean_reward_target_acc'] = best_backbone_group_mean_reward_target_acc
    res['best_reward_target_for_goal'] = best_reward_target_for_goal
    res['r_dense'] = r_dense
    res['r_prev_group'] = r_prev_group
    res['r_best_group'] = r_best_group
    res['r_prev_backbone_group'] = r_prev_backbone_group
    res['r_best_backbone_group'] = r_best_backbone_group
    res['r_goal_best'] = r_goal_best
    res['r_goal_match'] = r_goal_match
    res['r_trainset_novelty'] = r_trainset_novelty
    res['r_generalization'] = r_generalization
    res['r_structure_group'] = r_structure_group
    res['r_structure_archive'] = r_structure_archive
    res['r_descriptor_diversity'] = r_descriptor_diversity
    res['r_cnn_diversity'] = r_cnn_diversity
    res['r_formal_success_signal'] = r_formal_success_signal
    res['r_batch_elite'] = r_batch_elite
    res['r_repeat_family'] = r_repeat_family
    res['r_plain_fuse_penalty'] = r_plain_fuse_penalty
    res['r_template_penalty'] = r_template_penalty
    res['r_history_context'] = r_history_context
    res['r_no_progress_penalty'] = r_no_progress_penalty
    res['batch_elite_rank'] = None
    res['batch_elite_tier'] = "none"
    res['batch_elite_threshold_passed'] = False
    res['goal_tag_hit_count'] = goal_tag_hit_count
    res['goal_tag_total_count'] = goal_tag_total_count
    res['goal_tag_hit_rate'] = goal_tag_hit_rate
    res['prev_target_reward_target_acc'] = prev_target_reward_target_acc
    res['best_target_reward_target_acc'] = best_target_reward_target_acc
    res['backbone_prev_target_reward_target_acc'] = backbone_prev_target_reward_target_acc
    res['backbone_best_target_reward_target_acc'] = backbone_best_target_reward_target_acc
    res['prev_target_train_acc'] = prev_target_train_acc
    res['best_target_train_acc'] = best_target_train_acc
    res['executable_candidate'] = executable_candidate
    res['discovery_candidate'] = discovery_candidate
    res['formal_success_candidate'] = formal_success_candidate
    res['signature'] = f"{normalize_pattern_name(effective_pattern_name)}_{graph_info.graph_hash[:6]}"
    res['graph_hash'] = graph_info.graph_hash
    res['family_id'] = graph_info.family_id
    res['family_expr'] = graph_info.family_expr
    res['family_hash'] = graph_info.family_hash
    res['backbone_signature'] = backbone_signature
    res['cnn_signature'] = cnn_signature
    res['cnn_expr'] = cnn_expr
    res['archive_snapshot_backbone_freq'] = archive_snapshot_backbone_freq
    res['archive_snapshot_cnn_freq'] = archive_snapshot_cnn_freq
    res['archive_snapshot_backbone_cnn_freq'] = archive_snapshot_backbone_cnn_freq
    res['batch_same_backbone_count'] = batch_same_backbone_count
    res['batch_same_backbone_cnn_count'] = batch_same_backbone_cnn_count
    res['descriptor_key'] = graph_info.descriptor_key
    res['dominant_descriptor_key'] = dominant_descriptor_key
    res['dominant_descriptor_share'] = dominant_descriptor_share
    res['dominant_backbone_cnn_signature'] = dominant_backbone_cnn_signature
    res['dominant_backbone_cnn_share'] = dominant_backbone_cnn_share
    res['unique_descriptor_count'] = len(descriptor_archive_counts)
    res['dominant_descriptor_repeat'] = dominant_descriptor_repeat
    res['dominant_cnn_repeat'] = dominant_cnn_repeat
    res['descriptor_reward_cap_applied'] = descriptor_reward_cap_applied
    res['cnn_reward_cap_applied'] = cnn_reward_cap_applied
    res['history_exploration_pressure'] = float(training_context.get('exploration_pressure') or 0.0)
    res['minimal_init_template'] = minimal_init_template
    res['graph_expr'] = graph_info.graph_expr
    res['pattern_name'] = effective_pattern_name
    res['suggested_pattern_name'] = graph_info.suggested_pattern_name
    res['open_discovery'] = {
        'r_primary': r_primary,
        'r_tiebreak': r_tiebreak,
        'r_trainset_novelty': r_trainset_novelty,
        'r_dense': r_dense,
        'r_formal_success_signal': r_formal_success_signal,
        'r_prev_group': r_prev_group,
        'r_best_group': r_best_group,
        'r_prev_backbone_group': r_prev_backbone_group,
        'r_best_backbone_group': r_best_backbone_group,
        'r_goal_best': r_goal_best,
        'r_goal_match': r_goal_match,
        'r_generalization': r_generalization,
        'r_structure_group': r_structure_group,
        'r_structure_archive': r_structure_archive,
        'r_descriptor_diversity': r_descriptor_diversity,
        'r_cnn_diversity': r_cnn_diversity,
        'r_batch_elite': r_batch_elite,
        'r_repeat_family': r_repeat_family,
        'r_plain_fuse_penalty': r_plain_fuse_penalty,
        'r_template_penalty': r_template_penalty,
        'r_history_context': r_history_context,
        'r_no_progress_penalty': r_no_progress_penalty,
        'batch_elite_rank': None,
        'batch_elite_tier': "none",
        'batch_elite_threshold_passed': False,
        'group_baseline_train_acc': group_baseline_train_acc,
        'global_group_baseline_reward_target_acc': global_group_baseline_reward_target_acc,
        'group_baseline_reward_target_acc': effective_group_baseline_reward_target_acc,
        'group_backbone_baseline_reward_target_acc': backbone_group_baseline_reward_target_acc,
        'reward_target_metric': reward_metric_for_payload,
        'reward_target_value': reward_target_value_for_payload,
        'best_closed_group_mean_reward_target_acc': best_closed_group_mean_reward_target_acc,
        'best_closed_group_mean_train_acc': best_closed_group_mean_train_acc,
        'best_closed_group_mean_test_acc': best_closed_group_mean_test_acc,
        'best_backbone_group_mean_reward_target_acc': best_backbone_group_mean_reward_target_acc,
        'best_reward_target_for_goal': best_reward_target_for_goal,
        'goal_tag_hit_count': goal_tag_hit_count,
        'goal_tag_total_count': goal_tag_total_count,
        'goal_tag_hit_rate': goal_tag_hit_rate,
        'prev_target_reward_target_acc': prev_target_reward_target_acc,
        'best_target_reward_target_acc': best_target_reward_target_acc,
        'backbone_prev_target_reward_target_acc': backbone_prev_target_reward_target_acc,
        'backbone_best_target_reward_target_acc': backbone_best_target_reward_target_acc,
        'prev_target_train_acc': prev_target_train_acc,
        'best_target_train_acc': best_target_train_acc,
        'group_train_acc_gain': group_train_acc_gain,
        'group_train_acc_improved': group_train_acc_improved,
        'group_reward_target_gain': group_reward_target_gain,
        'group_reward_target_improved': group_reward_target_improved,
        'backbone_reward_target_gain': backbone_reward_target_gain,
        'backbone_reward_target_improved': backbone_reward_target_improved,
        'reward_batch_index': reward_batch_index,
        'reward_group_id': reward_group_id,
        'group_warmup': group_warmup,
        'prompt_goal_tags': list(prompt_goal_tags or []),
        'batch_same_graph_count': batch_same_graph_count,
        'batch_same_family_count': batch_same_family_count,
        'batch_same_descriptor_count': batch_same_descriptor_count,
        'batch_same_backbone_count': batch_same_backbone_count,
        'batch_same_backbone_cnn_count': batch_same_backbone_cnn_count,
        'archive_snapshot_family_freq': archive_snapshot_family_freq,
        'archive_snapshot_descriptor_freq': archive_snapshot_descriptor_freq,
        'archive_snapshot_backbone_freq': archive_snapshot_backbone_freq,
        'archive_snapshot_cnn_freq': archive_snapshot_cnn_freq,
        'archive_snapshot_backbone_cnn_freq': archive_snapshot_backbone_cnn_freq,
        'macro_structure_ok': passes_macro_structure_gate(graph_info),
        'is_multi_stage_architecture': is_multi_stage_architecture(graph_info),
        'is_shallow_one_shot_fuse': shallow_one_shot,
        'family_id': graph_info.family_id,
        'family_hash': graph_info.family_hash,
        'backbone_signature': backbone_signature,
        'cnn_signature': cnn_signature,
        'cnn_expr': cnn_expr,
        'descriptor_key': graph_info.descriptor_key,
        'dominant_descriptor_key': dominant_descriptor_key,
        'dominant_descriptor_share': dominant_descriptor_share,
        'dominant_backbone_cnn_signature': dominant_backbone_cnn_signature,
        'dominant_backbone_cnn_share': dominant_backbone_cnn_share,
        'unique_descriptor_count': len(descriptor_archive_counts),
        'dominant_descriptor_repeat': dominant_descriptor_repeat,
        'dominant_cnn_repeat': dominant_cnn_repeat,
        'descriptor_reward_cap_applied': descriptor_reward_cap_applied,
        'cnn_reward_cap_applied': cnn_reward_cap_applied,
        'history_exploration_pressure': float(training_context.get('exploration_pressure') or 0.0),
        'minimal_init_template': minimal_init_template,
        'depth': graph_info.depth,
        'merges': graph_info.merges,
        'max_fan_in': graph_info.max_fan_in,
        'backbone_calls': graph_info.backbone_calls,
        'fractal_calls': graph_info.fractal_calls,
        'stem_calls': graph_info.stem_calls,
        'project_calls': graph_info.project_calls,
        'fuse_calls': graph_info.fuse_calls,
        'is_plain_parallel_triple': graph_info.is_plain_parallel_triple,
        'is_legacy_pattern_name': graph_info.is_legacy_pattern_name,
        'parse_ok': graph_info.parse_ok,
        'novel_vs_trainset_family': novel_vs_trainset_family,
        'novel_vs_trainset_graph': novel_vs_trainset_graph,
        'archive_snapshot_family_freq': archive_snapshot_family_freq,
        'batch_same_family_count': batch_same_family_count,
        'stage_name': stage_name,
        'stage_index': RL_STAGE_TO_INDEX.get(stage_name, 0),
        'stage_uses_formal_eval': _stage_uses_formal_eval(stage_name),
        'stage_uses_static_only': _stage_uses_static_only(stage_name),
        'executable_candidate': executable_candidate,
        'discovery_candidate': discovery_candidate,
        'formal_success_candidate': formal_success_candidate,
    }
    return res


def reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Optional[Dict[str, Any]] = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_descriptor_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_cnn_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_cnn_pair_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
    group_baseline_reward_target_acc: Optional[float] = None,
    reward_batch_index: Optional[int] = None,
    reward_group_id: Optional[int] = None,
    group_warmup: bool = False,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    """Reward open-ended motif discovery while keeping the existing XML output ABI."""
    return base_discovery_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )


def _apply_batch_elite_bonuses(scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
    if group_context["group_warmup"] or str(group_context.get("current_stage_name")) == STAGE1_STRUCTURE_EXPLORE:
        return

    eligible: List[Tuple[float, Dict[str, Any]]] = []

    for item in scored_results:
        res = item["result"]
        graph_info = item["graph_info"]
        reward_target_value = _result_reward_target_value(res)
        if not _is_executable_candidate(res, graph_info):
            continue
        if not _has_completed_formal_epoch(res):
            continue
        if reward_target_value is None:
            continue
        eligible.append((float(reward_target_value), item))

    eligible.sort(key=lambda pair: pair[0], reverse=True)
    elite_summaries: List[str] = []
    max_elites = min(len(BATCH_ELITE_SOFT_BONUSES), len(BATCH_ELITE_IMPROVING_BONUSES))
    for rank, (reward_target_float, item) in enumerate(eligible[:max_elites]):
        threshold_baseline = _optional_float(
            item["result"].get("group_backbone_baseline_reward_target_acc", item["result"].get("group_baseline_reward_target_acc"))
        )
        threshold = (
            float(threshold_baseline) + GROUP_IMPROVEMENT_DELTA
            if threshold_baseline is not None
            else None
        )
        threshold_passed = threshold is not None and reward_target_float >= threshold
        tier = "improving" if threshold_passed else "soft"
        bonus = (
            BATCH_ELITE_IMPROVING_BONUSES[rank]
            if threshold_passed
            else BATCH_ELITE_SOFT_BONUSES[rank]
        )
        res = item["result"]
        graph_info = item["graph_info"]
        if float(res.get("r_no_progress_penalty", 0.0) or 0.0) < 0.0:
            res["r_no_progress_penalty"] = 0.0
        res["r_batch_elite"] = bonus
        res["batch_elite_rank"] = rank + 1
        res["batch_elite_tier"] = tier
        res["batch_elite_threshold_passed"] = threshold_passed
        total_reward, r_primary, r_tiebreak = _recompute_discovery_reward(res, graph_info)
        res["reward"] = total_reward
        open_discovery = res.setdefault("open_discovery", {})
        open_discovery["r_batch_elite"] = bonus
        open_discovery["r_primary"] = r_primary
        open_discovery["r_tiebreak"] = r_tiebreak
        open_discovery["batch_elite_rank"] = rank + 1
        open_discovery["batch_elite_tier"] = tier
        open_discovery["batch_elite_threshold_passed"] = threshold_passed
        item["score"] = float(total_reward)
        elite_summaries.append(
            f"#{rank + 1} target={reward_target_float:.4f} tier={tier} bonus={bonus:.3f} "
            f"struct={float(res.get('r_structure_group', 0.0) or 0.0) + float(res.get('r_structure_archive', 0.0) or 0.0):.3f}"
        )
    if elite_summaries:
        code_logger.log_to_file(
            f"[Reward Batch Elite] reward_batch_index={group_context['reward_batch_index']} "
            + "; ".join(elite_summaries)
        )

def _reward_failure_result(
    *,
    error: str,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    return _attach_group_context(
        {
            "reward": -1.0,
            "built_ok": False,
            "forward_ok": False,
            "forward_shape_ok": False,
            "trained_step_ok": False,
            "backward_ok": False,
            "loss_start": None,
            "loss_end": None,
            "loss_drop": None,
            "loss_drop_ok": False,
            "train_acc": None,
            "val_metric": None,
            "latency_ms": None,
            "params_m": None,
            "error": error,
        },
        seed_accuracy_baseline=seed_accuracy_baseline,
        group_context=group_context,
    )


def _prepare_local_reward_entries(
    prompts,
    completions,
    *,
    seed_accuracy_baselines: List[float],
    group_context: Dict[str, Any],
    precompute_eval: bool = True,
) -> List[Dict[str, Any]]:
    runtime_rank = _distributed_rank()
    batch_graph_infos: List[Any] = []
    batch_backbone_model_names: List[List[str]] = []
    batch_prompt_goal_tags = [extract_prompt_goal_tags(prompt) for prompt in prompts]

    for i, completion in enumerate(completions):
        _, init_code, forward_code = extract_completion_blocks(completion)
        backbone_model_names = _extract_backbone_model_names(init_code)
        if init_code and forward_code:
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
        else:
            graph_info = None
        batch_graph_infos.append(graph_info)
        batch_backbone_model_names.append(backbone_model_names)

    local_entries: List[Dict[str, Any]] = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        graph_info = batch_graph_infos[i]
        local_entries.append(
            {
                "rank": runtime_rank,
                "local_index": i,
                "prompt": prompt,
                "completion": completion,
                "graph_info": graph_info,
                "backbone_model_names": batch_backbone_model_names[i],
                "backbone_signature": build_backbone_signature(batch_backbone_model_names[i]),
                "cnn_signature": (
                    str(getattr(graph_info, "cnn_signature", "") or "")
                    if graph_info is not None
                    else "incomplete_cnn"
                ),
                "prompt_goal_tags": batch_prompt_goal_tags[i],
                "goal_key": primary_goal_key(batch_prompt_goal_tags[i]),
                "seed_accuracy_baseline": seed_accuracy_baselines[i],
                "precomputed_eval_result": None,
            }
        )
    if precompute_eval:
        _precompute_eval_results(local_entries, group_context=group_context)
    return local_entries


def _build_global_reward_entries(gathered_entries: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    global_entries: List[Dict[str, Any]] = []
    for global_index, entry in enumerate(
        entry
        for rank_entries in gathered_entries
        for entry in list(rank_entries or [])
    ):
        merged_entry = dict(entry)
        merged_entry["global_index"] = global_index
        global_entries.append(merged_entry)
    return global_entries


def _select_global_reward_entries_for_rank(
    entries: List[Dict[str, Any]],
    *,
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    total_entries = len(entries)
    start = (total_entries * int(rank)) // max(1, int(world_size))
    end = (total_entries * (int(rank) + 1)) // max(1, int(world_size))
    return [dict(entry) for entry in entries[start:end]]


def _merge_gathered_reward_entries(
    gathered_entries: List[List[Dict[str, Any]]],
    *,
    expected_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    merged_entries = [
        dict(entry)
        for rank_entries in gathered_entries
        for entry in list(rank_entries or [])
    ]
    merged_entries.sort(key=lambda entry: int(entry.get("global_index", -1)))
    if expected_count is not None and len(merged_entries) != int(expected_count):
        raise RuntimeError(
            f"Distributed reward merge expected {expected_count} entries, but received {len(merged_entries)}"
        )
    return merged_entries


def _build_batched_eval_specs(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_cfg_builder = getattr(evaluate_code_and_reward, "_nngpt_eval_cfg_builder", build_stage_eval_cfg)
    batched_eval_entries: List[Dict[str, Any]] = []
    batched_eval_specs: List[Dict[str, Any]] = []

    for entry in entries:
        if entry.get("precomputed_eval_result") is not None:
            continue

        completion = str(entry.get("completion", ""))
        graph_info = entry.get("graph_info")
        block_code, init_code, forward_code = extract_completion_blocks(completion)
        if not block_code or not init_code or not forward_code:
            continue
        if "self.pattern" in forward_code or graph_info is None:
            continue

        pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""
        final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
        if not final_code:
            continue

        spec = {
            "code": final_code,
            "in_shape": (1, 3, 224, 224),
            "out_shape": (10,),
            "prm": {
                "lr": 0.01,
                "batch": 64,
                "dropout": 0.3,
                "momentum": 0.9,
                "transform": FORMAL_REWARD_TRANSFORM,
                "epoch": 1,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed_accuracy_baseline": entry["seed_accuracy_baseline"],
            "reward_batch_index": group_context["reward_batch_index"],
            "completion_index": int(entry.get("global_index", entry["local_index"])),
            "batch_last_item": False,
        }
        if callable(eval_cfg_builder):
            spec["cfg"] = _invoke_eval_cfg_builder(
                eval_cfg_builder,
                stage_name=str(group_context.get("current_stage_name") or current_stage_name),
                in_shape=(1, 3, 224, 224),
                out_shape=(10,),
                prm=spec["prm"],
                cfg=None,
                device=spec["device"],
            )

        batched_eval_entries.append(entry)
        batched_eval_specs.append(spec)

    if batched_eval_specs:
        batched_eval_specs[-1]["batch_last_item"] = True

    return batched_eval_entries, batched_eval_specs


def _precompute_eval_results(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
) -> None:
    batched_eval_entries, batched_eval_specs = _build_batched_eval_specs(
        entries,
        group_context=group_context,
    )
    if not batched_eval_specs:
        return
    rank = _distributed_rank()
    local_rank = env_int("LOCAL_RANK", 0)
    started_at = time.time()
    print(
        "[Reward Precompute Local] start "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"reward_batch_index={group_context.get('reward_batch_index')} "
        f"entries={len(batched_eval_specs)} "
        f"wall_time={started_at:.6f}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    batched_eval_results = evaluate_code_and_reward_batch(batched_eval_specs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ended_at = time.time()
    elapsed_seconds = max(0.0, ended_at - started_at)
    print(
        "[Reward Precompute Local] end "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"reward_batch_index={group_context.get('reward_batch_index')} "
        f"entries={len(batched_eval_specs)} "
        f"elapsed_seconds={elapsed_seconds:.2f} "
        f"wall_time={ended_at:.6f}"
    )
    for entry, eval_result in zip(batched_eval_entries, batched_eval_results):
        entry["precomputed_eval_result"] = eval_result


def _format_reward_trace_context(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict) or not context:
        return ""
    preferred_keys = (
        "freeze_backbones",
        "formal_eval_backend",
        "formal_eval_duration_seconds",
        "trainer_device",
        "trainer_in_shape",
        "dataset_out_shape",
        "forward_output_shape",
        "params_m",
        "batch",
        "epoch",
        "epoch_limit_minutes",
        "transform",
        "num_workers",
        "estimated_training_time_minutes",
        "reported_accuracy",
        "reported_duration_seconds",
    )
    parts = []
    for key in preferred_keys:
        if key in context and context[key] is not None:
            parts.append(f"{key}={context[key]!r}")
    code_trace = context.get("code_trace")
    if isinstance(code_trace, dict):
        for key in ("references_input_spec", "assigns_input_spec", "references_pattern_attr", "line_count"):
            if key in code_trace and code_trace[key] is not None:
                parts.append(f"code_trace.{key}={code_trace[key]!r}")
    return ", ".join(parts)


def _log_reward_failure_trace(entry: Dict[str, Any], res: Dict[str, Any]) -> None:
    graph_info = entry.get("graph_info")
    pattern_name = getattr(graph_info, "suggested_pattern_name", None) if graph_info is not None else None
    branches = [("root", res)]
    frozen_eval = res.get("frozen_eval")
    unfrozen_eval = res.get("unfrozen_eval")
    if isinstance(frozen_eval, dict):
        branches.append(("frozen", frozen_eval))
    if isinstance(unfrozen_eval, dict):
        branches.append(("unfrozen", unfrozen_eval))

    seen = set()
    for branch_name, payload in branches:
        error = payload.get("error")
        if not error:
            continue
        stage = payload.get("error_stage")
        hint = payload.get("error_hint")
        context = payload.get("error_context")
        dedupe_key = (branch_name, str(error), str(stage), str(hint), repr(context))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        trace_message = (
            f"[Reward Failure Trace] rank={entry['rank']} "
            f"batch_index={entry['local_index']} "
            f"branch={branch_name} "
            f"pattern={pattern_name!r} "
            f"stage={stage or 'unknown'} "
            f"error={error!r}"
        )
        if hint:
            trace_message += f" hint={hint!r}"
        formatted_context = _format_reward_trace_context(context)
        if formatted_context:
            trace_message += f" context=({formatted_context})"
        code_logger.log_to_file(trace_message)


def _score_reward_entries(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
    archive_snapshot_family_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    archive_snapshot_descriptor_counts = dict(descriptor_archive_counts)
    archive_snapshot_backbone_signature_counts = dict(backbone_signature_archive_counts)
    archive_snapshot_cnn_signature_counts = dict(cnn_signature_archive_counts)
    archive_snapshot_backbone_cnn_pair_counts = dict(backbone_cnn_pair_archive_counts)
    batch_graph_hashes = [
        entry["graph_info"].graph_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_family_hashes = [
        entry["graph_info"].family_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_descriptor_keys = [
        entry["graph_info"].descriptor_key if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_backbone_signatures = [
        _entry_backbone_signature(entry)
        for entry in entries
    ]
    batch_cnn_signatures = [
        _entry_cnn_signature(entry)
        for entry in entries
    ]
    scored_results: List[Dict[str, Any]] = []

    for position, entry in enumerate(entries):
        index = int(entry["local_index"])
        completion_index = int(entry.get("global_index", index))
        code_logger.log_to_file("=" * 50)
        try:
            res = reward_fn(
                entry["completion"],
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                precomputed_eval_result=entry.get("precomputed_eval_result"),
                graph_info=entry.get("graph_info"),
                batch_graph_hashes=batch_graph_hashes,
                batch_family_hashes=batch_family_hashes,
                batch_descriptor_keys=batch_descriptor_keys,
                batch_backbone_signatures=batch_backbone_signatures,
                batch_cnn_signatures=batch_cnn_signatures,
                prompt_goal_tags=entry.get("prompt_goal_tags"),
                archive_snapshot_family_counts=archive_snapshot_family_counts,
                archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
                archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
                archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
                archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
                group_baseline_train_acc=group_context["group_baseline_train_acc"],
                group_baseline_reward_target_acc=group_context["group_baseline_reward_target_acc"],
                reward_batch_index=group_context["reward_batch_index"],
                reward_group_id=group_context["reward_group_id"],
                group_warmup=group_context["group_warmup"],
                completion_index=completion_index,
                batch_last_item=position == (len(entries) - 1),
            )
            res = _attach_group_context(
                res,
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            dispatch_parts = []
            if res.get("worker_slot") is not None:
                dispatch_parts.append(f"worker_slot={res.get('worker_slot')}")
            if res.get("assigned_gpu") is not None:
                dispatch_parts.append(f"assigned_gpu={res.get('assigned_gpu')}")
            if res.get("worker_device") is not None:
                dispatch_parts.append(f"worker_device={res.get('worker_device')}")
            if dispatch_parts:
                code_logger.log_to_file(
                    f"[Reward Dispatch] rank={entry['rank']} batch_index={index}, " + ", ".join(dispatch_parts)
                )
            _log_reward_failure_trace(entry, res)
            score = float(res.get("reward", -2.0))
        except PersistentEvalWorkerError:
            raise
        except Exception as exc:
            code_logger.log_to_file(f"Reward calculation failed at rank={entry['rank']} index={index}: {exc}")
            res = _reward_failure_result(
                error=str(exc),
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            score = -1.0
        scored_results.append(
            {
                **entry,
                "result": res,
                "score": score,
            }
        )

    _apply_batch_elite_bonuses(scored_results, group_context)
    for item in scored_results:
        item["score"] = float(item["result"].get("reward", item.get("score", -1.0)))
    return scored_results


def _finalize_scored_results(scored_results: List[Dict[str, Any]]) -> None:
    global B_index

    current_batch_results: List[Dict[str, Any]] = []
    stage_name = str(current_stage_name)
    for item in scored_results:
        index = int(item["local_index"])
        prompt = item["prompt"]
        completion = item["completion"]
        graph_info = item["graph_info"]
        goal_key = item["goal_key"]
        res = item["result"]
        score = float(item["score"])
        sig = res.get("signature", "unknown")
        backbone_signature = _result_backbone_signature(res)
        cnn_signature = _result_cnn_signature(res, graph_info)
        backbone_cnn_pair_key = _backbone_cnn_pair_key(backbone_signature, cnn_signature)

        is_executable = _is_executable_candidate(res, graph_info)
        is_trainable = _is_trainable_candidate(res, graph_info)
        reward_target_value = _result_reward_target_value(res)
        if reward_target_value is not None:
            current_batch_results.append(res)
        if is_executable:
            if stage_name != STAGE1_STRUCTURE_EXPLORE or bool(res.get("discovery_candidate")):
                _record_current_group_trainable_sample(goal_key, res, graph_info)
            graph_archive_counts[graph_info.graph_hash] += 1
            family_archive_counts[graph_info.family_id] += 1
            family_hash_archive_counts[graph_info.family_hash] += 1
            descriptor_archive_key = str(res.get("descriptor_key") or getattr(graph_info, "descriptor_key", "") or "")
            if descriptor_archive_key:
                descriptor_archive_counts[descriptor_archive_key] += 1
            if backbone_signature:
                backbone_signature_archive_counts[backbone_signature] += 1
            if cnn_signature:
                cnn_signature_archive_counts[cnn_signature] += 1
            if backbone_signature and cnn_signature:
                backbone_cnn_pair_archive_counts[backbone_cnn_pair_key] += 1
            motif_name_counts[res.get("pattern_name", graph_info.suggested_pattern_name)] += 1
            get_goal_counter(goal_graph_archive_counts, goal_key)[graph_info.graph_hash] += 1
            get_goal_counter(goal_family_hash_archive_counts, goal_key)[graph_info.family_hash] += 1
            current_best = family_metric_best.get(graph_info.family_hash, float("-inf"))
            gain_value = res.get("group_reward_target_gain")
            family_metric_best[graph_info.family_hash] = max(
                current_best,
                float(gain_value if gain_value is not None else float("-inf")),
            )
            if bool(res.get("discovery_candidate")):
                discovery_family_hashes_seen.add(str(graph_info.family_hash))

        code_logger.log_to_file(
            f"Rank {item['rank']} batch index {index}, Motif: {res.get('pattern_name')}, Signature: {sig}, Result: {res}"
        )

        should_save = (
            bool(graph_info)
            and graph_info.parse_ok
            and res.get("built_ok")
            and res.get("forward_shape_ok")
            and res.get("backward_ok")
            and _has_completed_formal_epoch(res)
        )
        save_gate_reason = "ok"
        if should_save and backbone_signature and cnn_signature:
            saved_best = saved_best_reward_target_by_backbone_cnn.get(backbone_cnn_pair_key)
            if (
                saved_backbone_cnn_pair_counts.get(backbone_cnn_pair_key, 0) > 0
                and (
                    reward_target_value is None
                    or (
                        saved_best is not None
                        and float(reward_target_value) < float(saved_best) + SAVE_DUPLICATE_BACKBONE_CNN_DELTA
                    )
                )
            ):
                should_save = False
                save_gate_reason = "duplicate_backbone_cnn_signature"

        if should_save:
            pattern_override = "" if graph_info.has_custom_pattern_name else res.get("suggested_pattern_name", "")
            block_code, init_code, forward_code = extract_completion_blocks(completion)
            if pattern_override:
                init_code = ensure_pattern_name(init_code, pattern_override)
            final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
            normalized_completion = render_completion_xml(block_code, init_code, forward_code)
            out_path = run_epoch_dir(0)
            model_dir = synth_dir(out_path) / f"B{B_index}"
            model_dir.mkdir(exist_ok=True, parents=True)

            code_file = model_dir / new_nn_file
            with open(code_file, "w") as handle:
                handle.write(final_code)

            create_file(model_dir, new_out_file, normalized_completion)
            code_logger.log_to_file(f"[INFO] Saved successful code to B{B_index} (Signature: {sig})")
            saved_graph_counts[graph_info.graph_hash] += 1
            saved_family_hash_counts[graph_info.family_hash] += 1
            if backbone_signature:
                saved_backbone_signature_counts[backbone_signature] += 1
            if cnn_signature:
                saved_cnn_signature_counts[cnn_signature] += 1
            if backbone_signature and cnn_signature:
                saved_backbone_cnn_pair_counts[backbone_cnn_pair_key] += 1
                saved_best_reward_target_by_backbone_cnn[backbone_cnn_pair_key] = max(
                    float(saved_best_reward_target_by_backbone_cnn.get(backbone_cnn_pair_key, float("-inf"))),
                    float(reward_target_value if reward_target_value is not None else float("-inf")),
                )
            get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] += 1
            B_index += 1
        elif (
            bool(graph_info)
            and graph_info.parse_ok
            and res.get("built_ok")
            and res.get("forward_shape_ok")
            and res.get("backward_ok")
            and _has_completed_formal_epoch(res)
        ):
            code_logger.log_to_file(
                f"[INFO] Skipped save for signature={sig} backbone={backbone_signature} cnn={cnn_signature} "
                f"reason={save_gate_reason} reward_target={reward_target_value!r}"
            )

        generation_total = _current_generation_total() + 1
        _record_generation_event(
            {
                "generation_total": generation_total,
                "reward_batch_index": res.get("reward_batch_index"),
                "reward_group_id": res.get("reward_group_id"),
                "stage_name": str(res.get("current_stage_name") or current_stage_name),
                "stage_index": int(res.get("current_stage_index") or RL_STAGE_TO_INDEX.get(current_stage_name, 0)),
                "family_hash": str(res.get("family_hash") or getattr(graph_info, "family_hash", "") or ""),
                "graph_hash": str(res.get("graph_hash") or getattr(graph_info, "graph_hash", "") or ""),
                "descriptor_key": str(res.get("descriptor_key") or getattr(graph_info, "descriptor_key", "") or ""),
                "backbone_signature": backbone_signature,
                "cnn_signature": cnn_signature,
                "backbone_cnn_pair_key": backbone_cnn_pair_key,
                "reward": score,
                "reward_target_metric": str(res.get("reward_target_metric") or ""),
                "reward_target_value": reward_target_value,
                "formal_reward_epochs": list(res.get("formal_reward_epochs") or []),
                "formal_reward_max_epoch": int(res.get("formal_reward_max_epoch", 0) or 0),
                "formal_horizon_test_acc": dict(res.get("formal_horizon_test_acc") or {}),
                "formal_horizon_train_acc": dict(res.get("formal_horizon_train_acc") or {}),
                "formal_horizon_scores": dict(res.get("formal_horizon_scores") or {}),
                "formal_reward_target_value": _optional_float(res.get("formal_reward_target_value")),
                "loss_end": _optional_float(res.get("loss_end")),
                "best_epoch_loss": _optional_float(res.get("best_epoch_loss")),
                "avg_epoch_loss": _optional_float(res.get("avg_epoch_loss")),
                "epochs_completed": int(res.get("epochs_completed", 0) or 0),
                "training_context_metric_name": str(res.get("training_context_metric_name") or ""),
                "training_context_metric_value": _optional_float(res.get("training_context_metric_value")),
                "trained_step_ok": bool(res.get("trained_step_ok")),
                "backward_ok": bool(res.get("backward_ok")),
                "loss_drop_ok": bool(res.get("loss_drop_ok")),
                "executable_candidate": bool(res.get("executable_candidate", is_executable)),
                "discovery_candidate": bool(res.get("discovery_candidate")),
                "formal_success_candidate": bool(res.get("formal_success_candidate", is_trainable)),
                "dominant_backbone_signature": dominant_backbone_signature,
                "dominant_backbone_share": dominant_backbone_share,
                "dominant_cnn_signature": dominant_cnn_signature,
                "dominant_cnn_share": dominant_cnn_share,
                "dominant_backbone_cnn_pair": dominant_backbone_cnn_pair,
                "dominant_backbone_cnn_share": dominant_backbone_cnn_share,
            }
        )

        code_logger.log_generation(prompt, completion, score, res)

    update_current_group_metrics(current_batch_results)
    group_close_result = close_reward_group_if_needed()
    if group_close_result is not None:
        code_logger.log_to_file(f"[Reward Group] {group_close_result}")


def _print_discovery_metrics() -> None:
    total_valid = sum(family_hash_archive_counts.values())
    unique_count = len(graph_archive_counts)
    unique_families = len(family_archive_counts)
    unique_skeletons = len(family_hash_archive_counts)
    unique_descriptors = len(descriptor_archive_counts)
    unique_backbones = len(backbone_signature_archive_counts)
    unique_cnns = len(cnn_signature_archive_counts)
    unique_backbone_cnn_pairs = len(backbone_cnn_pair_archive_counts)

    if total_valid > 0:
        most_common_count = family_hash_archive_counts.most_common(1)[0][1]
        dominant_share = most_common_count / total_valid
        import math
        entropy = -sum(
            (count / total_valid) * math.log2(count / total_valid)
            for count in family_hash_archive_counts.values()
            if count > 0
        )
    else:
        dominant_share = 0.0
        entropy = 0.0

    print(
        f"\n[Discovery Metrics] Unique Graphs: {unique_count}, "
        f"Families: {unique_families}, Skeletons: {unique_skeletons}, Descriptors: {unique_descriptors}, "
        f"Backbone Buckets: {unique_backbones}, CNN Signatures: {unique_cnns}, Backbone+CNN Pairs: {unique_backbone_cnn_pairs}, "
        f"Dominant Family Share: {dominant_share:.2%}, Entropy: {entropy:.2f}"
    )
    print(f"[Graph Archive] Top 5 Exact Graphs: {dict(graph_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Family IDs: {dict(family_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Skeletons: {dict(family_hash_archive_counts.most_common(5))}")
    print(f"[Descriptor Archive] Top 5: {dict(descriptor_archive_counts.most_common(5))}")
    print(f"[Backbone Archive] Top 5: {dict(backbone_signature_archive_counts.most_common(5))}")
    print(f"[CNN Archive] Top 5: {dict(cnn_signature_archive_counts.most_common(5))}")
    print(f"[Backbone+CNN Archive] Top 5: {dict(backbone_cnn_pair_archive_counts.most_common(5))}")
    print(f"[Motif Names] Top 5: {dict(motif_name_counts.most_common(5))}")
    goal_summary = {
        goal_key: len(counter)
        for goal_key, counter in goal_family_hash_archive_counts.items()
    }
    print(f"[Goal Skeleton Coverage] {goal_summary}")


def compute_reward(prompts, completions, **kwargs):
    clear_extraction_meta_cache()
    seed_accuracy_baselines = require_sample_accuracy_baselines(kwargs, len(completions))
    group_context = current_reward_group_context()

    try:
        expected_world_size = max(1, env_int("WORLD_SIZE", 1))
        distributed_mode = _distributed_initialized() and _distributed_world_size() > 1
        if expected_world_size > 1 and not distributed_mode:
            raise RuntimeError(
                "compute_reward expected an initialized torch.distributed process group "
                f"for WORLD_SIZE={expected_world_size}, but it is not initialized"
            )

        rank = _distributed_rank()
        precompute_eval = not distributed_mode
        if not precompute_eval:
            print(
                "[Reward Precompute Local] skip "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                "reason='distributed_global_sharded_gpu_eval'"
            )
        local_entries = _prepare_local_reward_entries(
            prompts,
            completions,
            seed_accuracy_baselines=seed_accuracy_baselines,
            group_context=group_context,
            precompute_eval=precompute_eval,
        )
        archive_snapshot_family_counts = dict(family_hash_archive_counts)

        if not distributed_mode:
            scored_results = _score_reward_entries(
                local_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            rewards = [-1.0] * len(completions)
            for item in scored_results:
                rewards[int(item["local_index"])] = float(item["score"])
            _finalize_scored_results(scored_results)
            _print_discovery_metrics()
            return rewards

        print(
            "[Reward Gather] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"local_entries={len(local_entries)}"
        )
        gathered_entries = _all_gather_object(local_entries)
        total_entries = sum(len(rank_entries or []) for rank_entries in gathered_entries)
        print(
            "[Reward Gather] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"gathered_ranks={len(gathered_entries)} "
            f"total_entries={total_entries}"
        )

        global_entries = _build_global_reward_entries(gathered_entries)
        assigned_entries = _select_global_reward_entries_for_rank(
            global_entries,
            rank=rank,
            world_size=len(gathered_entries),
        )
        print(
            "[Reward Shard] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"global_entries={len(global_entries)} "
            f"assigned_entries={len(assigned_entries)}"
        )
        _precompute_eval_results(
            assigned_entries,
            group_context=group_context,
        )
        print(
            "[Reward Shard] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"assigned_entries={len(assigned_entries)}"
        )

        gathered_precomputed_entries = _all_gather_object(assigned_entries)
        merged_precomputed_entries = _merge_gathered_reward_entries(
            gathered_precomputed_entries,
            expected_count=len(global_entries),
        )

        if is_main_process():
            print(
                "[Reward Score] start "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                f"entries={len(merged_precomputed_entries)}"
            )
            scored_results = _score_reward_entries(
                merged_precomputed_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            _finalize_scored_results(scored_results)
            _print_discovery_metrics()
            print(
                "[Reward Score] end "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                f"entries={len(scored_results)}"
            )

            rewards_by_rank: Dict[int, List[float]] = {
                world_rank: [-1.0] * len(gathered_entries[world_rank])
                for world_rank in range(len(gathered_entries))
            }
            for item in scored_results:
                rewards_by_rank[int(item["rank"])][int(item["local_index"])] = float(item["score"])

            broadcast_payload = {
                "rewards_by_rank": rewards_by_rank,
                "reward_state": capture_reward_runtime_state(),
            }
        else:
            broadcast_payload = None

        print(
            "[Reward Broadcast] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')}"
        )
        synced_payload = _broadcast_object(broadcast_payload, src=0)
        print(
            "[Reward Broadcast] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')}"
        )
        restore_reward_runtime_state(synced_payload.get("reward_state"))
        return list(synced_payload["rewards_by_rank"].get(rank, [-1.0] * len(completions)))
    finally:
        clear_extraction_meta_cache()

PROMPT_TEMPLATE = SFTUtil.open_discovery_prompt_template
PROMPT_BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
PROMPT_INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
PROMPT_FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"

def load_rl_dataset(tokenizer):
    """Load seed tasks for open-ended architecture discovery."""
    data = api.data(task='img-classification', nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = api.data(only_best_accuracy=True, task='img-classification', dataset='cifar-10')

    print(f"Loaded {len(data)} examples for RL")
    bootstrap_trainset_reference_library(data)

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = _coerce_accuracy_baseline(row.get('accuracy'), context="seed row accuracy")
        for profile in goal_profiles:
            target_pattern = SFTUtil.goal_profile_target_pattern(profile)
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                target_pattern=target_pattern,
                design_brief=profile["brief"],
                tag_realization=profile.get("realization", profile["brief"]),
                goal_tag_parser_cues=SFTUtil.goal_tag_parser_cues(profile["tags"]),
                module_hints=", ".join(module_hints),
                block_signature=PROMPT_BLOCK_SIGNATURE,
                init_signature=PROMPT_INIT_SIGNATURE,
                forward_signature=PROMPT_FORWARD_SIGNATURE,
            )

            messages = [{"role": "user", "content": user_prompt}]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            prompts.append({
                "prompt": prompt_str,
                "accuracy": accuracy,
                "goal_name": profile["name"],
                "target_tags": ", ".join(profile["tags"]),
            })

    rl_dataset = Dataset.from_list(prompts)
    return rl_dataset.shuffle(seed=42)

def main():
    global active_rl_model
    global active_rl_tokenizer
    global current_stage_name

    torch.cuda.empty_cache()
    resume_checkpoint_dir = _resolve_resume_checkpoint_dir()
    resume_manifest = None
    restored_reward_state_path = None
    resume_stage_override = os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
    if resume_checkpoint_dir is not None:
        resume_manifest = _load_json_if_exists(resume_checkpoint_dir / "runtime_manifest.json")
        if resume_manifest is None:
            resume_manifest = _load_json_if_exists(resume_checkpoint_dir / "stage_manifest.json")
        # Restore runtime state through the shared helper before stage-specific overrides run.
        restored_reward_state_path = TrainingRuntime.restore_or_reset_runtime_state(
            resume_checkpoint_dir,
            _reward_runtime_hooks(),
            legacy_state_filenames=("reward_state.json",),
        )
        if resume_stage_override:
            current_state_stage = str(current_stage_name)
            if current_state_stage != resume_stage_override:
                print(
                    "[RL] Resume stage override "
                    f"checkpoint_stage={current_state_stage} requested_stage={resume_stage_override}"
                )
                current_stage_name = resume_stage_override
        print(
            "[RL] Resuming from checkpoint "
            f"dir={resume_checkpoint_dir} stage={current_stage_name} "
            f"generation_total={_current_generation_total()} reward_batch_index={reward_batch_index}"
        )
        if restored_reward_state_path is not None:
            print(f"[RL] Restored runtime state from {restored_reward_state_path}")
    else:
        TrainingRuntime.restore_or_reset_runtime_state(
            None,
            _reward_runtime_hooks(),
            legacy_state_filenames=("reward_state.json",),
        )
    precision = best_mixed_precision()
    runtime = get_distributed_runtime_info()
    runtime_settings = resolve_rl_runtime_settings(runtime)
    rank = int(runtime.get("rank", 0))
    local_rank = int(runtime.get("local_rank", 0))
    raw_local_rank = int(runtime.get("raw_local_rank", 0))
    world_size = int(runtime.get("world_size", 1))
    use_deepspeed = _resolve_rl_deepspeed_enabled(runtime)
    deepspeed_config_path = _resolve_rl_deepspeed_config_path() if use_deepspeed else None
    os.environ["NNGPT_SFT_USE_DEEPSPEED"] = "1" if use_deepspeed else "0"
    if deepspeed_config_path is not None:
        os.environ["NNGPT_SFT_DEEPSPEED_CONFIG"] = deepspeed_config_path
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    hf_deepspeed_config = _maybe_init_hf_deepspeed_config(deepspeed_config_path) if use_deepspeed else None

    print(f"Using RL base model: {base_model}")
    print(
        "[RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    print(f"[RL] DeepSpeed Enabled: {use_deepspeed}")
    if deepspeed_config_path is not None:
        print(f"[RL] DeepSpeed Config: {deepspeed_config_path}")
    print(f"[RL] Fixed training device: {train_device}")
    print(f"[RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    print(f"[RL] Current stage: {current_stage_name}")
    print(
        "[RL] Runtime limits: "
        f"dataset_limit={runtime_settings['dataset_limit']} "
        f"max_completion_length={runtime_settings['max_completion_length']} "
        f"grad_accum={runtime_settings['grad_accum']} "
        f"effective_train_batch_size={runtime_settings['effective_train_batch_size']} "
        f"requested_global_num_generations={runtime_settings['requested_global_num_generations']} "
        f"global_num_generations={runtime_settings['global_num_generations']} "
        f"effective_global_num_generations={runtime_settings['effective_global_num_generations']}"
    )
    if runtime_settings["global_num_generations_adapted"]:
        print(
            "[RL] Generation plan adapted "
            f"requested={runtime_settings['requested_global_num_generations']} "
            f"effective={runtime_settings['effective_global_num_generations']} "
            f"valid_generation_values={runtime_settings['valid_generation_values']} "
            f"world_size={world_size}"
        )
    tokenizer = TrainerRuntime.load_tokenizer(base_model)

    # Load RL dataset (limit for training speed)
    rl_dataset = load_rl_dataset(tokenizer)
    dataset_limit = runtime_settings["dataset_limit"]
    if len(rl_dataset) > dataset_limit:
        rl_dataset = rl_dataset.select(range(dataset_limit))

    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=base_model,
        precision=precision,
        train_device=train_device,
        use_deepspeed=use_deepspeed,
    )
    _ = hf_deepspeed_config

    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        model = TrainerRuntime.maybe_merge_initial_adapter(
            model,
            enabled=True,
            adapter_path=SAVED_MODEL_PATH,
            label="extra SFT",
            load_message=f"Loading extra SFT adapter from {SAVED_MODEL_PATH}...",
        )

    model = prepare_model_for_kbit_training(model)
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Apply LoRA specifically for RL phase
    peft_config = TrainerRuntime.build_lora_config(
        r=16,
        alpha=32,
        dropout=0.05,
    )
    resume_adapter_dir = (resume_checkpoint_dir / "adapter") if resume_checkpoint_dir is not None else None
    model = TrainerRuntime.attach_or_resume_lora(
        model,
        peft_config=peft_config,
        stage_adapter_dir=resume_adapter_dir,
        log_prefix="[RL]",
        missing_adapter_message=f"Missing adapter directory under resume checkpoint: {resume_adapter_dir}",
        load_message=f"[RL] Loading RL adapter from {resume_adapter_dir}..." if resume_adapter_dir is not None else None,
    )
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Enable gradient checkpointing to save memory
    TrainerRuntime.enable_non_reentrant_gradient_checkpointing(
        model,
        log_prefix="[RL]",
    )

    model.print_trainable_parameters()
    active_rl_model = model
    active_rl_tokenizer = tokenizer
    evaluate_code_and_reward._nngpt_eval_cfg_builder = build_stage_eval_cfg
    stage_entry_generation_totals.setdefault(current_stage_name, _current_generation_total())
    stage_entry_reward_batches.setdefault(current_stage_name, reward_batch_index)
    if resume_checkpoint_dir is None:
        _append_stage_event(
            {
                "event": "entered",
                "reason": "initial_stage_entry",
                "previous_stage_name": None,
                "next_stage_name": current_stage_name,
            }
        )
        _save_stage_checkpoint(
            "entered",
            stage_name=current_stage_name,
            reason="initial_stage_entry",
        )

    grpo_config = _build_rl_grpo_config(
        precision=precision,
        use_deepspeed=use_deepspeed,
        deepspeed_config_path=deepspeed_config_path,
        runtime_settings=runtime_settings,
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward,
        args=grpo_config,
    )
    trainer_gc_patch_stats = TrainerRuntime.enforce_non_reentrant_gradient_checkpointing(trainer.model)
    print(
        "[RL] Trainer gradient checkpointing enforcement: "
        f"roots={trainer_gc_patch_stats['roots']} modules={trainer_gc_patch_stats['modules']} use_reentrant=False"
    )
    prewarm_eval_workers(timeout_seconds=60.0, require_gpu=True)
    register_stage_checkpoint_signal_handlers()

    print("Starting GRPO training for Backbone Search...")
    try:
        TrainerRuntime.train_grpo(
            trainer=trainer,
            trainer_checkpoint=None,
            log_prefix="[RL]",
        )
    except Exception as exc:
        if is_cuda_oom_error(exc):
            log_cuda_oom_diagnostics("rl/trainer.train", exc)
        raise
    finally:
        shutdown_eval_worker()

    model_out = run_model_out()
    print(f"Saving model to {model_out}...")
    model.save_pretrained(model_out)
    _save_stage_checkpoint(
        "completed",
        stage_name=current_stage_name,
        reason="trainer_completed",
    )
    try:
        code_logger.save_log()
    except Exception as exc:
        code_logger.log_to_file(f"[RL] save_log failed: {type(exc).__name__}: {exc}")
    print("Model saved successfully!")

    return model

if __name__ == "__main__":
    from ab.gpt.util.simple_logger import SimpleCodeLogger
    from ab.gpt.util.Reward import evaluate_code_and_reward
    from typing import Dict

    # Ensure directories exist
    log_dir = run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    code_logger = SimpleCodeLogger(log_dir)

    # 清空旧模型目录
    if _resolve_resume_checkpoint_dir() is None:
        print(f"Cleaning existing models in {run_epoch_dir()}...")
        shutil.rmtree(run_epoch_dir(), ignore_errors=True)
    else:
        print(f"Resuming run: keeping existing synthesized models under {run_epoch_dir()}")

    main()

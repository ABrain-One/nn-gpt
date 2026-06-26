import json
import os
import re
import shutil
import random
import inspect
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset as TorchDataset
from ab.gpt.util.Const import conf_dir

try:
    import faulthandler

    faulthandler.enable(all_threads=True)
except Exception:
    pass


# ── SFT runtime configuration ─────────────────────────────────────────────
SFT_BASE_MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
SFT_INIT_ADAPTER = ""
SFT_LOAD_INITIAL_ADAPTER = False
SFT_INITIAL_ADAPTER_MODE = "trainable"
SFT_INITIAL_ADAPTER_DTYPE = "fp32"
SFT_SAVE_RL_MODEL = False
SFT_MODEL_OUT = "rl_backbone_model_sft"
SFT_LOG_DIR = "rl_output/sft"
SFT_EPOCH_ROOT = "out/nngpt/llm/epoch_sft"
SFT_TRAINER_OUT = "grpo_backbone_outputs/sft"
SFT_TEMPERATURE = 0.8
SFT_TOP_P = 0.95
SFT_TOP_K = 50
SFT_NUM_GENERATIONS = 8
SFT_GRAD_ACCUM = 8
SFT_MAX_PROMPT_LENGTH = 3500
SFT_MAX_COMPLETION_LENGTH = 1200
SFT_GENERATION_BATCH_SIZE = 8
SFT_MAX_STEPS = 125
SFT_DATASET_LIMIT = 500
SFT_FEEDBACK_CHAR_BUDGET = 0
SFT_LR = 5e-5
SFT_NUM_EPOCHS = 5
SFT_SAVE_STEPS = 25
SFT_KL_COEF = 0.04
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05
SFT_RL_NN_PREFIXES = ("rl-bb-test1",)
SFT_RL_PROMPT_MODE = "sft_aligned"
SFT_RL_RESUME_STAGE = "stage2_formal_explore"
SFT_FORMAL_REWARD_EPOCHS = "1"
SFT_DEEPSPEED_DEFAULT_CONFIG = str(conf_dir / "DeepSpeedSftGrpo.json")
SFT_MODE_DEFAULT = "auto"
SFT_REWARD_EXCLUDE_TRAIN_GPU = False
SFT_COMPACT_AFTER_MODEL_LOAD = False
SFT_COMPACT_FLOAT_PARAMS = False
SFT_LENGTH_SOFT_TOKEN_LIMIT = 850
SFT_LENGTH_HARD_TOKEN_LIMIT = 1000
SFT_LENGTH_FAILURE_TOKEN_LIMIT = 1200
SFT_LENGTH_SOFT_PENALTY = 0.05
SFT_LENGTH_HARD_EXTRA_PENALTY = 0.20
SFT_REPEAT_LINE_PENALTY = 0.03
SFT_REPEAT_LINE_MAX_PENALTY = 0.15
SFT_RUNTIME_SOURCE_INFO: Dict[str, str] = {}

# CIFAR-10 reward evaluation via nn-dataset / NNEval-aligned formal acc.
SFT_EVAL_IMAGE_SIZE = 128
SFT_EVAL_BATCH_SIZE = 64
SFT_EVAL_TRAIN_SUBSET = 0
SFT_EVAL_VAL_SUBSET = 0
SFT_EVAL_TRAIN_EPOCHS = 1
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_FULL_TEST_ACC = True
SFT_EVAL_RUN_UNFROZEN = False
SFT_EVAL_LIMIT_SECONDS = 900
SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES = 30
SFT_EVAL_DATA_ROOT = "data_v2"
SFT_EVAL_DOWNLOAD = True
SFT_EVAL_SPLIT_PROTOCOL = "trainvaltest"
SFT_EVAL_SPLIT_SEED = 42
SFT_EVAL_SPLIT_ROLE = "reward_eval"
SFT_VAL_METRIC_BASELINE = 0.10
SFT_FORMAL_DATASET_CLASS_COUNTS = {
    "cifar-10": 10,
    "cifar-100": 100,
    "imagenette": 10,
}
SFT_FORMAL_DATASET_ALIASES = {
    "cifar10": "cifar-10",
    "cifar-10": "cifar-10",
    "cifar100": "cifar-100",
    "cifar-100": "cifar-100",
    "imagenette": "imagenette",
}

import ab.gpt.TuneRL as TuneRL
from ab.gpt.rl_pipeline.completion import (
    BLOCK_SIGNATURE,
    FORWARD_SIGNATURE,
    INIT_SIGNATURE,
    clear_extraction_meta_cache,
    extract_completion_blocks_tolerant,
    extract_completion_meta,
)
import ab.gpt.rl_pipeline.trainer_runtime as TrainerRuntime
import ab.gpt.util.Reward as RewardUtil
import ab.gpt.util.SFTUtil as SFTUtil
import ab.gpt.util.training_runtime as TrainingRuntime

SFT_EVAL_TRANSFORM = TuneRL.FORMAL_REWARD_TRANSFORM


_TRAIN_GPU_TOKENS_ENV = "NNGPT_TRAIN_GPU_TOKENS"
_AUX_GPU_TOKENS_ENV = "NNGPT_AUX_GPU_TOKENS"
_REWARD_GPU_TOKENS_ENV = "NNGPT_REWARD_GPU_TOKENS"


def normalize_sft_formal_dataset(dataset: str | None = None) -> str:
    raw = str(dataset or os.getenv("NNGPT_RL_FORMAL_DATASET", "cifar-10")).strip().lower()
    if raw not in SFT_FORMAL_DATASET_ALIASES:
        allowed = ", ".join(sorted(SFT_FORMAL_DATASET_CLASS_COUNTS))
        raise ValueError(f"Unsupported NNGPT_RL_FORMAL_DATASET={raw!r}; expected one of: {allowed}")
    return SFT_FORMAL_DATASET_ALIASES[raw]


def resolve_sft_formal_dataset() -> str:
    return normalize_sft_formal_dataset()


def resolve_sft_formal_n_classes(dataset: str | None = None) -> int:
    return int(SFT_FORMAL_DATASET_CLASS_COUNTS[normalize_sft_formal_dataset(dataset)])


def resolve_sft_formal_out_shape(dataset: str | None = None) -> tuple[int, ...]:
    return (resolve_sft_formal_n_classes(dataset),)


def _normalize_sft_eval_split_protocol(_split_protocol: str | None = None) -> str:
    return "trainvaltest"


def _describe_sft_eval_split(formal_dataset: str, split_protocol: str) -> tuple[str, str, str]:
    dataset = normalize_sft_formal_dataset(formal_dataset)
    _normalize_sft_eval_split_protocol(split_protocol)
    if dataset == "cifar-10":
        return "cifar10-train[45k]", "cifar10-train[5k]", "cifar10-test[10k]"
    if dataset == "cifar-100":
        return "cifar100-train[45k]", "cifar100-train[5k]", "cifar100-test[10k]"
    if dataset == "imagenette":
        return "imagenette-train[7500]", "imagenette-train[1969]", "imagenette-test[3925]"
    raise ValueError(f"Unsupported formal dataset: {formal_dataset!r}")


def _stage1_fixed_failure_reward(res: Dict[str, Any], meta: Dict[str, Any], graph_info) -> Optional[float]:
    if str(res.get("current_stage_name") or "") != TuneRL.STAGE1_STRUCTURE_EXPLORE:
        return None
    if TuneRL._is_trainable_candidate(res, graph_info):
        return None

    if (
        not meta.get("xml_tag_exact")
        or int(meta.get("xml_tag_count", 0) or 0) < 3
        or int(meta.get("class_count", 0) or 0) > 0
        or int(meta.get("import_count", 0) or 0) > 0
        or int(meta.get("bad_signature_count", 0) or 0) > 0
        or not meta.get("dual_backbone_ok")
        or not res.get("built_ok")
    ):
        return -3.0
    if not res.get("forward_ok") or not res.get("forward_shape_ok"):
        return -1.0
    return -0.25


def _estimate_completion_tokens(completion: str) -> tuple[int, str]:
    tokenizer = getattr(TuneRL, "active_rl_tokenizer", None)
    if tokenizer is not None:
        try:
            tokenized = tokenizer(completion, add_special_tokens=False)
            return int(len(tokenized.input_ids)), "tokenizer"
        except Exception:
            pass
    return int(max(1, round(len(completion) / 2.4))), "char_estimate"


def _count_repeated_code_lines(completion: str) -> tuple[int, int]:
    counts: Dict[str, int] = {}
    for raw_line in completion.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<") or len(line) < 12:
            continue
        line = re.sub(r"\s+", " ", line)
        counts[line] = counts.get(line, 0) + 1
    repeated_groups = sum(1 for value in counts.values() if value >= 3)
    repeated_excess = sum(value - 2 for value in counts.values() if value >= 3)
    return repeated_groups, repeated_excess


def _completion_compactness_penalty(completion: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    token_count, token_count_source = _estimate_completion_tokens(completion)
    soft_limit = max(1, _env_int("NNGPT_SFT_LENGTH_SOFT_TOKEN_LIMIT", SFT_LENGTH_SOFT_TOKEN_LIMIT))
    hard_limit = max(soft_limit + 1, _env_int("NNGPT_SFT_LENGTH_HARD_TOKEN_LIMIT", SFT_LENGTH_HARD_TOKEN_LIMIT))
    failure_limit = max(hard_limit + 1, _env_int("NNGPT_SFT_LENGTH_FAILURE_TOKEN_LIMIT", SFT_LENGTH_FAILURE_TOKEN_LIMIT))
    soft_penalty = max(0.0, _env_float("NNGPT_SFT_LENGTH_SOFT_PENALTY", SFT_LENGTH_SOFT_PENALTY))
    hard_extra_penalty = max(
        0.0,
        _env_float("NNGPT_SFT_LENGTH_HARD_EXTRA_PENALTY", SFT_LENGTH_HARD_EXTRA_PENALTY),
    )
    repeat_unit = max(0.0, _env_float("NNGPT_SFT_REPEAT_LINE_PENALTY", SFT_REPEAT_LINE_PENALTY))
    repeat_cap = max(0.0, _env_float("NNGPT_SFT_REPEAT_LINE_MAX_PENALTY", SFT_REPEAT_LINE_MAX_PENALTY))

    length_penalty = 0.0
    if token_count > soft_limit:
        if token_count <= hard_limit:
            length_penalty = -soft_penalty * ((token_count - soft_limit) / float(hard_limit - soft_limit))
        else:
            hard_span = max(1, failure_limit - hard_limit)
            length_penalty = -soft_penalty - hard_extra_penalty * min(
                1.0,
                (token_count - hard_limit) / float(hard_span),
            )

    repeated_groups, repeated_excess = _count_repeated_code_lines(completion)
    repeated_line_penalty = -min(repeat_cap, repeat_unit * repeated_excess)
    xml_incomplete_length_cap = bool(token_count > failure_limit and not meta.get("xml_tag_exact"))
    return {
        "completion_token_count": token_count,
        "completion_token_count_source": token_count_source,
        "completion_length_soft_limit": soft_limit,
        "completion_length_hard_limit": hard_limit,
        "completion_length_failure_limit": failure_limit,
        "repeated_line_groups": repeated_groups,
        "repeated_line_excess": repeated_excess,
        "r_length_compactness": length_penalty,
        "r_repeated_line_penalty": repeated_line_penalty,
        "xml_incomplete_length_cap": xml_incomplete_length_cap,
    }


def raw_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Dict[str, Any] | None = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    batch_block_signatures: List[str] = None,
    batch_backbone_block_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    prompt_target_pattern: str = "",
    archive_snapshot_family_counts: Dict[str, int] = None,
    archive_snapshot_descriptor_counts: Dict[str, int] = None,
    archive_snapshot_backbone_signature_counts: Dict[str, int] = None,
    archive_snapshot_cnn_signature_counts: Dict[str, int] = None,
    archive_snapshot_graph_counts: Dict[str, int] = None,
    archive_snapshot_block_signature_counts: Dict[str, int] = None,
    archive_snapshot_backbone_cnn_pair_counts: Dict[str, int] = None,
    archive_snapshot_backbone_block_pair_counts: Dict[str, int] = None,
    archive_snapshot_backbone_block_best_quality: Dict[str, float] = None,
    group_baseline_train_acc: float | None = None,
    group_baseline_reward_target_acc: float | None = None,
    reward_batch_index: int | None = None,
    reward_group_id: int | None = None,
    group_warmup: bool = False,
    completion_index: int | None = None,
    batch_last_item: bool = False,
):
    res = TuneRL.base_discovery_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        batch_block_signatures=batch_block_signatures,
        batch_backbone_block_signatures=batch_backbone_block_signatures,
        prompt_goal_tags=prompt_goal_tags,
        prompt_target_pattern=prompt_target_pattern,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_graph_counts=archive_snapshot_graph_counts,
        archive_snapshot_block_signature_counts=archive_snapshot_block_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        archive_snapshot_backbone_block_pair_counts=archive_snapshot_backbone_block_pair_counts,
        archive_snapshot_backbone_block_best_quality=archive_snapshot_backbone_block_best_quality,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )
    meta = extract_completion_meta(completion)
    raw_delta = 0.0

    raw_delta -= 0.35 * min(int(meta.get("class_count", 0)), 2)
    raw_delta -= 0.12 * min(int(meta.get("import_count", 0)), 2)
    raw_delta -= 0.35 * min(int(meta.get("bad_signature_count", 0)), 2)

    xml_tag_count = int(meta.get("xml_tag_count", 0))
    if xml_tag_count < 3:
        raw_delta -= 0.45 * (3 - xml_tag_count)
    if not meta.get("xml_tag_exact"):
        raw_delta -= 1.20
    if not meta.get("exact_block_signature"):
        raw_delta -= 0.50
    if not meta.get("exact_init_signature"):
        raw_delta -= 0.60
    if not meta.get("exact_forward_signature"):
        raw_delta -= 0.60

    terminal_penalty = 0.0
    if meta.get("trailing_after_forward"):
        terminal_penalty -= 0.20
        if int(meta.get("trailing_after_forward_chars", 0) or 0) > 300:
            terminal_penalty -= 0.10
    jupyter_artifact_count = int(meta.get("jupyter_artifact_count", 0) or 0)
    if jupyter_artifact_count:
        terminal_penalty -= 0.25 * min(jupyter_artifact_count, 2)
    raw_delta += terminal_penalty

    if meta.get("dual_backbone_ok"):
        raw_delta += 0.45
    else:
        if not meta.get("dual_backbone_init_ok"):
            raw_delta -= 1.75
        if not meta.get("dual_backbone_forward_ok"):
            raw_delta -= 1.75

    shape_contract_delta = 0.0
    if str(res.get("current_stage_name") or TuneRL.current_stage_name) != TuneRL.STAGE1_STRUCTURE_EXPLORE:
        if res.get("built_ok") and res.get("forward_ok") and res.get("forward_shape_ok"):
            shape_contract_delta = 0.12
        elif res.get("built_ok") and res.get("forward_ok"):
            shape_contract_delta = -0.12
        elif res.get("built_ok"):
            shape_contract_delta = -0.08
    raw_delta += shape_contract_delta

    res["reward"] = TuneRL._apply_trainability_clamp(
        res,
        float(res.get("reward", -2.0)) + raw_delta,
        graph_info,
    )

    core_format_violation = bool(
        not meta.get("xml_tag_exact")
        or not meta.get("exact_block_signature")
        or not meta.get("exact_init_signature")
        or not meta.get("exact_forward_signature")
    )
    class_count = int(meta.get("class_count", 0))
    import_count = int(meta.get("import_count", 0))
    minor_hygiene_violation = class_count > 0 or import_count > 0
    severe_hygiene_violation = class_count > 2 or import_count > 2

    if core_format_violation:
        res["reward"] = min(float(res["reward"]), -3.0)
    elif severe_hygiene_violation:
        res["reward"] = min(float(res["reward"]), -2.0)
    elif minor_hygiene_violation:
        res["reward"] = min(float(res["reward"]), -1.5)

    if not meta.get("dual_backbone_ok"):
        res["reward"] = min(float(res["reward"]), -3.5)
    elif group_warmup:
        if TuneRL._is_trainable_candidate(res, graph_info):
            res["reward"] = float(res.get("warmup_dense_reward") or 0.0)
        else:
            res["reward"] = -float(res.get("warmup_dense_reward") or 0.18)

    fixed_failure_reward = _stage1_fixed_failure_reward(res, meta, graph_info)
    if fixed_failure_reward is not None:
        res["reward"] = fixed_failure_reward
        res["stage1_fixed_failure_reward"] = True
    elif str(res.get("current_stage_name") or "") == TuneRL.STAGE1_STRUCTURE_EXPLORE:
        if TuneRL._is_trainable_candidate(res, graph_info):
            trainability_bonus = 0.10
            res["reward"] = float(res["reward"]) + trainability_bonus
            res["stage1_trainability_bonus"] = trainability_bonus

    compactness = _completion_compactness_penalty(completion, meta)
    res.update(compactness)
    res["reward"] = (
        float(res.get("reward", -2.0))
        + float(compactness["r_length_compactness"])
        + float(compactness["r_repeated_line_penalty"])
    )
    if compactness["xml_incomplete_length_cap"]:
        res["reward"] = min(float(res["reward"]), -0.8)
    if TuneRL._reward_variant_is_strong_repeat_penalty() and bool(res.get("strong_repeat_penalty_applied")):
        res["reward"] = min(float(res["reward"]), 0.0)
    res["reward"] = TuneRL._apply_target_structure_reward_gate(
        res,
        float(res.get("reward", -2.0)),
    )

    res["raw_extraction"] = {
        **meta,
        "raw_delta": raw_delta,
        "shape_contract_delta": shape_contract_delta,
        "terminal_penalty": terminal_penalty,
    }
    res.setdefault("backbone_model_names", list(meta.get("backbone_model_names", [])))
    return res


def _sft_runtime_state_hooks() -> TrainingRuntime.RuntimeStateHooks:
    # SFT reuses the RL reward runtime state so checkpoint resume stays compatible.
    return TrainingRuntime.RuntimeStateHooks(
        capture=TuneRL.capture_reward_runtime_state,
        restore=TuneRL.restore_reward_runtime_state,
        reset=TuneRL.reset_reward_runtime_state,
    )


def _repo_model_dir(model_id: str) -> Path:
    return Path("out/llm") / model_id


def _repo_tokenizer_dir(model_id: str) -> Path:
    return Path("out/tokenizer") / model_id


def resolve_sft_base_model_id() -> str:
    return _env_str("NNGPT_SFT_BASE_MODEL_ID", SFT_BASE_MODEL_ID)


def resolve_sft_tokenizer_id() -> str:
    return _env_str("NNGPT_SFT_TOKENIZER_ID", "")


def _has_model_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    return any(
        (model_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _has_tokenizer_files(tokenizer_dir: Path) -> bool:
    if not tokenizer_dir.is_dir():
        return False
    return any(
        (tokenizer_dir / filename).exists()
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    )


def resolve_sft_model_sources() -> tuple[str, str, str]:
    base_model_id = resolve_sft_base_model_id()
    tokenizer_id = resolve_sft_tokenizer_id()
    base_model_path = Path(base_model_id).expanduser()
    if base_model_path.is_dir():
        if not _has_model_files(base_model_path):
            raise RuntimeError(f"Configured SFT base model path has no model files: {base_model_path}")
        if tokenizer_id:
            tokenizer_path = Path(tokenizer_id).expanduser()
            if tokenizer_path.is_dir():
                if not _has_tokenizer_files(tokenizer_path):
                    raise RuntimeError(f"Configured SFT tokenizer path has no tokenizer files: {tokenizer_path}")
                return str(base_model_path), str(tokenizer_path), "explicit-path+explicit-tokenizer-path"
            return str(base_model_path), tokenizer_id, "explicit-path+explicit-tokenizer-id"
        if _has_tokenizer_files(base_model_path):
            return str(base_model_path), str(base_model_path), "explicit-path"
        out_llm_root = Path("out/llm").resolve()
        try:
            relative_model_id = str(base_model_path.resolve().relative_to(out_llm_root))
        except ValueError:
            relative_model_id = ""
        if relative_model_id:
            repo_tokenizer_dir = _repo_tokenizer_dir(relative_model_id)
            if _has_tokenizer_files(repo_tokenizer_dir):
                return str(base_model_path), str(repo_tokenizer_dir), "explicit-path+out/tokenizer"
        raise RuntimeError(f"Configured SFT base model path has no tokenizer files: {base_model_path}")

    repo_model_dir = _repo_model_dir(base_model_id)
    repo_tokenizer_dir = _repo_tokenizer_dir(base_model_id)

    if _has_model_files(repo_model_dir):
        if _has_tokenizer_files(repo_model_dir):
            return str(repo_model_dir), str(repo_model_dir), "out/llm"
        if _has_tokenizer_files(repo_tokenizer_dir):
            return str(repo_model_dir), str(repo_tokenizer_dir), "out/llm+out/tokenizer"
        return str(repo_model_dir), base_model_id, "out/llm+model-id-tokenizer"

    if _has_tokenizer_files(repo_tokenizer_dir):
        return base_model_id, str(repo_tokenizer_dir), "model-id+out/tokenizer"

    return base_model_id, base_model_id, "model-id-download"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return str(default)
    return str(raw).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _env_optional_json(name: str) -> Any | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return json.loads(raw)


def resolve_sft_rl_seed() -> int:
    return TuneRL.resolve_rl_seed()


def _set_optional_grpo_config(
    config_kwargs: Dict[str, Any],
    signature_parameters: Dict[str, inspect.Parameter],
    name: str,
    value: Any,
) -> None:
    if value is None:
        return
    if name not in signature_parameters:
        raise RuntimeError(f"Installed GRPOConfig does not support `{name}`")
    config_kwargs[name] = value


def resolve_sft_init_adapter() -> str:
    return _env_str("NNGPT_SFT_INIT_ADAPTER", SFT_INIT_ADAPTER)


def resolve_sft_load_initial_adapter() -> bool:
    raw = os.getenv("NNGPT_SFT_LOAD_INITIAL_ADAPTER")
    if raw is not None and raw != "":
        return _env_flag("NNGPT_SFT_LOAD_INITIAL_ADAPTER", SFT_LOAD_INITIAL_ADAPTER)
    return bool(SFT_LOAD_INITIAL_ADAPTER or resolve_sft_init_adapter().strip())


def resolve_sft_initial_adapter_mode() -> str:
    mode = _env_str("NNGPT_SFT_INITIAL_ADAPTER_MODE", SFT_INITIAL_ADAPTER_MODE).strip().lower()
    aliases = {
        "train": "trainable",
        "trainable": "trainable",
        "peft": "trainable",
        "peft_unmerged": "trainable",
        "unmerged": "trainable",
        "merge": "merge",
        "merged": "merge",
        "merge_and_unload": "merge",
    }
    if mode not in aliases:
        raise ValueError("NNGPT_SFT_INITIAL_ADAPTER_MODE must be one of: trainable, merge")
    return aliases[mode]


def resolve_sft_initial_adapter_dtype_label() -> str:
    raw = _env_str("NNGPT_SFT_INITIAL_ADAPTER_DTYPE", SFT_INITIAL_ADAPTER_DTYPE).strip().lower()
    aliases = {
        "": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "full": "fp32",
        "float16": "fp16",
        "fp16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "mixed": "precision",
        "precision": "precision",
        "auto": "precision",
    }
    if raw not in aliases:
        raise ValueError("NNGPT_SFT_INITIAL_ADAPTER_DTYPE must be one of: fp32, fp16, bf16, precision")
    return aliases[raw]


def resolve_sft_initial_adapter_dtype(label: str, precision: Dict[str, Any]) -> Any:
    import torch

    if label == "precision":
        return precision["torch_dtype"]
    if label == "fp32":
        return torch.float32
    if label == "fp16":
        return torch.float16
    if label == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("NNGPT_SFT_INITIAL_ADAPTER_DTYPE=bf16 requested, but CUDA bf16 is not supported.")
        return torch.bfloat16
    raise ValueError(f"Unsupported SFT initial adapter dtype label: {label}")


def resolve_sft_rl_nn_prefixes() -> tuple[str, ...]:
    raw = os.getenv("NNGPT_SFT_RL_NN_PREFIXES", "").strip()
    if not raw:
        return tuple(SFT_RL_NN_PREFIXES)
    prefixes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not prefixes:
        raise ValueError("NNGPT_SFT_RL_NN_PREFIXES must contain at least one prefix")
    return prefixes


def resolve_sft_rl_prompt_mode() -> str:
    mode = _env_str("NNGPT_SFT_RL_PROMPT_MODE", SFT_RL_PROMPT_MODE).strip().lower()
    aliases = {
        "sft": "sft_aligned",
        "sft-aligned": "sft_aligned",
        "sft_aligned": "sft_aligned",
        "goal": "goal_profiles",
        "goal-profile": "goal_profiles",
        "goal_profiles": "goal_profiles",
        "open_discovery": "goal_profiles",
    }
    if mode not in aliases:
        raise ValueError(
            "NNGPT_SFT_RL_PROMPT_MODE must be one of: sft_aligned, goal_profiles"
        )
    return aliases[mode]


def resolve_sft_save_rl_model() -> bool:
    return _env_flag("NNGPT_SFT_SAVE_RL_MODEL", SFT_SAVE_RL_MODEL)


def resolve_sft_temperature() -> float:
    return _env_float("NNGPT_SFT_TEMPERATURE", SFT_TEMPERATURE)


def resolve_sft_top_p() -> float:
    return _env_float("NNGPT_SFT_TOP_P", SFT_TOP_P)


def resolve_sft_top_k() -> int:
    return _env_int("NNGPT_SFT_TOP_K", SFT_TOP_K)


def resolve_sft_model_out() -> str:
    return _env_str("NNGPT_SFT_MODEL_OUT", SFT_MODEL_OUT)


def resolve_sft_log_dir() -> str:
    return _env_str("NNGPT_SFT_LOG_DIR", SFT_LOG_DIR)


def resolve_sft_epoch_root() -> str:
    return _env_str("NNGPT_SFT_EPOCH_ROOT", SFT_EPOCH_ROOT)


def resolve_sft_trainer_out() -> str:
    return _env_str("NNGPT_SFT_TRAINER_OUT", SFT_TRAINER_OUT)


def resolve_sft_num_epochs() -> int:
    return _env_int("NNGPT_SFT_NUM_EPOCHS", SFT_NUM_EPOCHS)


def resolve_sft_save_steps() -> int:
    return max(1, _env_int("NNGPT_SFT_SAVE_STEPS", SFT_SAVE_STEPS))


def resolve_sft_save_total_limit() -> int:
    return max(1, _env_int("NNGPT_SFT_SAVE_TOTAL_LIMIT", 4))


def _resolve_sft_resume_spec() -> Dict[str, Any]:
    resume_spec = TrainingRuntime.resolve_resume_spec(
        trainer_env="NNGPT_SFT_RESUME_TRAINER_CHECKPOINT",
        stage_env="NNGPT_SFT_RESUME_STAGE_CHECKPOINT",
        initial_adapter_active=resolve_sft_load_initial_adapter(),
        initial_adapter_label="NNGPT_SFT_LOAD_INITIAL_ADAPTER",
        legacy_state_filenames=("reward_state.json",),
    )
    return {
        "mode": resume_spec.mode,
        "trainer_checkpoint": resume_spec.trainer_checkpoint,
        "stage_checkpoint_dir": resume_spec.stage_checkpoint_dir,
        "stage_adapter_dir": resume_spec.stage_adapter_dir,
        "active": resume_spec.active,
    }


def _sft_grpo_signature_parameters() -> Dict[str, inspect.Parameter]:
    return dict(inspect.signature(TuneRL.GRPOConfig.__init__).parameters)


def _sft_trainer_checkpoint_supported() -> bool:
    signature_parameters = _sft_grpo_signature_parameters()
    return "save_strategy" in signature_parameters and "save_steps" in signature_parameters


def _runtime_is_main_process(runtime: Dict[str, Any]) -> bool:
    return int(runtime.get("rank", 0)) == 0


def _run_git_command(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _sft_git_metadata() -> Dict[str, Any]:
    return {
        "commit": _run_git_command("rev-parse", "HEAD"),
        "commit_short": _run_git_command("rev-parse", "--short", "HEAD"),
        "commit_subject": _run_git_command("log", "-1", "--pretty=%s"),
        "branch": _run_git_command("branch", "--show-current"),
        "dirty": bool(_run_git_command("status", "--porcelain")),
    }


def _selected_env(names: List[str]) -> Dict[str, str]:
    return {name: os.environ[name] for name in names if os.getenv(name) not in (None, "")}


def _write_sft_run_config(
    *,
    runtime: Dict[str, Any],
    runtime_settings: Dict[str, Any],
    generation_kwargs: Dict[str, Any],
    resume_spec: Dict[str, Any],
    gpu_role_plan: Optional[Dict[str, Any]],
    reward_worker_plan: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: Optional[str],
    restored_runtime_state_path: Optional[Any],
) -> None:
    source_info = dict(SFT_RUNTIME_SOURCE_INFO)
    reward_env_names = [
        TuneRL.REWARD_VARIANT_ENV,
        "NNGPT_RL_FORMAL_REWARD_EPOCHS",
        "NNGPT_RL_FORMAL_DATASET",
        "NNGPT_RL_RESUME_STAGE",
        "NNGPT_RL_STAGE1_ONLY",
        "NNGPT_RL_SEED",
        "NNGPT_RL_KL_COEF",
        "NNGPT_RL_FORMAL_EVAL_LIMIT_SECONDS",
        "NNGPT_RL_FORMAL_EPOCH_LIMIT_MINUTES",
    ]
    sampling_env_names = [
        "NNGPT_SFT_TEMPERATURE",
        "NNGPT_SFT_TOP_P",
        "NNGPT_SFT_TOP_K",
        "NNGPT_SFT_NUM_GENERATIONS",
        "NNGPT_SFT_MAX_STEPS",
        "NNGPT_SFT_GENERATION_BATCH_SIZE",
        "NNGPT_SFT_STEPS_PER_GENERATION",
        "NNGPT_SFT_MAX_PROMPT_LENGTH",
        "NNGPT_SFT_MAX_COMPLETION_LENGTH",
        "NNGPT_SFT_GENERATION_KWARGS_JSON",
        "NNGPT_SFT_STOP_AFTER_FORWARD_XML",
    ]
    adapter_env_names = [
        "NNGPT_SFT_BASE_MODEL_ID",
        "NNGPT_SFT_TOKENIZER_ID",
        "NNGPT_SFT_LOAD_INITIAL_ADAPTER",
        "NNGPT_SFT_INITIAL_ADAPTER_MODE",
        "NNGPT_SFT_INITIAL_ADAPTER_DTYPE",
        "NNGPT_SFT_INIT_ADAPTER",
        "NNGPT_SFT_RESUME_TRAINER_CHECKPOINT",
        "NNGPT_SFT_RESUME_STAGE_CHECKPOINT",
    ]
    archive_env_names = [
        "NNGPT_SFT_LOG_DIR",
        "NNGPT_SFT_MODEL_OUT",
        "NNGPT_SFT_TRAINER_OUT",
        "NNGPT_SFT_EPOCH_ROOT",
        "NNGPT_SFT_APPEND_LOGS",
    ]
    split_protocol = SFT_EVAL_SPLIT_PROTOCOL
    split_seed = _env_int("NNGPT_SFT_EVAL_SPLIT_SEED", SFT_EVAL_SPLIT_SEED)
    eval_split_role = _env_str("NNGPT_SFT_EVAL_SPLIT_ROLE", SFT_EVAL_SPLIT_ROLE)
    formal_dataset = resolve_sft_formal_dataset()
    formal_out_shape = resolve_sft_formal_out_shape(formal_dataset)
    train_set_label, reward_eval_label, heldout_test_label = _describe_sft_eval_split(
        formal_dataset,
        split_protocol,
    )
    payload = {
        "phase": "four_pattern_reward_ablation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _sft_git_metadata(),
        "model_source": source_info.get("model_source", TuneRL.base_model),
        "tokenizer_source": source_info.get("tokenizer_source", getattr(TuneRL, "tokenizer_source", TuneRL.base_model)),
        "source_mode": source_info.get("source_mode", ""),
        "reward": {
            "variant": TuneRL.resolve_reward_variant(),
            "env": _selected_env(reward_env_names),
            "formal_epochs": os.getenv("NNGPT_RL_FORMAL_REWARD_EPOCHS", SFT_FORMAL_REWARD_EPOCHS),
        },
        "init": {
            "load_initial_adapter": resolve_sft_load_initial_adapter(),
            "initial_adapter_mode": resolve_sft_initial_adapter_mode(),
            "initial_adapter_dtype": resolve_sft_initial_adapter_dtype_label(),
            "init_adapter": resolve_sft_init_adapter(),
            "resume": resume_spec,
            "restored_runtime_state_path": str(restored_runtime_state_path) if restored_runtime_state_path else None,
            "adapter_env": _selected_env(adapter_env_names),
        },
        "sampling": {
            "seed": resolve_sft_rl_seed(),
            "temperature": resolve_sft_temperature(),
            "top_p": resolve_sft_top_p(),
            "top_k": resolve_sft_top_k(),
            "runtime_settings": runtime_settings,
            "generation_kwargs": generation_kwargs,
            "env": _selected_env(sampling_env_names),
        },
        "prompt": {
            "nn_prefixes": list(resolve_sft_rl_nn_prefixes()),
            "prompt_mode": resolve_sft_rl_prompt_mode(),
            "feedback_char_budget": _env_int("NNGPT_SFT_FEEDBACK_CHAR_BUDGET", SFT_FEEDBACK_CHAR_BUDGET),
        },
        "archive": {
            "fresh": not bool(resume_spec.get("active")),
            "append_logs": _env_flag("NNGPT_SFT_APPEND_LOGS", False),
            "log_dir": resolve_sft_log_dir(),
            "model_out": resolve_sft_model_out(),
            "trainer_out": resolve_sft_trainer_out(),
            "epoch_root": resolve_sft_epoch_root(),
            "env": _selected_env(archive_env_names),
        },
        "evaluator": {
            "dataset": formal_dataset,
            "out_shape": list(formal_out_shape),
            "n_classes": int(formal_out_shape[0]),
            "transform": SFT_EVAL_TRANSFORM,
            "resize": SFT_EVAL_IMAGE_SIZE,
            "batch": SFT_EVAL_BATCH_SIZE,
            "split_protocol": split_protocol,
            "split_seed": split_seed,
            "eval_split_role": eval_split_role,
            "train_subset_size": SFT_EVAL_TRAIN_SUBSET,
            "val_subset_size": SFT_EVAL_VAL_SUBSET,
            "train_set": train_set_label,
            "reward_eval_set": reward_eval_label,
            "heldout_test_set": heldout_test_label,
            "train_epochs": SFT_EVAL_TRAIN_EPOCHS,
            "formal_epochs": os.getenv("NNGPT_RL_FORMAL_REWARD_EPOCHS", SFT_FORMAL_REWARD_EPOCHS),
            "full_test_acc": SFT_EVAL_FULL_TEST_ACC,
            "run_unfrozen": SFT_EVAL_RUN_UNFROZEN,
            "worker_eval_limit_seconds": SFT_EVAL_LIMIT_SECONDS,
            "formal_epoch_limit_minutes": SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
            "data_root": SFT_EVAL_DATA_ROOT,
            "download": SFT_EVAL_DOWNLOAD,
        },
        "runtime": {
            "distributed": runtime,
            "gpu_role_plan": gpu_role_plan,
            "reward_worker_plan": reward_worker_plan,
            "use_deepspeed": use_deepspeed,
            "deepspeed_config_path": deepspeed_config_path,
        },
    }
    log_dir = Path(resolve_sft_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _resolved_visible_cuda_device_tokens(visible_cuda_devices: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        raw = raw.strip()
        if raw in {"", "-1"}:
            return []
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if tokens:
            return tokens
    return [str(index) for index in range(max(0, int(visible_cuda_devices)))]


def _resolve_sft_mode(visible_gpu_tokens: List[str]) -> Dict[str, str]:
    role_plan = TrainingRuntime.resolve_role_plan(
        visible_gpu_tokens=visible_gpu_tokens,
        requested_mode_env="NNGPT_SFT_MODE",
        default_mode=SFT_MODE_DEFAULT,
    )
    return {
        "requested_mode": str(role_plan.requested_mode),
        "resolved_mode": str(role_plan.resolved_mode),
    }


def resolve_sft_reward_exclude_train_gpu() -> bool:
    if "NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU" not in os.environ and "NNGPT_RL_REWARD_EXCLUDE_TRAIN_GPU" in os.environ:
        return _env_flag("NNGPT_RL_REWARD_EXCLUDE_TRAIN_GPU", SFT_REWARD_EXCLUDE_TRAIN_GPU)
    return _env_flag("NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU", SFT_REWARD_EXCLUDE_TRAIN_GPU)


def resolve_sft_compact_after_model_load() -> bool:
    return _env_flag("NNGPT_SFT_COMPACT_AFTER_MODEL_LOAD", SFT_COMPACT_AFTER_MODEL_LOAD)


def resolve_sft_compact_float_params() -> bool:
    return _env_flag("NNGPT_SFT_COMPACT_FLOAT_PARAMS", SFT_COMPACT_FLOAT_PARAMS)


def _sft_reward_gpu_tokens_for_plan(
    *,
    resolved_mode: str,
    visible_gpu_tokens: List[str],
    train_gpu_tokens: List[str],
    default_reward_gpu_tokens: List[str],
) -> List[str]:
    if resolved_mode != "split" or not resolve_sft_reward_exclude_train_gpu():
        return list(default_reward_gpu_tokens)
    reward_gpu_tokens = [token for token in visible_gpu_tokens if token not in set(train_gpu_tokens)]
    if not reward_gpu_tokens:
        raise RuntimeError(
            "NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1 requires at least one non-training GPU "
            f"in split mode; visible_gpu_tokens={visible_gpu_tokens} train_gpu_tokens={train_gpu_tokens}"
        )
    return reward_gpu_tokens


def _configure_sft_gpu_role_env(visible_cuda_devices: int) -> Dict[str, Any]:
    visible_gpu_tokens = _resolved_visible_cuda_device_tokens(visible_cuda_devices)
    role_plan = TrainingRuntime.resolve_role_plan(
        visible_gpu_tokens=visible_gpu_tokens,
        requested_mode_env="NNGPT_SFT_MODE",
        default_mode=SFT_MODE_DEFAULT,
    )
    train_gpu_tokens = list(role_plan.train_gpu_tokens)
    reward_gpu_tokens = _sft_reward_gpu_tokens_for_plan(
        resolved_mode=str(role_plan.resolved_mode),
        visible_gpu_tokens=list(role_plan.visible_gpu_tokens),
        train_gpu_tokens=train_gpu_tokens,
        default_reward_gpu_tokens=list(role_plan.aux_gpu_tokens),
    )
    if train_gpu_tokens:
        os.environ[_TRAIN_GPU_TOKENS_ENV] = ",".join(train_gpu_tokens)
    else:
        os.environ.pop(_TRAIN_GPU_TOKENS_ENV, None)
    if reward_gpu_tokens:
        os.environ[_AUX_GPU_TOKENS_ENV] = ",".join(reward_gpu_tokens)
        os.environ[_REWARD_GPU_TOKENS_ENV] = ",".join(reward_gpu_tokens)
    else:
        os.environ.pop(_AUX_GPU_TOKENS_ENV, None)
        os.environ.pop(_REWARD_GPU_TOKENS_ENV, None)
    return {
        "requested_mode": str(role_plan.requested_mode),
        "resolved_mode": str(role_plan.resolved_mode),
        "visible_gpu_tokens": list(role_plan.visible_gpu_tokens),
        "train_gpu_tokens": list(train_gpu_tokens),
        "reward_gpu_tokens": list(reward_gpu_tokens),
        "reward_exclude_train_gpu": resolve_sft_reward_exclude_train_gpu(),
    }


def _suggested_sft_worker_count(runtime: Dict[str, Any]) -> int:
    _ = runtime
    return 1


def _validate_sft_visible_worker_count(runtime: Dict[str, Any]) -> None:
    world_size = int(runtime.get("world_size", 1))
    if world_size == 1:
        return
    raise RuntimeError(
        "SFT RL runs with a single training rank so reward workers can use assigned GPUs: "
        f"world_size={world_size}, "
        f"visible_cuda_devices={int(runtime.get('visible_gpu_count', 0))}, "
        f"suggested_nproc_per_node={_suggested_sft_worker_count(runtime)}. "
        "Launch without torchrun or use --nproc_per_node=1."
    )


def _resolve_sft_local_rank(runtime: Dict[str, Any]) -> Dict[str, Any]:
    visible_cuda_devices = int(runtime.get("visible_gpu_count", 0))
    rank = int(runtime.get("rank", 0))
    world_size = int(runtime.get("world_size", 1))

    if visible_cuda_devices < 1:
        raise RuntimeError(
            f"SFT RL requires at least one visible CUDA device, got {visible_cuda_devices}"
        )
    local_rank = 0

    return {
        "rank": rank,
        "world_size": world_size,
        "raw_local_rank": int(runtime.get("raw_local_rank", 0)),
        "local_rank": local_rank,
        "visible_cuda_devices": visible_cuda_devices,
        "resolution": "single_process_training",
    }


def _maybe_relaunch_sft_with_visible_gpu_workers() -> None:
    if os.getenv("NNGPT_SFT_AUTO_TORCHRUN_DONE", "") == "1":
        return
    if os.getenv("WORLD_SIZE") not in (None, "", "1"):
        return
    if os.getenv("LOCAL_RANK") not in (None, ""):
        return

    import torch

    if not torch.cuda.is_available():
        return
    visible_cuda_devices = int(torch.cuda.device_count())
    _configure_sft_gpu_role_env(visible_cuda_devices)
    if visible_cuda_devices <= 1:
        return

    master_port = _resolve_sft_master_port()
    os.environ["NNGPT_SFT_AUTO_TORCHRUN_DONE"] = "1"
    print(
        "[SFT RL] Relaunching with single training worker: "
        f"nproc_per_node=1 master_addr={os.environ['MASTER_ADDR']} master_port={master_port}"
    )
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            f"--master_addr={os.environ['MASTER_ADDR']}",
            f"--master_port={master_port}",
            "-m",
            "ab.gpt.TuneRLSft",
        ],
        os.environ,
    )


def _maybe_init_single_process_deepspeed_group(
    *,
    use_deepspeed: bool,
    world_size: int,
    visible_cuda_devices: int,
    local_rank: int,
) -> None:
    import torch

    if not use_deepspeed:
        return
    if int(world_size) > 1:
        return
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_dir = Path(tempfile.mkdtemp(prefix="nngpt_sft_pg_"))
    init_file = (init_dir / "store").resolve()
    init_file.touch()
    init_method = init_file.as_uri()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=1,
        rank=0,
    )
    print(
        "[SFT RL] Initialized single-process distributed group for DeepSpeed: "
        f"backend={backend} local_rank={local_rank} visible_cuda_devices={visible_cuda_devices}"
    )


def resolve_sft_runtime_settings(runtime: Dict[str, Any]) -> Dict[str, int]:
    grad_accum = _env_int("NNGPT_SFT_GRAD_ACCUM", SFT_GRAD_ACCUM)
    generation_plan = TuneRL.resolve_generation_plan(
        runtime,
        env_name="NNGPT_SFT_NUM_GENERATIONS",
        default=SFT_NUM_GENERATIONS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
    )
    return {
        "dataset_limit": _env_int(
            "NNGPT_SFT_DATASET_LIMIT",
            SFT_DATASET_LIMIT,
        ),
        "grad_accum": grad_accum,
        "max_prompt_length": _env_int(
            "NNGPT_SFT_MAX_PROMPT_LENGTH",
            SFT_MAX_PROMPT_LENGTH,
        ),
        "max_completion_length": _env_int(
            "NNGPT_SFT_MAX_COMPLETION_LENGTH",
            SFT_MAX_COMPLETION_LENGTH,
        ),
        "effective_train_batch_size": generation_plan["effective_train_batch_size"],
        "requested_global_num_generations": generation_plan["requested_global_num_generations"],
        "global_num_generations": generation_plan["global_num_generations"],
        "effective_global_num_generations": generation_plan["effective_global_num_generations"],
        "global_num_generations_adapted": generation_plan["global_num_generations_adapted"],
        "valid_generation_values": generation_plan["valid_generation_values"],
        "generation_batch_size": _env_int("NNGPT_SFT_GENERATION_BATCH_SIZE", SFT_GENERATION_BATCH_SIZE),
        "steps_per_generation": _env_optional_int("NNGPT_SFT_STEPS_PER_GENERATION"),
        "max_steps": _env_int("NNGPT_SFT_MAX_STEPS", SFT_MAX_STEPS),
    }


def _resolve_sft_deepspeed_enabled(runtime: Dict[str, Any]) -> bool:
    raw = os.getenv("NNGPT_SFT_USE_DEEPSPEED")
    if raw is None or raw == "":
        return int(runtime.get("world_size", 1)) > 1
    return _env_flag("NNGPT_SFT_USE_DEEPSPEED", False)


def _resolve_sft_deepspeed_config_path() -> str:
    raw_path = os.getenv("NNGPT_SFT_DEEPSPEED_CONFIG", SFT_DEEPSPEED_DEFAULT_CONFIG)
    config_path = Path(raw_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"SFT DeepSpeed config not found: {config_path}")
    return str(config_path)


def _maybe_init_hf_deepspeed_config(config_path: str) -> Any:
    last_error: Exception | None = None
    for module_name in ("transformers.integrations", "transformers.deepspeed"):
        try:
            module = __import__(module_name, fromlist=["HfDeepSpeedConfig"])
            config_cls = getattr(module, "HfDeepSpeedConfig", None)
            if config_cls is not None:
                return config_cls(config_path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "DeepSpeed ZeRO-3 requested for SFT GRPO, but HfDeepSpeedConfig could not be imported"
    ) from last_error


def _bootstrap_run_token(runtime: Dict[str, Any] | None = None) -> str:
    runtime = runtime or {}
    for value in (
        os.getenv("TORCHELASTIC_RUN_ID"),
        os.getenv("MASTER_PORT"),
        os.getenv("SLURM_JOB_ID"),
    ):
        if value:
            return str(value)
    return f"rank0_world{int(runtime.get('world_size', 1))}"


def _resolve_sft_master_port() -> str:
    master_port = str(os.getenv("MASTER_PORT", "")).strip()
    if master_port:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        return master_port

    slurm_job_id = str(os.getenv("SLURM_JOB_ID", "")).strip()
    if slurm_job_id.isdigit():
        master_port = str(20000 + (int(slurm_job_id) % 20000))
    else:
        master_port = str(20000 + (os.getpid() % 20000))
    os.environ["MASTER_PORT"] = master_port
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    return master_port


def _bootstrap_sentinel_path(log_dir: str, runtime: Dict[str, Any] | None = None) -> Path:
    return Path(log_dir) / f".sft_bootstrap_complete.{_bootstrap_run_token(runtime)}"


def _wait_for_bootstrap_sentinel(
    sentinel_path: Path,
    *,
    timeout_seconds: float = 600.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if sentinel_path.exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rank0 bootstrap sentinel: {sentinel_path}")


def _cifar_eval_error(error: Exception, *, seed_accuracy_baseline: float | None = None) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_msg = f"{error_type}: {error}"
    return {
        "reward": 0.0,
        "components": {
            "reward": 0.0,
            "r_build": 0.0,
            "r_forward_shape": 0.0,
            "r_backward": 0.0,
            "r_loss_drop": 0.0,
            "r_forward": 0.0,
            "r_trainstep": 0.0,
            "r_metric": 0.0,
            "r_eff": 0.0,
            "r_critic": 0.0,
            "r_kl": 0.0,
        },
        "test_acc": None,
        "val_metric": None,
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
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
        "reward_batch_index": None,
        "reward_group_id": None,
        "group_warmup": False,
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": [],
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "reward_target_metric": "frozen_test_acc",
        "reward_target_value": None,
        "error": error_msg,
    }


def _torch_hub_checkpoints_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints"


def _print_runtime_cache_roots() -> None:
    print(f"[SFT RL] HF_HOME={os.environ.get('HF_HOME', '')!r}")
    print(f"[SFT RL] TORCH_HOME={os.environ.get('TORCH_HOME', '')!r}")
    print(f"[SFT RL] Torch hub checkpoints dir: {_torch_hub_checkpoints_dir()}")


def _configured_device_tokens(env_name: str) -> List[str]:
    raw = os.getenv(env_name, "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def _validate_configured_gpu_role_tokens(runtime: Dict[str, Any]) -> Dict[str, Any]:
    visible_tokens = _resolved_visible_cuda_device_tokens(int(runtime.get("visible_gpu_count", 0)))
    mode_info = _resolve_sft_mode(visible_tokens)
    resolved_mode = str(mode_info["resolved_mode"])
    train_tokens = _configured_device_tokens(_TRAIN_GPU_TOKENS_ENV)
    reward_tokens = _configured_device_tokens(_REWARD_GPU_TOKENS_ENV)
    expected_train_tokens = visible_tokens[:1]
    default_reward_tokens = list(visible_tokens) if resolved_mode == "split" else list(expected_train_tokens)
    expected_reward_tokens = _sft_reward_gpu_tokens_for_plan(
        resolved_mode=resolved_mode,
        visible_gpu_tokens=visible_tokens,
        train_gpu_tokens=expected_train_tokens,
        default_reward_gpu_tokens=default_reward_tokens,
    )
    if train_tokens != expected_train_tokens:
        raise RuntimeError(
            "Invalid configured training GPU tokens for SFT mode: "
            f"resolved_mode={resolved_mode} train_tokens={train_tokens} expected={expected_train_tokens}"
        )
    if reward_tokens != expected_reward_tokens:
        raise RuntimeError(
            "Invalid configured reward GPU tokens for SFT mode: "
            f"resolved_mode={resolved_mode} reward_tokens={reward_tokens} expected={expected_reward_tokens}"
        )
    return {
        "requested_mode": str(mode_info["requested_mode"]),
        "resolved_mode": resolved_mode,
        "visible_tokens": visible_tokens,
        "train_tokens": train_tokens,
        "reward_tokens": reward_tokens,
        "reward_exclude_train_gpu": resolve_sft_reward_exclude_train_gpu(),
    }


def _validate_gpu_reward_worker_bindings(warmup_diagnostics: Dict[str, Any]) -> None:
    workers = list(warmup_diagnostics.get("workers", []) or [])
    failures: List[str] = []
    for worker in workers:
        if worker.get("assigned_gpu") is None:
            continue
        cuda_visible_devices = str(worker.get("cuda_visible_devices", "")).strip()
        if int(worker.get("cuda_device_count", 0) or 0) != 1:
            failures.append(
                f"slot={worker.get('slot')} expected cuda_device_count=1 got {worker.get('cuda_device_count')!r}"
            )
        if str(worker.get("worker_device", "")) != "cuda:0":
            failures.append(
                f"slot={worker.get('slot')} expected worker_device='cuda:0' got {worker.get('worker_device')!r}"
            )
        if not cuda_visible_devices:
            failures.append(
                f"slot={worker.get('slot')} expected non-empty CUDA_VISIBLE_DEVICES in worker handshake"
            )
        if not bool(worker.get("physical_binding_verified", False)):
            failures.append(
                f"slot={worker.get('slot')} physical_binding_verified=False"
            )
    if failures:
        raise RuntimeError(
            "SFT reward worker GPU binding hard validation failed: " + "; ".join(failures)
        )


def evaluate_code_and_reward_cifar(
    code: str,
    *,
    stage_name=None,
    in_shape=(1, 3, 256, 256),
    out_shape=None,
    prm=None,
    device: str = "cpu",
    val_metric_baseline=None,
    seed_accuracy_baseline=None,
    cfg=None,
    reward_batch_index: int | None = None,
    completion_index: int | None = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    try:
        import torch

        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        if prm is None:
            prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
        defaults = {
            "lr": 1e-2,
            "momentum": 0.9,
            "batch": SFT_EVAL_BATCH_SIZE,
            "epoch": 1,
            "transform": SFT_EVAL_TRANSFORM,
        }
        prm = {**defaults, **prm}
        cfg = build_sft_reward_eval_cfg(
            stage_name=stage_name,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            cfg=cfg,
            device=eval_device,
        )
        effective_out_shape = (int(cfg.n_classes),)

        return RewardUtil.evaluate_code_and_reward(
            code,
            in_shape=in_shape,
            out_shape=effective_out_shape,
            prm=prm,
            device=eval_device,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )
    except Exception as exc:
        return _cifar_eval_error(exc, seed_accuracy_baseline=seed_accuracy_baseline)


def build_sft_reward_eval_cfg(
    *,
    stage_name=None,
    in_shape=(1, 3, 256, 256),
    out_shape=None,
    prm=None,
    cfg=None,
    device=None,
    **_unused_kwargs,
):
    import torch

    del _unused_kwargs

    eval_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
    defaults = {
        "lr": 1e-2,
        "momentum": 0.9,
        "batch": SFT_EVAL_BATCH_SIZE,
        "epoch": 1,
        "transform": SFT_EVAL_TRANSFORM,
    }
    prm = {**defaults, **prm}
    effective_stage_name = str(stage_name or TuneRL.current_stage_name)
    formal_dataset = normalize_sft_formal_dataset(getattr(cfg, "formal_dataset", None))
    effective_out_shape = resolve_sft_formal_out_shape(formal_dataset) if out_shape is None else tuple(out_shape)
    if tuple(effective_out_shape) == (10,) and formal_dataset != "cifar-10":
        effective_out_shape = resolve_sft_formal_out_shape(formal_dataset)

    if cfg is None:
        cfg = TuneRL.build_stage_eval_cfg(
            stage_name=effective_stage_name,
            in_shape=tuple(in_shape),
            out_shape=tuple(effective_out_shape),
            prm=prm,
            device=eval_device,
        )

    effective_n_classes = resolve_sft_formal_n_classes(formal_dataset)
    return RewardUtil.EvalConfig(
        device=eval_device,
        input_shape=tuple(getattr(cfg, "input_shape", in_shape)),
        n_classes=effective_n_classes,
        train_epochs=int(prm.get("epoch", getattr(cfg, "train_epochs", SFT_EVAL_TRAIN_EPOCHS)) or SFT_EVAL_TRAIN_EPOCHS),
        train_steps=getattr(cfg, "train_steps", None),
        max_val_batches=SFT_EVAL_VAL_BATCHES,
        default_batch_size=SFT_EVAL_BATCH_SIZE,
        train_subset_size=SFT_EVAL_TRAIN_SUBSET,
        val_subset_size=SFT_EVAL_VAL_SUBSET,
        data_root=SFT_EVAL_DATA_ROOT,
        download=SFT_EVAL_DOWNLOAD,
        split_protocol=SFT_EVAL_SPLIT_PROTOCOL,
        split_seed=_env_int("NNGPT_SFT_EVAL_SPLIT_SEED", SFT_EVAL_SPLIT_SEED),
        eval_split_role=_env_str("NNGPT_SFT_EVAL_SPLIT_ROLE", SFT_EVAL_SPLIT_ROLE),
        measure_latency=getattr(cfg, "measure_latency", True),
        kl_div=getattr(cfg, "kl_div", None),
        critic_fn=getattr(cfg, "critic_fn", None),
        weights=getattr(cfg, "weights", None),
        eval_limit_seconds=getattr(cfg, "eval_limit_seconds", SFT_EVAL_LIMIT_SECONDS),
        budget_probe_batches=getattr(cfg, "budget_probe_batches", None),
        run_unfrozen_backbone_eval=False,
        full_test_acc=SFT_EVAL_FULL_TEST_ACC,
        reward_target_metric=getattr(cfg, "reward_target_metric", "frozen_test_acc"),
        formal_nn_eval=getattr(cfg, "formal_nn_eval", False),
        static_only=getattr(cfg, "static_only", False),
        formal_task=getattr(cfg, "formal_task", "img-classification"),
        formal_dataset=formal_dataset,
        formal_metric=getattr(cfg, "formal_metric", "acc"),
        formal_epoch_limit_minutes=getattr(
            cfg,
            "formal_epoch_limit_minutes",
            SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
        ),
    )


def _is_trainable_architecture(res: Dict[str, Any], graph_info) -> bool:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    return bool(
        parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and res.get("backward_ok")
    )


def _reapply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    stage_name = str(res.get("current_stage_name") or TuneRL.current_stage_name)
    if (
        stage_name == TuneRL.STAGE1_STRUCTURE_EXPLORE
        and bool(res.get("static_only") or res.get("stage_uses_static_only"))
    ):
        reward_value = TuneRL._apply_executability_clamp(res, reward_value, graph_info)
        if bool(res.get("forward_shape_ok")):
            return max(reward_value, 0.05)
        return reward_value

    if not parse_ok:
        reward_value = min(reward_value, -0.25)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        reward_value = min(reward_value, -0.8 + build_partial)
    elif not res.get("forward_ok"):
        reward_value = min(reward_value, -0.40)
    elif not res.get("forward_shape_ok"):
        reward_value = min(reward_value, -0.30)
    elif not res.get("backward_ok"):
        reward_value = min(reward_value, -0.10)
    return reward_value


def sft_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Dict[str, Any] | None = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    batch_block_signatures: List[str] = None,
    batch_backbone_block_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    prompt_target_pattern: str = "",
    archive_snapshot_family_counts: Dict[str, int] = None,
    archive_snapshot_descriptor_counts: Dict[str, int] = None,
    archive_snapshot_backbone_signature_counts: Dict[str, int] = None,
    archive_snapshot_cnn_signature_counts: Dict[str, int] = None,
    archive_snapshot_graph_counts: Dict[str, int] = None,
    archive_snapshot_block_signature_counts: Dict[str, int] = None,
    archive_snapshot_backbone_cnn_pair_counts: Dict[str, int] = None,
    archive_snapshot_backbone_block_pair_counts: Dict[str, int] = None,
    archive_snapshot_backbone_block_best_quality: Dict[str, float] = None,
    group_baseline_train_acc: float | None = None,
    group_baseline_reward_target_acc: float | None = None,
    reward_batch_index: int | None = None,
    reward_group_id: int | None = None,
    group_warmup: bool = False,
    completion_index: int | None = None,
    batch_last_item: bool = False,
):
    res = raw_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        batch_block_signatures=batch_block_signatures,
        batch_backbone_block_signatures=batch_backbone_block_signatures,
        prompt_goal_tags=prompt_goal_tags,
        prompt_target_pattern=prompt_target_pattern,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_graph_counts=archive_snapshot_graph_counts,
        archive_snapshot_block_signature_counts=archive_snapshot_block_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        archive_snapshot_backbone_block_pair_counts=archive_snapshot_backbone_block_pair_counts,
        archive_snapshot_backbone_block_best_quality=archive_snapshot_backbone_block_best_quality,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )
    res["reward"] = _reapply_trainability_clamp(res, float(res.get("reward", -2.0)), graph_info)
    res["anti_collapse"] = {
        "goal_key": TuneRL.primary_goal_key(prompt_goal_tags, prompt_target_pattern),
        "trainable_ok": _is_trainable_architecture(res, graph_info),
        "anti_collapse_delta": 0.0,
    }
    return res


SFT_DISCOVERY_PROMPT_TEMPLATE = SFTUtil.open_discovery_rl_prompt_template


class DynamicSFTPromptDataset(TorchDataset):
    column_names = [
        "prompt",
        "accuracy",
        "goal_name",
        "target_tags",
        "goal_profile_id",
        "target_pattern",
        "prompt_mode",
    ]

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        *,
        block_signature: str,
        init_signature: str,
        forward_signature: str,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.block_signature = block_signature
        self.init_signature = init_signature
        self.forward_signature = forward_signature

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices) -> "DynamicSFTPromptDataset":
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        return DynamicSFTPromptDataset(
            [self.rows[int(index)] for index in indices],
            self.tokenizer,
            block_signature=self.block_signature,
            init_signature=self.init_signature,
            forward_signature=self.forward_signature,
        )

    def _render_prompt(self, row: Dict[str, Any]) -> str:
        prompt_mode = str(row.get("prompt_mode") or "goal_profiles")
        if prompt_mode == "sft_aligned":
            user_prompt = SFTUtil.format_backbone_prompt(
                accuracy=row["accuracy"],
                target_pattern=str(row["target_pattern"]),
            )
        else:
            profile = SFTUtil.open_discovery_goal_profiles[int(row["goal_profile_id"])]
            target_pattern = SFTUtil.goal_profile_target_pattern(profile)
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = SFT_DISCOVERY_PROMPT_TEMPLATE.format(
                accuracy=row["accuracy"],
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=", ".join(SFTUtil.legacy_patterns),
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                target_pattern=target_pattern,
                design_brief=profile["brief"],
                tag_realization=profile.get("realization", profile["brief"]),
                goal_tag_parser_cues=SFTUtil.goal_tag_parser_cues(profile["tags"]),
                module_hints=", ".join(module_hints),
                block_signature=self.block_signature,
                init_signature=self.init_signature,
                forward_signature=self.forward_signature,
            )
        feedback_char_budget = _env_int("NNGPT_SFT_FEEDBACK_CHAR_BUDGET", SFT_FEEDBACK_CHAR_BUDGET)
        if feedback_char_budget > 0:
            feedback_text = TuneRL.render_prompt_feedback_text(
                feedback_char_budget=feedback_char_budget,
            )
            feedback_section = "\n\n### Current Optimization Feedback\n" + feedback_text.strip() + "\n"
            contract_heading = "\n### Completion Contract\n"
            if contract_heading not in user_prompt:
                contract_heading = "\n### Contract\n"
            if contract_heading not in user_prompt:
                raise RuntimeError("SFT RL prompt template is missing a contract heading")
            user_prompt = user_prompt.replace(
                contract_heading,
                feedback_section + contract_heading,
                1,
            )
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        return {
            "prompt": self._render_prompt(row),
            "accuracy": row["accuracy"],
            "goal_name": row["goal_name"],
            "target_tags": row["target_tags"],
            "goal_profile_id": row["goal_profile_id"],
            "target_pattern": row["target_pattern"],
            "prompt_mode": row["prompt_mode"],
        }


def load_rl_dataset_sft(tokenizer) -> TuneRL.Dataset:
    """Load SFT-aligned RL prompts while rendering feedback lazily at access time."""
    runtime_settings = resolve_sft_runtime_settings(RewardUtil.get_distributed_runtime_info())
    nn_prefixes = resolve_sft_rl_nn_prefixes()
    prompt_mode = resolve_sft_rl_prompt_mode()
    data = TuneRL.api.data(task="img-classification", nn_prefixes=nn_prefixes)
    if data.empty:
        raise RuntimeError(f"No data found for SFT RL prefixes {nn_prefixes}; sync the dataset prefix before training.")

    print(f"Loaded {len(data)} examples for SFT RL prefixes={nn_prefixes} prompt_mode={prompt_mode}")
    TuneRL.bootstrap_trainset_reference_library(data)

    rows: List[Dict[str, Any]] = []

    if prompt_mode == "sft_aligned":
        for row_index, row in data.iterrows():
            full_code = row.get("nn_code")
            target_pattern = SFTUtil.extract_target_pattern_from_code(full_code) if isinstance(full_code, str) else None
            if not target_pattern:
                print(f"Skipping row {row_index} due to missing target_pattern")
                continue
            accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
            rows.append(
                {
                    "accuracy": accuracy,
                    "goal_name": str(target_pattern),
                    "target_tags": "",
                    "goal_profile_id": -1,
                    "target_pattern": str(target_pattern),
                    "prompt_mode": "sft_aligned",
                }
            )
    else:
        for _, row in data.iterrows():
            accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
            for profile_id, profile in enumerate(SFTUtil.open_discovery_goal_profiles):
                rows.append(
                    {
                        "accuracy": accuracy,
                        "goal_name": profile["name"],
                        "target_tags": ", ".join(profile["tags"]),
                        "goal_profile_id": profile_id,
                        "target_pattern": SFTUtil.goal_profile_target_pattern(profile),
                        "prompt_mode": "goal_profiles",
                    }
                )

    if not rows:
        raise RuntimeError(f"No SFT RL prompt rows built for prefixes={nn_prefixes} prompt_mode={prompt_mode}")

    random.Random(resolve_sft_rl_seed()).shuffle(rows)
    if len(rows) > runtime_settings["dataset_limit"]:
        rows = rows[:runtime_settings["dataset_limit"]]
    return DynamicSFTPromptDataset(
        rows,
        tokenizer,
        block_signature=BLOCK_SIGNATURE,
        init_signature=INIT_SIGNATURE,
        forward_signature=FORWARD_SIGNATURE,
    )


def sft_run_epoch_dir(*args) -> Path:
    epoch_dir = Path(resolve_sft_epoch_root())
    for value in args:
        epoch_dir = epoch_dir / f"A{value}"
    return epoch_dir


def _resolve_sft_generation_kwargs(tokenizer) -> Dict[str, Any]:
    kwargs = dict(_env_optional_json("NNGPT_SFT_GENERATION_KWARGS_JSON") or {})
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        kwargs.setdefault("eos_token_id", eos_token_id)
        kwargs.setdefault("pad_token_id", eos_token_id)
    kwargs.setdefault("use_cache", True)
    if _env_flag("NNGPT_SFT_STOP_AFTER_FORWARD_XML", True):
        kwargs.setdefault("stop_strings", ["</forward>"])
    return kwargs


def _attach_sft_generate_tokenizer(model, tokenizer) -> None:
    original_generate = model.generate

    def generate(*args, **kwargs):
        generation_config = kwargs.get("generation_config")
        if generation_config is not None and getattr(generation_config, "stop_strings", None):
            kwargs["tokenizer"] = tokenizer
        return original_generate(*args, **kwargs)

    model.generate = generate


def _build_sft_grpo_config(
    *,
    precision: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: str | None,
    runtime_settings: Dict[str, int],
    generation_kwargs: Dict[str, Any],
) -> Any:
    signature_parameters = _sft_grpo_signature_parameters()
    config_kwargs: Dict[str, Any] = {
        "temperature": resolve_sft_temperature(),
        "learning_rate": TuneRL.env_float("NNGPT_RL_LR", SFT_LR),
        "max_prompt_length": runtime_settings["max_prompt_length"],
        "max_completion_length": runtime_settings["max_completion_length"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": runtime_settings["grad_accum"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": resolve_sft_num_epochs(),
        "remove_unused_columns": False,
        "logging_steps": 1,
        "output_dir": resolve_sft_trainer_out(),
        "eval_strategy": "no",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": True,
        "num_generations": runtime_settings["global_num_generations"],
    }
    if "gradient_checkpointing_kwargs" in signature_parameters:
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    seed = resolve_sft_rl_seed()
    if "seed" in signature_parameters:
        config_kwargs["seed"] = seed
    if "data_seed" in signature_parameters:
        config_kwargs["data_seed"] = seed
    if "save_strategy" in signature_parameters:
        config_kwargs["save_strategy"] = "steps"
    if "save_steps" in signature_parameters:
        config_kwargs["save_steps"] = resolve_sft_save_steps()
    if "save_total_limit" in signature_parameters:
        config_kwargs["save_total_limit"] = resolve_sft_save_total_limit()
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "generation_batch_size",
        runtime_settings.get("generation_batch_size"),
    )
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "steps_per_generation",
        runtime_settings.get("steps_per_generation"),
    )
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "max_steps",
        runtime_settings.get("max_steps"),
    )
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "top_p",
        resolve_sft_top_p(),
    )
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "top_k",
        resolve_sft_top_k(),
    )
    use_vllm = _env_flag("NNGPT_SFT_USE_VLLM", False)
    if use_vllm:
        _set_optional_grpo_config(
            config_kwargs,
            signature_parameters,
            "use_vllm",
            True,
        )
        _set_optional_grpo_config(
            config_kwargs,
            signature_parameters,
            "vllm_mode",
            _env_str("NNGPT_SFT_VLLM_MODE", "colocate"),
        )
        _set_optional_grpo_config(
            config_kwargs,
            signature_parameters,
            "vllm_gpu_memory_utilization",
            TuneRL.env_float("NNGPT_SFT_VLLM_GPU_MEMORY_UTILIZATION", 0.25),
        )
        _set_optional_grpo_config(
            config_kwargs,
            signature_parameters,
            "vllm_tensor_parallel_size",
            _env_int("NNGPT_SFT_VLLM_TENSOR_PARALLEL_SIZE", 1),
        )
        _set_optional_grpo_config(
            config_kwargs,
            signature_parameters,
            "vllm_enable_sleep_mode",
            _env_flag("NNGPT_SFT_VLLM_ENABLE_SLEEP_MODE", True),
        )
    _set_optional_grpo_config(
        config_kwargs,
        signature_parameters,
        "generation_kwargs",
        generation_kwargs,
    )
    explicit_kl_coef = TuneRL.env_float("NNGPT_RL_KL_COEF", SFT_KL_COEF)
    if "beta" in signature_parameters:
        config_kwargs["beta"] = explicit_kl_coef
    elif "kl_coef" in signature_parameters:
        config_kwargs["kl_coef"] = explicit_kl_coef
    else:
        raise RuntimeError("Installed GRPOConfig does not expose `beta` or `kl_coef`; cannot set explicit KL control")
    if use_deepspeed:
        if "deepspeed" not in signature_parameters:
            raise RuntimeError("Installed GRPOConfig does not support the `deepspeed` argument")
        config_kwargs["deepspeed"] = deepspeed_config_path
        if "ds3_gather_for_generation" in signature_parameters:
            config_kwargs["ds3_gather_for_generation"] = False
    return TuneRL.GRPOConfig(**config_kwargs)


def run_sft_training(*, gpu_role_plan: Optional[Dict[str, Any]] = None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SFT RL requires CUDA for GRPO training, but no CUDA device is available")
    resume_spec = _resolve_sft_resume_spec()
    load_initial_adapter = resolve_sft_load_initial_adapter()
    initial_adapter_mode = resolve_sft_initial_adapter_mode()
    init_adapter_path = resolve_sft_init_adapter()
    save_rl_model = resolve_sft_save_rl_model()
    model_out_path = resolve_sft_model_out()
    trainer_checkpoint_supported = _sft_trainer_checkpoint_supported()
    if resume_spec["mode"] == "trainer":
        train_signature = inspect.signature(TuneRL.GRPOTrainer.train)
        if "resume_from_checkpoint" not in train_signature.parameters:
            raise RuntimeError("Installed GRPOTrainer does not support trainer resume_from_checkpoint.")
        if not trainer_checkpoint_supported:
            raise RuntimeError(
                "Installed GRPOConfig does not support save_strategy/save_steps, so trainer checkpoint resume is unavailable."
            )
    runtime = RewardUtil.get_distributed_runtime_info()
    runtime_settings = resolve_sft_runtime_settings(runtime)
    _validate_sft_visible_worker_count(runtime)
    local_rank_resolution = _resolve_sft_local_rank(runtime)
    visible_cuda_devices = int(local_rank_resolution["visible_cuda_devices"])
    local_rank = int(local_rank_resolution["local_rank"])
    raw_local_rank = int(local_rank_resolution["raw_local_rank"])
    rank = int(local_rank_resolution["rank"])
    world_size = int(local_rank_resolution["world_size"])
    use_deepspeed = _resolve_sft_deepspeed_enabled(runtime)
    deepspeed_config_path = _resolve_sft_deepspeed_config_path() if use_deepspeed else None
    os.environ["NNGPT_SFT_USE_DEEPSPEED"] = "1" if use_deepspeed else "0"
    if deepspeed_config_path is not None:
        os.environ["NNGPT_SFT_DEEPSPEED_CONFIG"] = deepspeed_config_path
    torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}"
    _maybe_init_single_process_deepspeed_group(
        use_deepspeed=use_deepspeed,
        world_size=world_size,
        visible_cuda_devices=visible_cuda_devices,
        local_rank=local_rank,
    )

    torch.cuda.empty_cache()
    trainer_checkpoint = resume_spec["trainer_checkpoint"]
    stage_checkpoint_dir = resume_spec["stage_checkpoint_dir"]
    stage_adapter_dir = resume_spec["stage_adapter_dir"]
    resume_state_dir = trainer_checkpoint if trainer_checkpoint is not None else stage_checkpoint_dir
    resume_stage_override = _env_str("NNGPT_RL_RESUME_STAGE", SFT_RL_RESUME_STAGE).strip()
    # Restore pipeline runtime bookkeeping the same way for trainer and stage checkpoints.
    restored_runtime_state_path = TrainingRuntime.restore_or_reset_runtime_state(
        resume_state_dir,
        _sft_runtime_state_hooks(),
        legacy_state_filenames=("reward_state.json",),
    )
    if resume_stage_override:
        TuneRL.apply_resume_stage_override(resume_stage_override, log_prefix="[SFT RL]")
    precision = TuneRL.best_mixed_precision()
    tokenizer_source = getattr(TuneRL, "tokenizer_source", TuneRL.base_model)
    if tokenizer_source != TuneRL.base_model:
        print(f"Using RL tokenizer: {tokenizer_source}")
    tokenizer = TrainerRuntime.load_tokenizer(tokenizer_source)
    generation_kwargs = _resolve_sft_generation_kwargs(tokenizer)
    initial_adapter_dtype_label = resolve_sft_initial_adapter_dtype_label()
    if load_initial_adapter and initial_adapter_mode != "trainable" and initial_adapter_dtype_label != "fp32":
        raise ValueError("NNGPT_SFT_INITIAL_ADAPTER_DTYPE only applies to trainable initial adapters.")
    initial_adapter_dtype = resolve_sft_initial_adapter_dtype(initial_adapter_dtype_label, precision)
    grpo_config = _build_sft_grpo_config(
        precision=precision,
        use_deepspeed=use_deepspeed,
        deepspeed_config_path=deepspeed_config_path,
        runtime_settings=runtime_settings,
        generation_kwargs=generation_kwargs,
    )
    hf_deepspeed_config = _maybe_init_hf_deepspeed_config(deepspeed_config_path) if use_deepspeed else None
    if not trainer_checkpoint_supported:
        print(
            "[SFT RL] Warning: installed GRPOConfig does not support save_strategy/save_steps; "
            "trainer checkpoints will not be produced."
        )
    if restored_runtime_state_path is not None:
        print(f"[SFT RL] Restored runtime state from {restored_runtime_state_path}")

    print(f"Using RL base model: {TuneRL.base_model}")
    print(
        "[SFT RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    if use_deepspeed and world_size <= 1:
        print(
            "[SFT RL] DeepSpeed single-process fallback active: "
            f"visible_cuda_devices={visible_cuda_devices} local_rank={local_rank}"
        )
    print(f"[SFT RL] DeepSpeed Enabled: {use_deepspeed}")
    if deepspeed_config_path is not None:
        print(f"[SFT RL] DeepSpeed Config: {deepspeed_config_path}")
    print(f"[SFT RL] Fixed training device: {train_device}")
    print(f"[SFT RL] Visible CUDA devices: {visible_cuda_devices}")
    print(f"[SFT RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    if load_initial_adapter and initial_adapter_mode == "trainable":
        print(
            "[SFT RL] Initial adapter dtype: "
            f"{initial_adapter_dtype_label} (torch_dtype={initial_adapter_dtype})"
        )
    print(f"[SFT RL] Current stage: {TuneRL.current_stage_name}")
    print(
        "[SFT RL] Runtime limits: "
        f"dataset_limit={runtime_settings['dataset_limit']} "
        f"max_prompt_length={runtime_settings['max_prompt_length']} "
        f"max_completion_length={runtime_settings['max_completion_length']} "
        f"grad_accum={runtime_settings['grad_accum']} "
        f"effective_train_batch_size={runtime_settings['effective_train_batch_size']} "
        f"requested_global_num_generations={runtime_settings['requested_global_num_generations']} "
        f"global_num_generations={runtime_settings['global_num_generations']} "
        f"effective_global_num_generations={runtime_settings['effective_global_num_generations']}"
    )
    print(
        "[SFT RL] Generation tuning: "
        f"max_steps={runtime_settings.get('max_steps')} "
        f"generation_batch_size={runtime_settings.get('generation_batch_size')} "
        f"steps_per_generation={runtime_settings.get('steps_per_generation')} "
        f"use_vllm={_env_flag('NNGPT_SFT_USE_VLLM', False)} "
        f"vllm_mode={_env_str('NNGPT_SFT_VLLM_MODE', 'colocate') if _env_flag('NNGPT_SFT_USE_VLLM', False) else None}"
    )
    print(f"[SFT RL] Generation kwargs: {generation_kwargs}")
    if runtime_settings["global_num_generations_adapted"]:
        print(
            "[SFT RL] Generation plan adapted "
            f"requested={runtime_settings['requested_global_num_generations']} "
            f"effective={runtime_settings['effective_global_num_generations']} "
            f"valid_generation_values={runtime_settings['valid_generation_values']} "
            f"world_size={world_size}"
        )
    reward_worker_plan = RewardUtil.get_reward_worker_plan()
    mode_validation = _validate_configured_gpu_role_tokens(runtime)
    print(
        "[SFT RL] Reward Worker Plan: "
        f"mode={reward_worker_plan['mode']} "
        f"reward_workers_per_gpu={reward_worker_plan.get('workers_per_gpu', 1)} "
        f"per_gpu_worker_counts={reward_worker_plan.get('per_gpu_worker_counts', [])} "
        f"rank={reward_worker_plan['rank']} "
        f"local_rank={reward_worker_plan['local_rank']} "
        f"world_size={reward_worker_plan['world_size']} "
        f"train_gpu={reward_worker_plan['train_gpu']} "
        f"reward_gpu_indices={reward_worker_plan['reward_gpu_indices']} "
        f"reward_gpu_tokens={reward_worker_plan.get('reward_gpu_tokens', [])} "
        f"reason={reward_worker_plan.get('reason', '')!r}"
    )
    print(
        "[SFT RL] GPU role validation: "
        f"requested_mode={mode_validation['requested_mode']} "
        f"resolved_mode={mode_validation['resolved_mode']} "
        f"train_tokens={mode_validation['train_tokens']} "
        f"reward_tokens={mode_validation['reward_tokens']} "
        f"reward_exclude_train_gpu={mode_validation['reward_exclude_train_gpu']}"
    )
    if _runtime_is_main_process(runtime):
        _write_sft_run_config(
            runtime=runtime,
            runtime_settings=runtime_settings,
            generation_kwargs=generation_kwargs,
            resume_spec=resume_spec,
            gpu_role_plan=gpu_role_plan,
            reward_worker_plan=reward_worker_plan,
            use_deepspeed=use_deepspeed,
            deepspeed_config_path=deepspeed_config_path,
            restored_runtime_state_path=restored_runtime_state_path,
        )
        print(f"[SFT RL] Run config: {Path(resolve_sft_log_dir()) / 'run_config.json'}")
    _print_runtime_cache_roots()

    rl_dataset = TuneRL.load_rl_dataset(tokenizer)
    if len(rl_dataset) > runtime_settings["dataset_limit"]:
        rl_dataset = rl_dataset.select(range(runtime_settings["dataset_limit"]))

    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=TuneRL.base_model,
        precision=precision,
        train_device=train_device,
        use_deepspeed=use_deepspeed,
    )
    if resolve_sft_compact_after_model_load():
        TrainerRuntime.compact_cuda_cache(log_prefix="[SFT RL]", stage="after_base_model_load")
    _ = hf_deepspeed_config

    if load_initial_adapter and initial_adapter_mode == "merge" and stage_adapter_dir is None:
        model = TrainerRuntime.maybe_merge_initial_adapter(
            model,
            enabled=True,
            adapter_path=init_adapter_path,
            label="SFT",
            empty_adapter_message="SFT_INIT_ADAPTER is empty, but SFT_LOAD_INITIAL_ADAPTER is True.",
            missing_adapter_message=f"Initial adapter not found: {init_adapter_path}",
            load_message=f"Loading initial SFT adapter from {init_adapter_path} for merge...",
        )

    model = TuneRL.prepare_model_for_kbit_training(model)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    if resolve_sft_compact_float_params():
        model = TrainerRuntime.cast_non_lora_float_parameter_dtype(
            model,
            dtype=precision["torch_dtype"],
            label="[SFT RL] Prepared model",
        )
    if resolve_sft_compact_after_model_load():
        TrainerRuntime.compact_cuda_cache(log_prefix="[SFT RL]", stage="after_prepare_kbit")

    if trainer_checkpoint is not None:
        model = TrainerRuntime.load_trainable_initial_adapter(
            model,
            enabled=True,
            adapter_path=str(trainer_checkpoint),
            label="trainer checkpoint",
            empty_adapter_message="SFT trainer checkpoint is empty.",
            missing_adapter_message=f"Missing SFT trainer checkpoint adapter: {trainer_checkpoint}",
            load_message=f"Loading trainable SFT trainer checkpoint adapter from {trainer_checkpoint}...",
        )
    elif stage_adapter_dir is not None:
        peft_config = TrainerRuntime.build_lora_config(
            r=SFT_LORA_R,
            alpha=SFT_LORA_ALPHA,
            dropout=SFT_LORA_DROPOUT,
        )
        model = TrainerRuntime.attach_or_resume_lora(
            model,
            peft_config=peft_config,
            stage_adapter_dir=stage_adapter_dir,
            log_prefix="[SFT RL]",
            missing_adapter_message=f"Missing adapter directory under SFT stage checkpoint: {stage_adapter_dir}",
        )
    elif load_initial_adapter and initial_adapter_mode == "trainable":
        model = TrainerRuntime.load_trainable_initial_adapter(
            model,
            enabled=True,
            adapter_path=init_adapter_path,
            label="SFT",
            adapter_dtype=initial_adapter_dtype,
            empty_adapter_message="SFT_INIT_ADAPTER is empty, but SFT_LOAD_INITIAL_ADAPTER is True.",
            missing_adapter_message=f"Initial adapter not found: {init_adapter_path}",
            load_message=f"Loading trainable initial SFT adapter from {init_adapter_path}...",
        )
    else:
        peft_config = TrainerRuntime.build_lora_config(
            r=SFT_LORA_R,
            alpha=SFT_LORA_ALPHA,
            dropout=SFT_LORA_DROPOUT,
        )
        model = TrainerRuntime.attach_or_resume_lora(
            model,
            peft_config=peft_config,
            stage_adapter_dir=None,
            log_prefix="[SFT RL]",
            missing_adapter_message=f"Missing adapter directory under SFT stage checkpoint: {stage_adapter_dir}",
        )
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    if resolve_sft_compact_float_params():
        model = TrainerRuntime.cast_non_lora_float_parameter_dtype(
            model,
            dtype=precision["torch_dtype"],
            label="[SFT RL] PEFT model",
        )
    if resolve_sft_compact_after_model_load():
        TrainerRuntime.compact_cuda_cache(log_prefix="[SFT RL]", stage="after_adapter_load")

    TrainerRuntime.enable_non_reentrant_gradient_checkpointing(
        model,
        log_prefix="[SFT RL]",
    )
    if resolve_sft_compact_after_model_load():
        TrainerRuntime.compact_cuda_cache(log_prefix="[SFT RL]", stage="after_gradient_checkpointing")
    model.print_trainable_parameters()
    TuneRL.active_rl_model = model
    TuneRL.active_rl_tokenizer = tokenizer

    trainer_kwargs = {
        "model": model,
        "train_dataset": rl_dataset,
        "reward_funcs": TuneRL.compute_reward,
        "args": grpo_config,
    }
    if "processing_class" in inspect.signature(TuneRL.GRPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = TuneRL.GRPOTrainer(**trainer_kwargs)
    _attach_sft_generate_tokenizer(model, tokenizer)
    print("[SFT RL] Stop-string generation tokenizer attached")
    trainer_gc_patch_stats = TrainerRuntime.enforce_non_reentrant_gradient_checkpointing(trainer.model)
    print(
        "[SFT RL] Trainer gradient checkpointing enforcement: "
        f"roots={trainer_gc_patch_stats['roots']} modules={trainer_gc_patch_stats['modules']} use_reentrant=False"
    )
    if trainer_checkpoint_supported:
        # Keep legacy reward_state.json in trainer checkpoints so old resumes still work.
        runtime_callback = TrainingRuntime.build_trainer_checkpoint_callback(
            _sft_runtime_state_hooks(),
            state_aliases=("reward_state.json",),
        )
        if runtime_callback is not None:
            trainer.add_callback(runtime_callback)
    warmup_diagnostics = RewardUtil.prewarm_eval_workers(timeout_seconds=60.0, require_gpu=True)
    _validate_gpu_reward_worker_bindings(warmup_diagnostics)
    if resolve_sft_compact_after_model_load():
        TrainerRuntime.compact_cuda_cache(log_prefix="[SFT RL]", stage="before_grpo_train")
    TuneRL.register_stage_checkpoint_signal_handlers()

    print("Starting GRPO training for Backbone Search...")
    try:
        TrainerRuntime.train_grpo(
            trainer=trainer,
            trainer_checkpoint=trainer_checkpoint,
            log_prefix="[SFT RL]",
        )
    except Exception as exc:
        if TuneRL.is_cuda_oom_error(exc):
            TuneRL.log_cuda_oom_diagnostics("sft/trainer.train", exc)
        raise
    finally:
        RewardUtil.shutdown_eval_worker()

    if save_rl_model:
        print(f"Saving model to {model_out_path}...")
        model.save_pretrained(model_out_path)
        print("Model saved successfully!")
    else:
        print("[SFT RL] Skipping RL adapter save. Next run will start from the same initial model.")
    if _runtime_is_main_process(runtime):
        TuneRL._save_stage_checkpoint(
            "completed",
            stage_name=TuneRL.current_stage_name,
            reason="sft_trainer_completed",
        )

    return model


def patch_sft_runtime() -> tuple[str, str, str]:
    """Patch TuneRL to use the SFT runtime and CIFAR-aware reward."""
    global SFT_RUNTIME_SOURCE_INFO
    model_source, tokenizer_source, source_mode = resolve_sft_model_sources()
    SFT_RUNTIME_SOURCE_INFO = {
        "model_source": model_source,
        "tokenizer_source": tokenizer_source,
        "source_mode": source_mode,
    }
    load_initial_adapter = resolve_sft_load_initial_adapter()
    init_adapter_path = resolve_sft_init_adapter()
    TuneRL.base_model = model_source
    TuneRL.tokenizer_source = tokenizer_source
    TuneRL.LOAD_EXISTING_MODEL = load_initial_adapter
    TuneRL.SAVED_MODEL_PATH = init_adapter_path if load_initial_adapter else ""
    TuneRL.PROMPT_TEMPLATE = SFT_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = extract_completion_blocks_tolerant
    TuneRL.clear_extraction_meta_cache = clear_extraction_meta_cache
    TuneRL.evaluate_code_and_reward = evaluate_code_and_reward_cifar
    setattr(TuneRL.evaluate_code_and_reward, "_nngpt_eval_cfg_builder", build_sft_reward_eval_cfg)
    TuneRL.reward_fn = sft_reward_fn
    TuneRL.load_rl_dataset = load_rl_dataset_sft
    TuneRL.run_log_dir = resolve_sft_log_dir
    TuneRL.run_model_out = resolve_sft_model_out
    TuneRL.run_epoch_dir = sft_run_epoch_dir
    return model_source, tokenizer_source, source_mode


def bootstrap_sft_runtime() -> None:
    """Initialize logging and reset extraction cache."""
    clear_extraction_meta_cache()
    RewardUtil.shutdown_eval_worker()
    runtime = RewardUtil.get_distributed_runtime_info()
    is_main = _runtime_is_main_process(runtime)
    resume_spec = _resolve_sft_resume_spec()
    if resume_spec["active"]:
        os.environ["NNGPT_SFT_APPEND_LOGS"] = "1"
    else:
        os.environ.pop("NNGPT_SFT_APPEND_LOGS", None)

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    sentinel_path = _bootstrap_sentinel_path(log_dir, runtime)
    trainer_out_dir = Path(resolve_sft_trainer_out())
    stale_files = (
        "generation_samples.jsonl",
        "group_progress.jsonl",
        "group_feedback_samples.jsonl",
        "best_group_feedback.json",
    )
    if is_main:
        if sentinel_path.exists():
            sentinel_path.unlink()
        if resume_spec["active"]:
            print(
                f"[SFT RL] Resume mode active: preserving logs and trainer outputs "
                f"(mode={resume_spec['mode']}, log_dir={log_dir}, trainer_out={trainer_out_dir})"
            )
        else:
            for filename in stale_files:
                path = Path(log_dir) / filename
                if path.exists():
                    print(f"Removing stale runtime log: {path}")
                    path.unlink()
            print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
            shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)
            print(f"Cleaning existing trainer outputs in {trainer_out_dir}...")
            shutil.rmtree(trainer_out_dir, ignore_errors=True)
        TuneRL.code_logger = TuneRL.SimpleCodeLogger(log_dir)
        sentinel_path.write_text(str(os.getpid()), encoding="utf-8")
        return

    TuneRL.code_logger = TuneRL.NullCodeLogger()
    _wait_for_bootstrap_sentinel(sentinel_path)


def main() -> None:
    import torch

    TuneRL.apply_rl_seed(log_prefix="[SFT RL]")
    resume_spec = _resolve_sft_resume_spec()
    gpu_role_plan: Dict[str, Any] = {
        "requested_mode": "auto",
        "resolved_mode": "single",
        "visible_gpu_tokens": [],
        "train_gpu_tokens": [],
        "reward_gpu_tokens": [],
        "reward_exclude_train_gpu": False,
    }
    if torch.cuda.is_available():
        gpu_role_plan = _configure_sft_gpu_role_env(int(torch.cuda.device_count()))
        print(
            "[SFT RL] GPU mode plan: "
            f"requested_mode={gpu_role_plan['requested_mode']} "
            f"resolved_mode={gpu_role_plan['resolved_mode']} "
            f"visible_gpu_tokens={gpu_role_plan['visible_gpu_tokens']} "
            f"train_gpu_tokens={gpu_role_plan['train_gpu_tokens']} "
            f"reward_gpu_tokens={gpu_role_plan['reward_gpu_tokens']} "
            f"reward_exclude_train_gpu={gpu_role_plan['reward_exclude_train_gpu']}"
        )
    _maybe_relaunch_sft_with_visible_gpu_workers()
    _validate_sft_visible_worker_count(RewardUtil.get_distributed_runtime_info())
    model_source, tokenizer_source, source_mode = patch_sft_runtime()
    bootstrap_sft_runtime()

    print(f"[SFT RL] Base model id: {resolve_sft_base_model_id()}")
    print(f"[SFT RL] Base model source ({source_mode}): {model_source}")
    if tokenizer_source != model_source:
        print(f"[SFT RL] Tokenizer source: {tokenizer_source}")
    print(f"[SFT RL] Resume mode: {resume_spec['mode']}")
    print(f"[SFT RL] Load init adapter: {resolve_sft_load_initial_adapter()}")
    print(f"[SFT RL] Initial adapter mode: {resolve_sft_initial_adapter_mode()}")
    print(f"[SFT RL] Reward excludes train GPU: {resolve_sft_reward_exclude_train_gpu()}")
    print(f"[SFT RL] Compact after model load: {resolve_sft_compact_after_model_load()}")
    print(f"[SFT RL] Compact float params: {resolve_sft_compact_float_params()}")
    if resolve_sft_load_initial_adapter():
        print(f"[SFT RL] Init adapter path: {resolve_sft_init_adapter()}")
        print(f"[SFT RL] Initial adapter dtype request: {resolve_sft_initial_adapter_dtype_label()}")
    if resume_spec["trainer_checkpoint"] is not None:
        print(f"[SFT RL] Resume trainer checkpoint: {resume_spec['trainer_checkpoint']}")
    if resume_spec["stage_checkpoint_dir"] is not None:
        print(f"[SFT RL] Resume stage checkpoint: {resume_spec['stage_checkpoint_dir']}")
    print(f"[SFT RL] Log dir: {resolve_sft_log_dir()}")
    print(f"[SFT RL] Trainer out: {resolve_sft_trainer_out()}")
    print(f"[SFT RL] Model out: {resolve_sft_model_out()}")
    print(f"[SFT RL] Stage1 only: {TuneRL.env_flag('NNGPT_RL_STAGE1_ONLY', False)}")
    print(f"[SFT RL] Num epochs: {resolve_sft_num_epochs()}")
    print(f"[SFT RL] Temperature: {resolve_sft_temperature()}")
    print(f"[SFT RL] Top-p: {resolve_sft_top_p()}")
    print(f"[SFT RL] Top-k: {resolve_sft_top_k()}")
    print(f"[SFT RL] Seed: {resolve_sft_rl_seed()}")
    print(f"[SFT RL] KL coef: {TuneRL.env_float('NNGPT_RL_KL_COEF', SFT_KL_COEF):.6f}")
    formal_dataset = resolve_sft_formal_dataset()
    formal_out_shape = resolve_sft_formal_out_shape(formal_dataset)
    train_set_label, reward_eval_label, heldout_test_label = _describe_sft_eval_split(
        formal_dataset,
        SFT_EVAL_SPLIT_PROTOCOL,
    )
    print(
        f"[SFT RL] Eval plan: stage1=static_only(no-check_nn), stage2/3=nn-dataset-formal({formal_dataset}), "
        f"out_shape={formal_out_shape}, n_classes={formal_out_shape[0]}, "
        f"transform={SFT_EVAL_TRANSFORM}, resize={SFT_EVAL_IMAGE_SIZE}, batch={SFT_EVAL_BATCH_SIZE}, "
        f"split={SFT_EVAL_SPLIT_PROTOCOL}, "
        f"split_seed={_env_int('NNGPT_SFT_EVAL_SPLIT_SEED', SFT_EVAL_SPLIT_SEED)}, "
        f"eval_split_role={_env_str('NNGPT_SFT_EVAL_SPLIT_ROLE', SFT_EVAL_SPLIT_ROLE)}, "
        f"train_set={train_set_label}, "
        f"reward_eval_set={reward_eval_label}, heldout_test_set={heldout_test_label}, train_epochs={SFT_EVAL_TRAIN_EPOCHS}, "
        f"freeze_only_backbone_eval=True, formal_epoch_limit_minutes={SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES}, "
        f"worker_eval_limit_seconds={SFT_EVAL_LIMIT_SECONDS}, "
        f"baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(f"[SFT RL] Save RL adapter: {resolve_sft_save_rl_model()}")

    run_sft_training(gpu_role_plan=gpu_role_plan)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        try:
            import ab.gpt.TuneRL as _TuneRL

            logger = getattr(_TuneRL, "code_logger", None)
            if logger is not None:
                logger.log_to_file("[SFT RL] Fatal exception:\n" + traceback.format_exc())
        except Exception:
            pass
        raise

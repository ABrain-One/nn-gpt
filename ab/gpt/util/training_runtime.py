"""Shared runtime helpers for GPU role planning and checkpoint resume/save."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence


@dataclass(frozen=True)
class RolePlan:
    requested_mode: str
    resolved_mode: str
    visible_gpu_tokens: list[str]
    train_gpu_tokens: list[str]
    aux_gpu_tokens: list[str]


@dataclass(frozen=True)
class ResumeSpec:
    mode: str
    trainer_checkpoint: Optional[Path]
    stage_checkpoint_dir: Optional[Path]
    stage_adapter_dir: Optional[Path]
    active: bool


@dataclass(frozen=True)
class RuntimeStateHooks:
    capture: Optional[Callable[[], Dict[str, Any]]] = None
    restore: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None
    reset: Optional[Callable[[], None]] = None


def _env_str(name: Optional[str], default: str = "") -> str:
    if not name:
        return str(default)
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload or {})


def _dedup_names(primary_name: str, extra_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for name in (primary_name, *extra_names):
        resolved = str(name or "").strip()
        if not resolved or resolved in names:
            continue
        names.append(resolved)
    return names


def _normalize_state_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"runtime state payload must be a dict, got {type(payload).__name__}")
    return dict(payload)


def _existing_state_path(
    checkpoint_dir: Path,
    *,
    state_filename: str,
    legacy_state_filenames: Sequence[str],
) -> Optional[Path]:
    for filename in _dedup_names(state_filename, legacy_state_filenames):
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate
    return None


def resolve_role_plan(
    *,
    visible_gpu_tokens: Sequence[str],
    requested_mode: Optional[str] = None,
    requested_mode_env: Optional[str] = None,
    default_mode: str = "auto",
    train_gpu_count: int = 1,
    split_uses_all_visible: bool = True,
) -> RolePlan:
    """Resolve train/aux GPU roles once so pipelines share the same split rules."""
    visible_tokens = [str(token).strip() for token in visible_gpu_tokens if str(token).strip()]
    resolved_requested_mode = str(
        requested_mode if requested_mode is not None else _env_str(requested_mode_env, default_mode)
    ).strip().lower()
    if resolved_requested_mode == "":
        resolved_requested_mode = str(default_mode).strip().lower()
    if resolved_requested_mode not in {"auto", "split", "single"}:
        raise ValueError(
            f"Invalid runtime mode {resolved_requested_mode!r}. Expected one of: auto, split, single."
        )
    if not visible_tokens:
        raise RuntimeError("Training runtime requires at least one visible CUDA device.")

    if resolved_requested_mode == "auto":
        resolved_mode = "split" if len(visible_tokens) >= 2 else "single"
    else:
        resolved_mode = resolved_requested_mode

    resolved_train_gpu_count = max(1, int(train_gpu_count))
    train_gpu_tokens = list(visible_tokens[:resolved_train_gpu_count])
    if resolved_mode == "split" and split_uses_all_visible:
        aux_gpu_tokens = list(visible_tokens)
    else:
        aux_gpu_tokens = list(train_gpu_tokens)

    return RolePlan(
        requested_mode=resolved_requested_mode,
        resolved_mode=resolved_mode,
        visible_gpu_tokens=list(visible_tokens),
        train_gpu_tokens=train_gpu_tokens,
        aux_gpu_tokens=aux_gpu_tokens,
    )


def resolve_resume_spec(
    *,
    trainer_checkpoint: Optional[str] = None,
    stage_checkpoint_dir: Optional[str] = None,
    trainer_env: Optional[str] = None,
    stage_env: Optional[str] = None,
    initial_adapter_active: bool = False,
    initial_adapter_label: str = "initial adapter",
    stage_adapter_dirname: str = "adapter",
    state_filename: str = "runtime_state.json",
    legacy_state_filenames: Sequence[str] = (),
) -> ResumeSpec:
    """Normalize mutually exclusive trainer/stage resume inputs into one spec."""
    trainer_raw = str(
        trainer_checkpoint if trainer_checkpoint is not None else _env_str(trainer_env, "")
    ).strip()
    stage_raw = str(
        stage_checkpoint_dir if stage_checkpoint_dir is not None else _env_str(stage_env, "")
    ).strip()

    trainer_path = Path(trainer_raw).expanduser().resolve() if trainer_raw else None
    normalized_stage_dir = Path(stage_raw).expanduser().resolve() if stage_raw else None
    if (
        normalized_stage_dir is not None
        and normalized_stage_dir.name == stage_adapter_dirname
        and _existing_state_path(
            normalized_stage_dir.parent,
            state_filename=state_filename,
            legacy_state_filenames=legacy_state_filenames,
        )
        is not None
    ):
        normalized_stage_dir = normalized_stage_dir.parent

    if trainer_path is not None and normalized_stage_dir is not None:
        raise ValueError(
            f"{trainer_env or 'trainer_checkpoint'} and {stage_env or 'stage_checkpoint_dir'} are mutually exclusive."
        )
    if (trainer_path is not None or normalized_stage_dir is not None) and bool(initial_adapter_active):
        raise ValueError(
            f"Resume mode cannot be combined with {initial_adapter_label}. "
            "Use either a resume checkpoint or an initial adapter, not both."
        )

    mode = "fresh"
    if trainer_path is not None:
        mode = "trainer"
    elif normalized_stage_dir is not None:
        mode = "stage"

    stage_adapter_dir = None
    if normalized_stage_dir is not None:
        if normalized_stage_dir.name == stage_adapter_dirname:
            stage_adapter_dir = normalized_stage_dir
        else:
            stage_adapter_dir = normalized_stage_dir / stage_adapter_dirname

    return ResumeSpec(
        mode=mode,
        trainer_checkpoint=trainer_path,
        stage_checkpoint_dir=normalized_stage_dir,
        stage_adapter_dir=stage_adapter_dir,
        active=mode != "fresh",
    )


def restore_or_reset_runtime_state(
    resume_checkpoint_dir: Optional[Path],
    hooks: RuntimeStateHooks,
    *,
    state_filename: str = "runtime_state.json",
    legacy_state_filenames: Sequence[str] = (),
) -> Optional[Path]:
    """Restore pipeline-owned runtime state from a checkpoint or reset it for a fresh run."""
    if resume_checkpoint_dir is None:
        if hooks.reset is not None:
            hooks.reset()
        return None

    checkpoint_dir = Path(resume_checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")

    state_path = _existing_state_path(
        checkpoint_dir,
        state_filename=state_filename,
        legacy_state_filenames=legacy_state_filenames,
    )
    if state_path is None:
        expected = ", ".join(_dedup_names(state_filename, legacy_state_filenames))
        raise FileNotFoundError(f"Missing runtime state under {checkpoint_dir} (expected one of: {expected})")

    if hooks.restore is not None:
        hooks.restore(_load_json(state_path))
    return state_path


def build_trainer_checkpoint_callback(
    hooks: RuntimeStateHooks,
    *,
    state_filename: str = "runtime_state.json",
    state_aliases: Sequence[str] = (),
):
    """Write runtime state alongside trainer checkpoints when the pipeline provides capture hooks."""
    if hooks.capture is None:
        return None

    from transformers import TrainerCallback

    names_to_write = _dedup_names(state_filename, state_aliases)

    class _RuntimeStateCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            global_step = int(getattr(state, "global_step", 0) or 0)
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
            payload = _normalize_state_payload(hooks.capture())
            for filename in names_to_write:
                _write_json(checkpoint_dir / filename, payload)
            return control

    return _RuntimeStateCheckpointCallback()


def save_runtime_checkpoint(
    checkpoint_dir: Path,
    *,
    hooks: RuntimeStateHooks,
    manifest: Optional[Dict[str, Any]] = None,
    state_filename: str = "runtime_state.json",
    state_aliases: Sequence[str] = (),
    manifest_filename: str = "runtime_manifest.json",
    manifest_aliases: Sequence[str] = (),
) -> Optional[Path]:
    """Persist runtime state plus optional manifest for stage-style checkpoints."""
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_path = None
    if hooks.capture is not None:
        payload = _normalize_state_payload(hooks.capture())
        state_names = _dedup_names(state_filename, state_aliases)
        state_path = resolved_checkpoint_dir / state_names[0]
        for filename in state_names:
            _write_json(resolved_checkpoint_dir / filename, payload)

    if manifest is not None:
        manifest_payload = _normalize_state_payload(manifest)
        for filename in _dedup_names(manifest_filename, manifest_aliases):
            _write_json(resolved_checkpoint_dir / filename, manifest_payload)

    return state_path

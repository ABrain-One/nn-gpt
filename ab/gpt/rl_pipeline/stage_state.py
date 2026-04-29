from collections import Counter
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Type


def counter_payload(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def nested_counter_payload(mapping: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return {
        str(key): counter_payload(counter)
        for key, counter in mapping.items()
    }


def restore_counter(counter: Counter, payload: Optional[Dict[str, int]]) -> None:
    counter.clear()
    if payload:
        counter.update({str(key): int(value) for key, value in payload.items()})


def restore_nested_counters(target: Dict[str, Counter], payload: Optional[Dict[str, Dict[str, int]]]) -> None:
    target.clear()
    for key, counter_payload_item in (payload or {}).items():
        restored_counter = Counter()
        restored_counter.update({str(inner_key): int(value) for inner_key, value in (counter_payload_item or {}).items()})
        target[str(key)] = restored_counter


def copy_history_items(items: Optional[List[Dict[str, Any]]], *, limit: int) -> List[Dict[str, Any]]:
    copied = [dict(item) for item in list(items or [])]
    if limit > 0 and len(copied) > limit:
        copied = copied[-limit:]
    return copied


def float_dict_payload(mapping: Dict[str, float]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in mapping.items()
    }


def restore_float_dict(target: Dict[str, float], payload: Optional[Dict[str, float]]) -> None:
    target.clear()
    target.update(
        {
            str(key): float(value)
            for key, value in (payload or {}).items()
        }
    )


def dominant_counter_entry(counter_payload_item: Optional[Dict[str, int]]) -> Tuple[Optional[str], float]:
    if not counter_payload_item:
        return None, 0.0
    total = sum(int(value) for value in counter_payload_item.values())
    if total <= 0:
        return None, 0.0
    key, count = max(counter_payload_item.items(), key=lambda item: int(item[1]))
    return str(key), float(count) / float(total)


def _feedback_summaries_from_payload(
    items: Optional[List[Dict[str, Any]]],
    feedback_summary_cls: Type[Any],
) -> List[Any]:
    return [feedback_summary_cls(**dict(item)) for item in (items or [])]


def capture_reward_runtime_state(
    runtime: MutableMapping[str, Any],
    *,
    max_stage_sample_history: int,
    max_stage_group_history: int,
    feedback_summary_payload: Callable[[List[Any]], List[Dict[str, Any]]],
    current_group_top_feedback_payload: Callable[[], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "B_index": runtime["B_index"],
        "reward_batch_index": runtime["reward_batch_index"],
        "current_group_id": runtime["current_group_id"],
        "current_group_reward_target_sum": runtime["current_group_reward_target_sum"],
        "current_group_reward_target_count": runtime["current_group_reward_target_count"],
        "current_group_frozen_train_acc_sum": runtime["current_group_frozen_train_acc_sum"],
        "current_group_frozen_train_acc_count": runtime["current_group_frozen_train_acc_count"],
        "current_group_frozen_test_acc_sum": runtime["current_group_frozen_test_acc_sum"],
        "current_group_frozen_test_acc_count": runtime["current_group_frozen_test_acc_count"],
        "current_group_unfrozen_train_acc_sum": runtime["current_group_unfrozen_train_acc_sum"],
        "current_group_unfrozen_train_acc_count": runtime["current_group_unfrozen_train_acc_count"],
        "current_group_unfrozen_test_acc_sum": runtime["current_group_unfrozen_test_acc_sum"],
        "current_group_unfrozen_test_acc_count": runtime["current_group_unfrozen_test_acc_count"],
        "prev_closed_group_mean_reward_target_acc": runtime["prev_closed_group_mean_reward_target_acc"],
        "best_closed_group_mean_reward_target_acc": runtime["best_closed_group_mean_reward_target_acc"],
        "prev_closed_group_train_acc_mean": runtime["prev_closed_group_train_acc_mean"],
        "best_closed_group_mean_train_acc": runtime["best_closed_group_mean_train_acc"],
        "prev_closed_group_mean_test_acc": runtime["prev_closed_group_mean_test_acc"],
        "best_closed_group_mean_test_acc": runtime["best_closed_group_mean_test_acc"],
        "best_closed_group_id": runtime["best_closed_group_id"],
        "best_reward_target_by_goal": {
            str(key): float(value)
            for key, value in runtime["best_reward_target_by_goal"].items()
        },
        "dominant_family_hash": runtime["dominant_family_hash"],
        "dominant_family_share": runtime["dominant_family_share"],
        "dominant_descriptor_key": runtime["dominant_descriptor_key"],
        "dominant_descriptor_share": runtime["dominant_descriptor_share"],
        "dominant_backbone_signature": runtime["dominant_backbone_signature"],
        "dominant_backbone_share": runtime["dominant_backbone_share"],
        "dominant_cnn_signature": runtime["dominant_cnn_signature"],
        "dominant_cnn_share": runtime["dominant_cnn_share"],
        "dominant_backbone_cnn_pair": runtime["dominant_backbone_cnn_pair"],
        "dominant_backbone_cnn_share": runtime["dominant_backbone_cnn_share"],
        "graph_archive_counts": counter_payload(runtime["graph_archive_counts"]),
        "family_archive_counts": counter_payload(runtime["family_archive_counts"]),
        "family_hash_archive_counts": counter_payload(runtime["family_hash_archive_counts"]),
        "descriptor_archive_counts": counter_payload(runtime["descriptor_archive_counts"]),
        "backbone_signature_archive_counts": counter_payload(runtime["backbone_signature_archive_counts"]),
        "cnn_signature_archive_counts": counter_payload(runtime["cnn_signature_archive_counts"]),
        "backbone_cnn_pair_archive_counts": counter_payload(runtime["backbone_cnn_pair_archive_counts"]),
        "family_metric_best": {
            str(key): float(value)
            for key, value in runtime["family_metric_best"].items()
        },
        "motif_name_counts": counter_payload(runtime["motif_name_counts"]),
        "saved_graph_counts": counter_payload(runtime["saved_graph_counts"]),
        "saved_family_hash_counts": counter_payload(runtime["saved_family_hash_counts"]),
        "saved_backbone_signature_counts": counter_payload(runtime["saved_backbone_signature_counts"]),
        "saved_cnn_signature_counts": counter_payload(runtime["saved_cnn_signature_counts"]),
        "saved_backbone_cnn_pair_counts": counter_payload(runtime["saved_backbone_cnn_pair_counts"]),
        "goal_graph_archive_counts": nested_counter_payload(runtime["goal_graph_archive_counts"]),
        "goal_family_hash_archive_counts": nested_counter_payload(runtime["goal_family_hash_archive_counts"]),
        "saved_goal_family_hash_counts": nested_counter_payload(runtime["saved_goal_family_hash_counts"]),
        "current_group_reward_target_sum_by_backbone": float_dict_payload(runtime["current_group_reward_target_sum_by_backbone"]),
        "current_group_reward_target_count_by_backbone": counter_payload(runtime["current_group_reward_target_count_by_backbone"]),
        "prev_closed_group_mean_reward_target_by_backbone": float_dict_payload(runtime["prev_closed_group_mean_reward_target_by_backbone"]),
        "best_closed_group_mean_reward_target_by_backbone": float_dict_payload(runtime["best_closed_group_mean_reward_target_by_backbone"]),
        "saved_best_reward_target_by_backbone_cnn": float_dict_payload(runtime["saved_best_reward_target_by_backbone_cnn"]),
        "prev_group_feedback": feedback_summary_payload(runtime["prev_group_feedback"]),
        "best_group_feedback": feedback_summary_payload(runtime["best_group_feedback"]),
        "current_group_top_feedback": current_group_top_feedback_payload(),
        "current_group_goal_best_candidates": {
            str(key): float(value)
            for key, value in runtime["current_group_goal_best_candidates"].items()
        },
        "current_stage_name": runtime["current_stage_name"],
        "stage_closed_group_counts": counter_payload(runtime["stage_closed_group_counts"]),
        "stage_best_group_mean_reward_target": {
            str(key): float(value)
            for key, value in runtime["stage_best_group_mean_reward_target"].items()
        },
        "stage_entry_generation_totals": {
            str(key): int(value)
            for key, value in runtime["stage_entry_generation_totals"].items()
        },
        "stage_entry_reward_batches": {
            str(key): int(value)
            for key, value in runtime["stage_entry_reward_batches"].items()
        },
        "generation_history": copy_history_items(runtime["generation_history"], limit=max_stage_sample_history),
        "closed_group_history": copy_history_items(runtime["closed_group_history"], limit=max_stage_group_history),
        "stage_event_history": copy_history_items(runtime["stage_event_history"], limit=max_stage_group_history),
        "discovery_family_hashes_seen": sorted(str(item) for item in runtime["discovery_family_hashes_seen"]),
        "recovery_active": bool(runtime["recovery_active"]),
        "recovery_start_generation_total": int(runtime["recovery_start_generation_total"]),
        "recovery_start_discovery_family_count": int(runtime["recovery_start_discovery_family_count"]),
    }


def _runtime_scalar_defaults(stage1_structure_explore: str) -> Dict[str, Any]:
    return {
        "B_index": 0,
        "reward_batch_index": 0,
        "current_group_id": 0,
        "current_group_reward_target_sum": 0.0,
        "current_group_reward_target_count": 0,
        "current_group_frozen_train_acc_sum": 0.0,
        "current_group_frozen_train_acc_count": 0,
        "current_group_frozen_test_acc_sum": 0.0,
        "current_group_frozen_test_acc_count": 0,
        "current_group_unfrozen_train_acc_sum": 0.0,
        "current_group_unfrozen_train_acc_count": 0,
        "current_group_unfrozen_test_acc_sum": 0.0,
        "current_group_unfrozen_test_acc_count": 0,
        "prev_closed_group_mean_reward_target_acc": None,
        "best_closed_group_mean_reward_target_acc": None,
        "prev_closed_group_train_acc_mean": None,
        "best_closed_group_mean_train_acc": None,
        "prev_closed_group_mean_test_acc": None,
        "best_closed_group_mean_test_acc": None,
        "best_closed_group_id": None,
        "dominant_family_hash": None,
        "dominant_family_share": 0.0,
        "dominant_descriptor_key": None,
        "dominant_descriptor_share": 0.0,
        "dominant_backbone_signature": None,
        "dominant_backbone_share": 0.0,
        "dominant_cnn_signature": None,
        "dominant_cnn_share": 0.0,
        "dominant_backbone_cnn_pair": None,
        "dominant_backbone_cnn_share": 0.0,
        "current_stage_name": stage1_structure_explore,
        "recovery_active": False,
        "recovery_start_generation_total": 0,
        "recovery_start_discovery_family_count": 0,
    }


def restore_reward_runtime_state(
    runtime: MutableMapping[str, Any],
    state: Optional[Dict[str, Any]],
    *,
    max_stage_sample_history: int,
    max_stage_group_history: int,
    stage1_structure_explore: str,
    feedback_summary_cls: Type[Any],
) -> None:
    if not state:
        return
    for name, default_value in _runtime_scalar_defaults(stage1_structure_explore).items():
        runtime[name] = state.get(name, default_value)

    restore_counter(runtime["graph_archive_counts"], state.get("graph_archive_counts"))
    restore_counter(runtime["family_archive_counts"], state.get("family_archive_counts"))
    restore_counter(runtime["family_hash_archive_counts"], state.get("family_hash_archive_counts"))
    restore_counter(runtime["descriptor_archive_counts"], state.get("descriptor_archive_counts"))
    restore_counter(runtime["backbone_signature_archive_counts"], state.get("backbone_signature_archive_counts"))
    restore_counter(runtime["cnn_signature_archive_counts"], state.get("cnn_signature_archive_counts"))
    restore_counter(runtime["backbone_cnn_pair_archive_counts"], state.get("backbone_cnn_pair_archive_counts"))
    runtime["family_metric_best"].clear()
    runtime["family_metric_best"].update(
        {
            str(key): float(value)
            for key, value in (state.get("family_metric_best") or {}).items()
        }
    )
    restore_counter(runtime["motif_name_counts"], state.get("motif_name_counts"))
    restore_counter(runtime["saved_graph_counts"], state.get("saved_graph_counts"))
    restore_counter(runtime["saved_family_hash_counts"], state.get("saved_family_hash_counts"))
    restore_counter(runtime["saved_backbone_signature_counts"], state.get("saved_backbone_signature_counts"))
    restore_counter(runtime["saved_cnn_signature_counts"], state.get("saved_cnn_signature_counts"))
    restore_counter(runtime["saved_backbone_cnn_pair_counts"], state.get("saved_backbone_cnn_pair_counts"))
    restore_nested_counters(runtime["goal_graph_archive_counts"], state.get("goal_graph_archive_counts"))
    restore_nested_counters(runtime["goal_family_hash_archive_counts"], state.get("goal_family_hash_archive_counts"))
    restore_nested_counters(runtime["saved_goal_family_hash_counts"], state.get("saved_goal_family_hash_counts"))
    restore_float_dict(runtime["current_group_reward_target_sum_by_backbone"], state.get("current_group_reward_target_sum_by_backbone"))
    restore_counter(runtime["current_group_reward_target_count_by_backbone"], state.get("current_group_reward_target_count_by_backbone"))
    restore_float_dict(runtime["prev_closed_group_mean_reward_target_by_backbone"], state.get("prev_closed_group_mean_reward_target_by_backbone"))
    restore_float_dict(runtime["best_closed_group_mean_reward_target_by_backbone"], state.get("best_closed_group_mean_reward_target_by_backbone"))
    restore_float_dict(runtime["saved_best_reward_target_by_backbone_cnn"], state.get("saved_best_reward_target_by_backbone_cnn"))

    runtime["best_reward_target_by_goal"].clear()
    runtime["best_reward_target_by_goal"].update(
        {
            str(key): float(value)
            for key, value in (state.get("best_reward_target_by_goal") or {}).items()
        }
    )

    runtime["prev_group_feedback"][:] = _feedback_summaries_from_payload(state.get("prev_group_feedback"), feedback_summary_cls)
    runtime["best_group_feedback"][:] = _feedback_summaries_from_payload(state.get("best_group_feedback"), feedback_summary_cls)
    runtime["current_group_top_feedback"][:] = _feedback_summaries_from_payload(state.get("current_group_top_feedback"), feedback_summary_cls)
    runtime["current_group_goal_best_candidates"].clear()
    runtime["current_group_goal_best_candidates"].update(
        {
            str(key): float(value)
            for key, value in (state.get("current_group_goal_best_candidates") or {}).items()
        }
    )
    restore_counter(runtime["stage_closed_group_counts"], state.get("stage_closed_group_counts"))
    runtime["stage_best_group_mean_reward_target"].clear()
    runtime["stage_best_group_mean_reward_target"].update(
        {
            str(key): float(value)
            for key, value in (state.get("stage_best_group_mean_reward_target") or {}).items()
        }
    )
    runtime["stage_entry_generation_totals"].clear()
    runtime["stage_entry_generation_totals"].update(
        {
            str(key): int(value)
            for key, value in (state.get("stage_entry_generation_totals") or {}).items()
        }
    )
    runtime["stage_entry_reward_batches"].clear()
    runtime["stage_entry_reward_batches"].update(
        {
            str(key): int(value)
            for key, value in (state.get("stage_entry_reward_batches") or {}).items()
        }
    )
    runtime["generation_history"][:] = copy_history_items(state.get("generation_history"), limit=max_stage_sample_history)
    runtime["closed_group_history"][:] = copy_history_items(state.get("closed_group_history"), limit=max_stage_group_history)
    runtime["stage_event_history"][:] = copy_history_items(state.get("stage_event_history"), limit=max_stage_group_history)
    runtime["discovery_family_hashes_seen"].clear()
    runtime["discovery_family_hashes_seen"].update(str(item) for item in (state.get("discovery_family_hashes_seen") or []))


def reset_reward_runtime_state(
    runtime: MutableMapping[str, Any],
    *,
    stage1_structure_explore: str,
    reset_current_group_feedback_state: Callable[[], None],
) -> None:
    for name in (
        "graph_archive_counts",
        "family_archive_counts",
        "family_hash_archive_counts",
        "descriptor_archive_counts",
        "backbone_signature_archive_counts",
        "cnn_signature_archive_counts",
        "backbone_cnn_pair_archive_counts",
        "family_metric_best",
        "motif_name_counts",
        "saved_graph_counts",
        "saved_family_hash_counts",
        "saved_backbone_signature_counts",
        "saved_cnn_signature_counts",
        "saved_backbone_cnn_pair_counts",
        "goal_graph_archive_counts",
        "goal_family_hash_archive_counts",
        "saved_goal_family_hash_counts",
        "current_group_reward_target_sum_by_backbone",
        "current_group_reward_target_count_by_backbone",
        "prev_closed_group_mean_reward_target_by_backbone",
        "best_closed_group_mean_reward_target_by_backbone",
        "saved_best_reward_target_by_backbone_cnn",
        "stage_closed_group_counts",
        "stage_best_group_mean_reward_target",
        "stage_entry_generation_totals",
        "stage_entry_reward_batches",
        "generation_history",
        "closed_group_history",
        "stage_event_history",
        "discovery_family_hashes_seen",
    ):
        runtime[name].clear()

    for name, default_value in _runtime_scalar_defaults(stage1_structure_explore).items():
        runtime[name] = default_value

    runtime["best_reward_target_by_goal"].clear()
    runtime["prev_group_feedback"].clear()
    runtime["best_group_feedback"].clear()
    reset_current_group_feedback_state()


# TuneRL stage/group orchestration helpers.
def current_generation_total(rl):
    return len(rl.generation_history)


def current_reward_group_context(rl):
    return {'reward_batch_index': rl.reward_batch_index + 1, 'reward_group_id': rl.current_group_id, 'group_warmup': rl.current_group_id == 0, 'group_baseline_train_acc': rl.prev_closed_group_train_acc_mean, 'group_baseline_reward_target_acc': rl.prev_closed_group_mean_reward_target_acc, 'group_baseline_test_acc': rl.prev_closed_group_mean_test_acc, 'best_closed_group_mean_train_acc': rl.best_closed_group_mean_train_acc, 'best_closed_group_mean_reward_target_acc': rl.best_closed_group_mean_reward_target_acc, 'best_closed_group_mean_test_acc': rl.best_closed_group_mean_test_acc, 'best_closed_group_id': rl.best_closed_group_id, 'dominant_family_hash': rl.dominant_family_hash, 'dominant_family_share': rl.dominant_family_share, 'dominant_descriptor_key': rl.dominant_descriptor_key, 'dominant_descriptor_share': rl.dominant_descriptor_share, 'dominant_backbone_signature': rl.dominant_backbone_signature, 'dominant_backbone_share': rl.dominant_backbone_share, 'dominant_cnn_signature': rl.dominant_cnn_signature, 'dominant_cnn_share': rl.dominant_cnn_share, 'dominant_backbone_cnn_pair': rl.dominant_backbone_cnn_pair, 'dominant_backbone_cnn_share': rl.dominant_backbone_cnn_share, 'current_stage_name': rl.current_stage_name, 'current_stage_index': rl.RL_STAGE_TO_INDEX.get(rl.current_stage_name, 0), 'generation_total': rl._current_generation_total(), 'stage_group_count': len(rl._recent_stage_group_window(rl.current_stage_name, rl.MAX_STAGE_GROUP_HISTORY)), 'recovery_active': bool(rl.recovery_active)}


def stage_checkpoint_root(rl):
    root = rl.Path(rl.run_model_out()).expanduser().resolve()
    return root / 'checkpoints'


def stage_checkpoint_dir(rl, stage_name):
    return rl._stage_checkpoint_root() / str(stage_name)


def stage_group_snapshot_payload(rl, current_group_payload=None):
    return {'current_stage_name': rl.current_stage_name, 'current_stage_index': rl._current_stage_index(), 'generation_total': rl._current_generation_total(), 'reward_batch_index': rl.reward_batch_index, 'current_group_id': rl.current_group_id, 'stage_group_count': len(rl._recent_stage_group_window(rl.current_stage_name, rl.MAX_STAGE_GROUP_HISTORY)), 'recovery_active': bool(rl.recovery_active), 'recovery_start_generation_total': int(rl.recovery_start_generation_total), 'recovery_start_discovery_family_count': int(rl.recovery_start_discovery_family_count), 'latest_closed_group': dict(current_group_payload or {}), 'recent_stage_groups': rl._recent_stage_group_window(rl.current_stage_name, 12), 'recent_stage_generations': rl._recent_stage_generation_window(rl.current_stage_name, 64)}


def save_stage_plot_snapshot(rl, output_path):
    try:
        completed = rl.subprocess.run(['python3', str(rl.Path(rl.__file__).resolve().parent / 'plot_rl_reward.py'), '--log-dir', str(rl.Path(rl.run_log_dir()).expanduser().resolve()), '--output', str(output_path)], capture_output=True, text=True, timeout=180, check=False)
        if completed.returncode != 0:
            rl.code_logger.log_to_file(f'[Stage Checkpoint] plot snapshot failed rc={completed.returncode}: {completed.stderr.strip()}')
    except Exception as exc:
        rl.code_logger.log_to_file(f'[Stage Checkpoint] plot snapshot error: {type(exc).__name__}: {exc}')


def save_stage_checkpoint(rl, event, *, stage_name=None, group_progress_payload=None, reason=None, save_plot_snapshot=True):
    if not rl.is_main_process():
        return None
    resolved_stage = str(stage_name or rl.current_stage_name)
    checkpoint_dir = rl._stage_checkpoint_dir(resolved_stage)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / 'adapter'
    tokenizer_dir = checkpoint_dir / 'tokenizer'
    runtime_state_path = checkpoint_dir / 'runtime_state.json'
    runtime_manifest_path = checkpoint_dir / 'runtime_manifest.json'
    reward_state_path = checkpoint_dir / 'reward_state.json'
    manifest_path = checkpoint_dir / 'stage_manifest.json'
    snapshot_path = checkpoint_dir / 'group_progress_snapshot.json'
    plot_path = checkpoint_dir / 'plot_snapshot.png'
    if rl.active_rl_model is not None:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        rl.active_rl_model.save_pretrained(str(adapter_dir))
    if rl.active_rl_tokenizer is not None:
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        rl.active_rl_tokenizer.save_pretrained(str(tokenizer_dir))
    rl._write_json(snapshot_path, rl._stage_group_snapshot_payload(group_progress_payload))
    manifest = {'event': str(event), 'reason': reason, 'stage_name': resolved_stage, 'stage_index': rl.RL_STAGE_TO_INDEX.get(resolved_stage, 0), 'generation_total': rl._current_generation_total(), 'reward_batch_index': rl.reward_batch_index, 'current_group_id': rl.current_group_id, 'stage_group_count': len(rl._recent_stage_group_window(resolved_stage, rl.MAX_STAGE_GROUP_HISTORY)), 'reward_target_metric': rl._stage_reward_target_metric(resolved_stage), 'recovery_active': bool(rl.recovery_active), 'stage_best_group_mean_reward_target': rl.stage_best_group_mean_reward_target.get(resolved_stage), 'latest_group_progress': dict(group_progress_payload or {}), 'checkpoint_dir': str(checkpoint_dir), 'adapter_dir': str(adapter_dir), 'tokenizer_dir': str(tokenizer_dir), 'runtime_state_path': str(runtime_state_path), 'runtime_manifest_path': str(runtime_manifest_path), 'reward_state_path': str(reward_state_path), 'stage_manifest_path': str(manifest_path), 'plot_snapshot_path': str(plot_path)}
    rl.TrainingRuntime.save_runtime_checkpoint(checkpoint_dir, hooks=rl._reward_runtime_hooks(), manifest=manifest, state_aliases=('reward_state.json',), manifest_aliases=('stage_manifest.json',))
    if save_plot_snapshot:
        rl._save_stage_plot_snapshot(plot_path)
    return checkpoint_dir


def maybe_update_stage_best_checkpoint(rl, group_progress_payload):
    stage_name = str(group_progress_payload.get('stage_name') or rl.current_stage_name)
    closed_mean = rl._optional_float(group_progress_payload.get('closed_mean_reward_target_acc'))
    if closed_mean is None:
        return
    previous_best = rl._optional_float(rl.stage_best_group_mean_reward_target.get(stage_name))
    if previous_best is None or closed_mean > previous_best:
        rl.stage_best_group_mean_reward_target[stage_name] = float(closed_mean)
        rl._save_stage_checkpoint('best', stage_name=stage_name, group_progress_payload=group_progress_payload, reason=f'stage_local_best_improved_to_{closed_mean:.6f}')


def transition_to_stage(rl, next_stage_name, *, event, reason, group_progress_payload=None):
    previous_stage = str(rl.current_stage_name)
    if previous_stage == next_stage_name and event == 'entered':
        return
    if previous_stage != next_stage_name and event != 'recovery_entered':
        rl._save_stage_checkpoint('completed', stage_name=previous_stage, group_progress_payload=group_progress_payload, reason=reason)
        rl.current_stage_name = str(next_stage_name)
        rl.stage_entry_generation_totals[rl.current_stage_name] = rl._current_generation_total()
        rl.stage_entry_reward_batches[rl.current_stage_name] = rl.reward_batch_index
        rl._reset_stage_comparison_state()
    if event == 'recovery_entered':
        rl.recovery_active = True
        rl.recovery_start_generation_total = rl._current_generation_total()
        rl.recovery_start_discovery_family_count = len(rl.discovery_family_hashes_seen)
    elif previous_stage != next_stage_name:
        rl.recovery_active = False
        rl.recovery_start_generation_total = 0
        rl.recovery_start_discovery_family_count = 0
    rl._append_stage_event({'event': event, 'reason': reason, 'previous_stage_name': previous_stage, 'next_stage_name': rl.current_stage_name})
    rl._save_stage_checkpoint(event, stage_name=rl.current_stage_name, group_progress_payload=group_progress_payload, reason=reason)


def evaluate_stage_transitions(rl, group_progress_payload):
    stage_name = str(group_progress_payload.get('stage_name') or rl.current_stage_name)
    if rl.recovery_active:
        rl.recovery_active = False
        rl.recovery_start_generation_total = 0
        rl.recovery_start_discovery_family_count = 0
    if rl._stage1_only_enabled() and stage_name == rl.STAGE1_STRUCTURE_EXPLORE:
        return
    if stage_name == rl.STAGE1_STRUCTURE_EXPLORE:
        force_promotion_snapshot = rl._stage1_force_promotion_ready()
        if force_promotion_snapshot is not None:
            rl._transition_to_stage(rl.STAGE2_FORMAL_EXPLORE, event='entered', reason=f"stage1_forced_promotion_after_plateau: stage_groups={force_promotion_snapshot['stage_group_count']}, recent_generations={force_promotion_snapshot['recent_generation_count']}, recent_executable_count={force_promotion_snapshot['recent_executable_count']}, recent_discovery_count={force_promotion_snapshot['recent_discovery_count']}, recent_unique_discovery_families={force_promotion_snapshot['recent_unique_discovery_families']}", group_progress_payload=group_progress_payload)
            return
        if rl._stage1_gate_ready():
            rl._transition_to_stage(rl.STAGE2_FORMAL_EXPLORE, event='entered', reason='stage1_gate_satisfied', group_progress_payload=group_progress_payload)
            return
    if stage_name == rl.STAGE2_FORMAL_EXPLORE and rl._stage2_gate_ready():
        rl._transition_to_stage(rl.STAGE3_FORMAL_OPTIMIZE, event='entered', reason='stage2_gate_satisfied', group_progress_payload=group_progress_payload)


def close_reward_group_if_needed(rl):
    rl.reward_batch_index += 1
    if rl.reward_batch_index % rl.GROUP_BATCH_SIZE != 0:
        return None
    previous_closed_reward_target_mean = rl.prev_closed_group_mean_reward_target_acc
    previous_best_reward_target_mean = rl.best_closed_group_mean_reward_target_acc
    previous_closed_train_mean = rl.prev_closed_group_train_acc_mean
    previous_closed_test_mean = rl.prev_closed_group_mean_test_acc
    stage_name = str(rl.current_stage_name)
    rl.stage_closed_group_counts[stage_name] += 1
    stage_group_index = int(rl.stage_closed_group_counts[stage_name])
    closed_mean_reward_target = rl._mean_from_accumulator(rl.current_group_reward_target_sum, rl.current_group_reward_target_count)
    closed_mean_train = rl._mean_from_accumulator(rl.current_group_frozen_train_acc_sum, rl.current_group_frozen_train_acc_count)
    closed_mean_test = rl._mean_from_accumulator(rl.current_group_frozen_test_acc_sum, rl.current_group_frozen_test_acc_count)
    closed_mean_unfrozen_train = rl._mean_from_accumulator(rl.current_group_unfrozen_train_acc_sum, rl.current_group_unfrozen_train_acc_count)
    closed_mean_unfrozen_test = rl._mean_from_accumulator(rl.current_group_unfrozen_test_acc_sum, rl.current_group_unfrozen_test_acc_count)
    closed_mean_reward_target_by_backbone = {str(backbone_signature): float(rl.current_group_reward_target_sum_by_backbone.get(backbone_signature, 0.0)) / float(count) for backbone_signature, count in rl.current_group_reward_target_count_by_backbone.items() if int(count) > 0}
    rl.prev_closed_group_mean_reward_target_acc = closed_mean_reward_target
    rl.prev_closed_group_train_acc_mean = closed_mean_train
    rl.prev_closed_group_mean_test_acc = closed_mean_test
    rl.prev_closed_group_mean_reward_target_by_backbone.clear()
    rl.prev_closed_group_mean_reward_target_by_backbone.update(closed_mean_reward_target_by_backbone)
    stage1_feedback_ready = bool(rl.current_group_top_feedback) or stage_name != rl.STAGE1_STRUCTURE_EXPLORE
    if rl.best_closed_group_mean_reward_target_acc is None or (closed_mean_reward_target is not None and closed_mean_reward_target > rl.best_closed_group_mean_reward_target_acc):
        if closed_mean_reward_target is not None:
            rl.best_closed_group_mean_reward_target_acc = closed_mean_reward_target
            rl.best_closed_group_id = rl.current_group_id
            if stage1_feedback_ready:
                rl.best_group_feedback[:] = list(rl.current_group_top_feedback[:rl.FEEDBACK_SUMMARY_LIMIT])
    for backbone_signature, backbone_mean in closed_mean_reward_target_by_backbone.items():
        rl.best_closed_group_mean_reward_target_by_backbone[backbone_signature] = max(float(backbone_mean), float(rl.best_closed_group_mean_reward_target_by_backbone.get(backbone_signature, float('-inf'))))
    for goal_key, candidate_best in rl.current_group_goal_best_candidates.items():
        rl.best_reward_target_by_goal[goal_key] = max(float(candidate_best), float(rl.best_reward_target_by_goal.get(goal_key, float('-inf'))))
    if rl.best_closed_group_mean_train_acc is None or (closed_mean_train is not None and closed_mean_train > rl.best_closed_group_mean_train_acc):
        if closed_mean_train is not None:
            rl.best_closed_group_mean_train_acc = closed_mean_train
    if rl.best_closed_group_mean_test_acc is None or (closed_mean_test is not None and closed_mean_test > rl.best_closed_group_mean_test_acc):
        if closed_mean_test is not None:
            rl.best_closed_group_mean_test_acc = closed_mean_test
    total_valid = sum(rl.family_hash_archive_counts.values())
    if total_valid > 0:
        rl.dominant_family_hash, dominant_count = rl.family_hash_archive_counts.most_common(1)[0]
        rl.dominant_family_share = dominant_count / total_valid
    else:
        rl.dominant_family_hash = None
        rl.dominant_family_share = 0.0
    descriptor_total = sum(rl.descriptor_archive_counts.values())
    if descriptor_total > 0:
        rl.dominant_descriptor_key, dominant_descriptor_count = rl.descriptor_archive_counts.most_common(1)[0]
        rl.dominant_descriptor_share = dominant_descriptor_count / descriptor_total
    else:
        rl.dominant_descriptor_key = None
        rl.dominant_descriptor_share = 0.0
    backbone_total = sum(rl.backbone_signature_archive_counts.values())
    if backbone_total > 0:
        rl.dominant_backbone_signature, dominant_backbone_count = rl.backbone_signature_archive_counts.most_common(1)[0]
        rl.dominant_backbone_share = dominant_backbone_count / backbone_total
    else:
        rl.dominant_backbone_signature = None
        rl.dominant_backbone_share = 0.0
    cnn_total = sum(rl.cnn_signature_archive_counts.values())
    if cnn_total > 0:
        rl.dominant_cnn_signature, dominant_cnn_count = rl.cnn_signature_archive_counts.most_common(1)[0]
        rl.dominant_cnn_share = dominant_cnn_count / cnn_total
    else:
        rl.dominant_cnn_signature = None
        rl.dominant_cnn_share = 0.0
    backbone_cnn_total = sum(rl.backbone_cnn_pair_archive_counts.values())
    if backbone_cnn_total > 0:
        rl.dominant_backbone_cnn_pair, dominant_backbone_cnn_count = rl.backbone_cnn_pair_archive_counts.most_common(1)[0]
        rl.dominant_backbone_cnn_share = dominant_backbone_cnn_count / backbone_cnn_total
    else:
        rl.dominant_backbone_cnn_pair = None
        rl.dominant_backbone_cnn_share = 0.0
    if stage1_feedback_ready:
        rl.prev_group_feedback[:] = list(rl.current_group_top_feedback[:rl.FEEDBACK_SUMMARY_LIMIT])
    progress_path, feedback_path, best_feedback_path = rl._group_feedback_paths()
    worker_info = rl.get_eval_worker_diagnostics()
    group_progress_payload = {'group_id': rl.current_group_id, 'group_warmup': rl.current_group_id == 0, 'generation_total': rl._current_generation_total(), 'reward_batch_index': rl.reward_batch_index, 'stage_name': stage_name, 'stage_index': rl.RL_STAGE_TO_INDEX.get(stage_name, 0), 'stage_group_index': stage_group_index, 'stage_reference_min_groups': int(rl.STAGE_REFERENCE_MIN_GROUPS.get(stage_name, 0)), 'reward_target_metric': rl._stage_reward_target_metric(stage_name), 'closed_mean_reward_target_acc': closed_mean_reward_target, 'prev_closed_group_mean_reward_target_acc': previous_closed_reward_target_mean, 'best_closed_group_mean_reward_target_acc': rl.best_closed_group_mean_reward_target_acc, 'closed_mean_train_acc': closed_mean_train, 'closed_mean_test_acc': closed_mean_test, 'closed_mean_unfrozen_train_acc': closed_mean_unfrozen_train, 'closed_mean_unfrozen_test_acc': closed_mean_unfrozen_test, 'prev_closed_group_mean_train_acc': previous_closed_train_mean, 'best_closed_group_mean_train_acc': rl.best_closed_group_mean_train_acc, 'prev_closed_group_mean_test_acc': previous_closed_test_mean, 'best_closed_group_mean_test_acc': rl.best_closed_group_mean_test_acc, 'best_closed_group_id': rl.best_closed_group_id, 'improvement_vs_prev': None if closed_mean_reward_target is None or previous_closed_reward_target_mean is None else float(closed_mean_reward_target - previous_closed_reward_target_mean), 'improvement_vs_best': None if closed_mean_reward_target is None or previous_best_reward_target_mean is None else float(closed_mean_reward_target - previous_best_reward_target_mean), 'dominant_family_hash': rl.dominant_family_hash, 'dominant_family_share': rl.dominant_family_share, 'dominant_descriptor_key': rl.dominant_descriptor_key, 'dominant_descriptor_share': rl.dominant_descriptor_share, 'dominant_backbone_signature': rl.dominant_backbone_signature, 'dominant_backbone_share': rl.dominant_backbone_share, 'dominant_cnn_signature': rl.dominant_cnn_signature, 'dominant_cnn_share': rl.dominant_cnn_share, 'dominant_backbone_cnn_pair': rl.dominant_backbone_cnn_pair, 'dominant_backbone_cnn_share': rl.dominant_backbone_cnn_share, 'closed_mean_reward_target_by_backbone': rl.StageState.float_dict_payload(closed_mean_reward_target_by_backbone), 'prev_closed_group_mean_reward_target_by_backbone': rl.StageState.float_dict_payload(rl.prev_closed_group_mean_reward_target_by_backbone), 'best_closed_group_mean_reward_target_by_backbone': rl.StageState.float_dict_payload(rl.best_closed_group_mean_reward_target_by_backbone), 'unique_descriptor_count': len(rl.descriptor_archive_counts), 'trainable_samples': rl.current_group_reward_target_count, 'main_process_rss_gib': rl._read_process_rss_gib(), 'worker_rss_gib': worker_info.get('total_rss_gib', worker_info.get('rss_gib')) if worker_info else None, 'prev_group_feedback': rl._feedback_summary_payload(rl.prev_group_feedback), 'best_group_feedback': rl._feedback_summary_payload(rl.best_group_feedback)}
    rl._record_closed_group_event(group_progress_payload)
    group_progress_payload.update(rl._stage_gate_snapshot())
    rl._append_jsonl(progress_path, group_progress_payload)
    for summary in rl.prev_group_feedback:
        sample_payload = {'group_id': rl.current_group_id, 'group_warmup': rl.current_group_id == 0, 'summary': rl.asdict(summary), 'closed_mean_reward_target_acc': closed_mean_reward_target, 'closed_mean_train_acc': closed_mean_train, 'closed_mean_test_acc': closed_mean_test}
        rl._append_jsonl(feedback_path, sample_payload)
    if rl.best_closed_group_id == rl.current_group_id and rl.best_closed_group_mean_reward_target_acc is not None:
        rl._write_json(best_feedback_path, {'group_id': rl.best_closed_group_id, 'best_closed_group_mean_reward_target_acc': rl.best_closed_group_mean_reward_target_acc, 'best_closed_group_mean_train_acc': rl.best_closed_group_mean_train_acc, 'best_closed_group_mean_test_acc': rl.best_closed_group_mean_test_acc, 'feedback': rl._feedback_summary_payload(rl.best_group_feedback)})
    target_text = 'n/a' if closed_mean_reward_target is None else f'{closed_mean_reward_target:.4f}'
    prev_target_text = 'n/a' if previous_closed_reward_target_mean is None else f'{previous_closed_reward_target_mean:.4f}'
    best_target_text = 'n/a' if rl.best_closed_group_mean_reward_target_acc is None else f'{rl.best_closed_group_mean_reward_target_acc:.4f}'
    train_text = 'n/a' if closed_mean_train is None else f'{closed_mean_train:.4f}'
    test_text = 'n/a' if closed_mean_test is None else f'{closed_mean_test:.4f}'
    print(f"[Reward Group] Closed group {rl.current_group_id} after {rl.GROUP_BATCH_SIZE} reward batches: mean_target_acc={target_text}, prev_target={prev_target_text}, best_target={best_target_text}, mean_frozen_train_acc={train_text}, mean_frozen_test_acc={test_text}, trainable_samples={rl.current_group_reward_target_count}, dominant_family={rl.dominant_family_hash or 'n/a'} ({rl.dominant_family_share:.2%})")
    rl.current_group_id += 1
    rl.current_group_reward_target_sum = 0.0
    rl.current_group_reward_target_count = 0
    rl.current_group_frozen_train_acc_sum = 0.0
    rl.current_group_frozen_train_acc_count = 0
    rl.current_group_frozen_test_acc_sum = 0.0
    rl.current_group_frozen_test_acc_count = 0
    rl.current_group_unfrozen_train_acc_sum = 0.0
    rl.current_group_unfrozen_train_acc_count = 0
    rl.current_group_unfrozen_test_acc_sum = 0.0
    rl.current_group_unfrozen_test_acc_count = 0
    rl.current_group_reward_target_sum_by_backbone.clear()
    rl.current_group_reward_target_count_by_backbone.clear()
    rl._reset_current_group_feedback_state()
    rl._save_stage_checkpoint('group_closed', stage_name=stage_name, group_progress_payload=group_progress_payload, reason='closed_group', save_plot_snapshot=False)
    rl._maybe_update_stage_best_checkpoint(group_progress_payload)
    rl._evaluate_stage_transitions(group_progress_payload)
    return group_progress_payload


def current_stage_index(rl):
    return int(rl.RL_STAGE_TO_INDEX.get(rl.current_stage_name, 0))


def history_trim_in_place(rl, items, *, limit):
    if limit > 0 and len(items) > limit:
        del items[:len(items) - limit]


def append_stage_event(rl, payload):
    event_payload = {'generation_total': rl._current_generation_total(), 'reward_batch_index': rl.reward_batch_index, 'reward_group_id': rl.current_group_id, 'stage_name': rl.current_stage_name, 'stage_index': rl._current_stage_index(), **dict(payload)}
    rl.stage_event_history.append(event_payload)
    rl._history_trim_in_place(rl.stage_event_history, limit=rl.MAX_STAGE_GROUP_HISTORY)
    return event_payload


def record_generation_event(rl, payload):
    rl.generation_history.append(dict(payload))
    rl._history_trim_in_place(rl.generation_history, limit=rl.MAX_STAGE_SAMPLE_HISTORY)
    return rl.generation_history[-1]


def record_closed_group_event(rl, payload):
    rl.closed_group_history.append(dict(payload))
    rl._history_trim_in_place(rl.closed_group_history, limit=rl.MAX_STAGE_GROUP_HISTORY)
    return rl.closed_group_history[-1]


def recent_stage_generation_window(rl, stage_name, max_items):
    if max_items <= 0:
        return []
    stage_entry_generation_total = int(rl.stage_entry_generation_totals.get(stage_name, 0) or 0)
    filtered = [dict(item) for item in rl.generation_history if str(item.get('stage_name')) == str(stage_name) and int(item.get('generation_total', 0) or 0) >= stage_entry_generation_total]
    if len(filtered) > max_items:
        filtered = filtered[-max_items:]
    return filtered


def recent_stage_group_window(rl, stage_name, max_items):
    if max_items <= 0:
        return []
    stage_entry_generation_total = int(rl.stage_entry_generation_totals.get(stage_name, 0) or 0)
    filtered = [dict(item) for item in rl.closed_group_history if str(item.get('stage_name')) == str(stage_name) and int(item.get('generation_total', 0) or 0) >= stage_entry_generation_total]
    if len(filtered) > max_items:
        filtered = filtered[-max_items:]
    return filtered

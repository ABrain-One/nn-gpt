import json
import os
import shutil
import subprocess
import sys
import time
from ab.gpt.util.Const import ab_root_path

REPO_ROOT = ab_root_path


from ab.gpt.util.CycleResults import collect_cycle_metrics, generate_cycle_results, save_cycle_results

DEFAULT_OUT = REPO_ROOT / 'out' / 'benchmarks' / 'tunenngen_cifar10_adaptive_science' / 'analogical_adaptive_peak_eval16'
DEFAULT_MARGIN = 0.02
PROBE_SIZE_PER_SIDE = 4
FOLLOWUP_SIZE = 8

PROBE_SCRIPT = REPO_ROOT / 'ab/gpt/TuneNNGen_7B_code_olympic_analogical_cifar10_edit_peak_probe_safe4_risky4_eval8.py'
FOLLOWUP_SCRIPT_MAP = {
    (6, 2): REPO_ROOT / 'ab/gpt/TuneNNGen_7B_code_olympic_analogical_cifar10_edit_peak_safe6_risky2_eval8.py',
    (4, 4): REPO_ROOT / 'ab/gpt/TuneNNGen_7B_code_olympic_analogical_cifar10_edit_peak_probe_safe4_risky4_eval8.py',
    (2, 6): REPO_ROOT / 'ab/gpt/TuneNNGen_7B_code_olympic_analogical_cifar10_edit_peak_safe2_risky6_eval8.py',
}


def _parse_model_idx(model_id):
    if not isinstance(model_id, str) or not model_id.startswith('B'):
        return None
    try:
        return int(model_id[1:])
    except ValueError:
        return None


def _stage_metrics(stage_dir):
    models_base_dir = stage_dir / 'llm' / 'epoch' / 'A0' / 'synth_nn'
    eval_results_list, model_dirs_list, successful_models, failed_models = collect_cycle_metrics(
        models_base_dir,
        stage_dir / 'llm' / 'epoch' / 'A0',
    )
    return {
        'models_base_dir': models_base_dir,
        'eval_results_list': eval_results_list,
        'model_dirs_list': model_dirs_list,
        'successful_models': successful_models,
        'failed_models': failed_models,
    }


def _best_probe_accuracy(eval_results_list, start_idx, end_idx):
    accuracies = []
    for row in eval_results_list:
        idx = _parse_model_idx(row.get('model_id'))
        if idx is None or idx < start_idx or idx >= end_idx:
            continue
        accuracy = row.get('accuracy')
        if accuracy is not None:
            accuracies.append(accuracy)
    return max(accuracies) if accuracies else None


def _decide_followup_split(safe_best, risky_best, margin):
    if safe_best is None and risky_best is None:
        return (4, 4), 'no_successful_probe_candidates'
    if safe_best is None:
        return (2, 6), 'safe_probe_failed'
    if risky_best is None:
        return (6, 2), 'risky_probe_failed'

    delta = risky_best - safe_best
    if delta >= margin:
        return (2, 6), f'risky_probe_better_by_{delta:.4f}'
    if delta <= -margin:
        return (6, 2), f'safe_probe_better_by_{-delta:.4f}'
    return (4, 4), f'probe_difference_within_margin_{abs(delta):.4f}'


def _run_stage(stage_name, script_path, out_dir):
    env = os.environ.copy()
    env['AB_GPT_NNGPT_DIR'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{stage_name}] starting -> {script_path.name}')
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)
    cycle_results = out_dir / 'cycle_results.json'
    if not cycle_results.exists():
        raise RuntimeError(f'missing cycle_results.json for {stage_name}: {out_dir}')
    print(f'[{stage_name}] finished -> {cycle_results}')


def _summarize_stage(stage_dir):
    cycle_path = stage_dir / 'cycle_results.json'
    if not cycle_path.exists():
        return None
    with open(cycle_path) as f:
        return json.load(f)


def _copy_stage_summary(stage_dir, final_dir, name):
    cycle_path = stage_dir / 'cycle_results.json'
    if cycle_path.exists():
        shutil.copy2(cycle_path, final_dir / f'{name}_cycle_results.json')


def main():
    final_dir = Path(os.environ.get('AB_GPT_NNGPT_DIR', str(DEFAULT_OUT))).resolve()
    margin = float(os.environ.get('AB_GPT_ADAPTIVE_MARGIN', str(DEFAULT_MARGIN)))
    stage1_dir = final_dir / 'stage1_probe'
    stage2_dir = final_dir / 'stage2_followup'

    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    _run_stage('stage1_probe', PROBE_SCRIPT, stage1_dir)
    stage1_metrics = _stage_metrics(stage1_dir)
    safe_best = _best_probe_accuracy(stage1_metrics['eval_results_list'], 0, PROBE_SIZE_PER_SIDE)
    risky_best = _best_probe_accuracy(stage1_metrics['eval_results_list'], PROBE_SIZE_PER_SIDE, PROBE_SIZE_PER_SIDE * 2)
    followup_split, decision_reason = _decide_followup_split(safe_best, risky_best, margin)
    followup_script = FOLLOWUP_SCRIPT_MAP[followup_split]

    decision_payload = {
        'objective': 'peak_accuracy_under_fixed_budget',
        'margin': margin,
        'probe_split': {'safe': PROBE_SIZE_PER_SIDE, 'risky': PROBE_SIZE_PER_SIDE},
        'probe_best_accuracy': {'safe': safe_best, 'risky': risky_best},
        'followup_split': {'safe': followup_split[0], 'risky': followup_split[1]},
        'followup_budget': FOLLOWUP_SIZE,
        'decision_reason': decision_reason,
        'stage1_script': str(PROBE_SCRIPT),
        'stage2_script': str(followup_script),
    }
    with open(final_dir / 'adaptive_decision.json', 'w') as f:
        json.dump(decision_payload, f, indent=2)

    print(f"[adaptive] decision -> safe={followup_split[0]} risky={followup_split[1]} ({decision_reason})")
    _run_stage('stage2_followup', followup_script, stage2_dir)

    stage2_metrics = _stage_metrics(stage2_dir)
    total_time_minutes = (time.time() - start_time) / 60.0

    combined_eval_results = stage1_metrics['eval_results_list'] + stage2_metrics['eval_results_list']
    combined_model_dirs = stage1_metrics['model_dirs_list'] + stage2_metrics['model_dirs_list']
    combined_successful_models = stage1_metrics['successful_models'] + stage2_metrics['successful_models']
    combined_failed_models = stage1_metrics['failed_models'] + stage2_metrics['failed_models']

    final_results = generate_cycle_results(
        cycle=0,
        models_base_dir=final_dir,
        eval_results_list=combined_eval_results,
        model_dirs_list=combined_model_dirs,
        successful_models=combined_successful_models,
        failed_models=combined_failed_models,
        cycle_time_minutes=total_time_minutes,
        current_alter_epoch_path=final_dir,
    )
    final_results['adaptive'] = {
        **decision_payload,
        'stage1_cycle': _summarize_stage(stage1_dir),
        'stage2_cycle': _summarize_stage(stage2_dir),
    }

    save_cycle_results(final_results, final_dir / 'cycle_results.json')
    _copy_stage_summary(stage1_dir, final_dir, 'stage1_probe')
    _copy_stage_summary(stage2_dir, final_dir, 'stage2_followup')
    print(json.dumps(final_results, indent=2))


if __name__ == '__main__':
    main()

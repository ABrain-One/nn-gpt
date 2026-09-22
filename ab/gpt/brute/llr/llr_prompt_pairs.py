"""
Standalone LLR training-pair construction, used to prepare the (vanilla -> llr)
dataset for the layerwise-LR fine-tune outside of the standard NNGenPrompt /
Tune.py pipeline. Not wired into the generic pipeline dispatch — run directly
(or from your own driver script) rather than through a prompt-config flag.
"""

import json
import re
from collections import defaultdict

import pandas as pd
from pandas import DataFrame

from ab.gpt.brute.llr.llr_baselines import (
    LLR_META_COLS,
    load_llr_metadata,
    load_vanilla_baselines,
)


def extract_layerwise_lr_strategy(code: str) -> str:
    if not isinstance(code, str):
        return 'unknown'
    m = re.search(r'#\s*Layerwise LR strategy:\s*(\S+)', code)
    return m.group(1) if m else 'unknown'


def enrich_pairs_with_llr_meta(data: DataFrame, meta: dict) -> DataFrame:
    """
    Add strategy metadata columns for the reference model (_2 suffix).
    Pulls from the pre-built CSV so no extra DB round-trip is needed.
    """
    if not meta or 'nn_2' not in data.columns:
        return data
    data = data.copy()
    for col in ('strategy', 'strategy_type', 'n_groups', 'multipliers', 'split_ratios', 'description', 'architecture'):
        data[f'{col}_2'] = data['nn_2'].map(lambda nn: meta.get(nn, {}).get(col, ''))
    return data


def match_llr_to_vanilla(llr_df: DataFrame, require_improve: bool) -> list[dict]:
    """
    Core matcher shared by the Selection and Mechanism bucket builders: for each
    evaluated llr model, recover its architecture + strategy spec from the
    metadata CSV, and pair it with its vanilla baseline at the SAME
    (dataset, epoch, transform) — a fair, confounder-free comparison.

    require_improve=True  -> keep only pairs where the llr model beat its baseline
    require_improve=False -> keep every matched pair regardless of outcome

    Returns a list of combined-row dicts (vanilla bare cols + '_2'-suffixed llr
    cols + strategy spec), each also carrying '_delta' (float) for ranking/capping.
    """
    meta = load_llr_metadata()
    if not meta or llr_df is None or llr_df.empty or 'nn' not in llr_df.columns:
        return []
    arch_names = {m.get('architecture') for m in meta.values()}
    uniform_map = {nn: m.get('architecture') for nn, m in meta.items()
                   if 'uniform' in (m.get('strategy', '').lower()) and m.get('architecture')}
    code_lut, vanilla_acc_lut, uniform_acc_lut = load_vanilla_baselines(arch_names, uniform_map)
    if not code_lut:
        return []

    rows = []
    matched = missing_arch = missing_base = lost_improve = 0
    for _, t in llr_df.iterrows():
        m = meta.get(t['nn'])
        if not m or not m.get('architecture'):
            missing_arch += 1
            continue
        if 'uniform' in (m.get('strategy', '').lower()):
            continue  # control group is a baseline, never a training target
        arch = m['architecture']
        ds = t.get('dataset')
        tf = t.get('transform')
        try:
            ep = int(t.get('epoch'))
        except (TypeError, ValueError):
            missing_base += 1
            continue
        v_code = code_lut.get(arch)
        # Baseline accuracy: uniform control (same conditions) first, then DB vanilla,
        # both transform-matched; final fallback is best baseline at (arch,dataset,epoch).
        v_acc = uniform_acc_lut.get((arch, ds, ep, tf))
        if v_acc is None:
            v_acc = vanilla_acc_lut.get((arch, ds, ep, tf))
        if v_acc is None:
            cands = [a for (n, d, e, _tf), a in vanilla_acc_lut.items()
                     if n == arch and d == ds and e == ep]
            v_acc = max(cands) if cands else None
        if v_code is None or v_acc is None:
            missing_base += 1
            continue
        t_acc = t.get('accuracy') or 0
        delta = t_acc - v_acc
        if require_improve and delta <= 0:
            lost_improve += 1
            continue
        # Vanilla baseline keeps bare column names. Only baseline fields the
        # prompt actually renders need to be accurate (nn_code, accuracy, epoch,
        # dataset, task, metric); the rest fall back to the target's values and
        # are never shown (the response uses the addon_/_2 side).
        combined = {
            'nn': arch,
            'nn_code': v_code,
            'accuracy': v_acc,
            'epoch': ep,
            'dataset': ds,
            'task': t.get('task'),
            'metric': t.get('metric'),
            'metric_code': t.get('metric_code'),
            'transform_code': t.get('transform_code'),
            'prm': t.get('prm'),
            '_delta': delta,
        }
        for col, val in t.items():
            combined[f'{col}_2'] = val
        for c in LLR_META_COLS:
            combined[f'{c}_2'] = m.get(c, '')
        rows.append(combined)
        matched += 1

    print(f"[VANILLA] matched={matched}, missing_arch={missing_arch}, "
          f"missing_baseline={missing_base}, below_vanilla={lost_improve}")
    return rows


def cap_per_group(rows: list[dict], group_cols: tuple, top_k: int) -> list[dict]:
    """
    Keep only the top_k rows (by '_delta', descending) per group_cols key.
    Used to balance the dataset across architectures — without this, an
    architecture with many winning strategies (e.g. 34 for GoogLeNot) would
    dominate training relative to one with a single narrow win.
    """
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c) for c in group_cols)
        groups[key].append(r)
    out = []
    for grp in groups.values():
        grp.sort(key=lambda r: r.get('_delta', 0), reverse=True)
        out.extend(grp[:top_k])
    return out


def build_vanilla_anchored_pairs(
    llr_df: DataFrame,
    improve: bool,
    max_rows: int | None,
    top_k_per_group: int | None = None,
    group_cols: tuple = ('nn', 'dataset'),
) -> DataFrame:
    """
    Build (vanilla -> llr) training pairs — the controlled before/after of the
    layerwise-LR injection.

    Two modes, selected by `improve`:
      improve=True  -> "Selection" bucket: only pairs where the llr strategy beat
                        its vanilla baseline. With top_k_per_group set, keeps only
                        the best K strategies per (architecture, dataset) so a few
                        architectures with many wins don't dominate the dataset.
      improve=False -> "Mechanism" bucket: every matched pair regardless of
                        outcome — teaches faithful spec->code injection on
                        architectures/strategies that never won, including
                        architectures with zero wins anywhere.

    Column layout mirrors the self-join convention the prompt expects:
        baseline (vanilla) -> bare names   -> {nn_code}, {accuracy}, ...
        target   (llr)     -> '_2' suffix  -> {addon_nn_code}=nn_code_2, ...
    The llr strategy spec (multipliers/splits/n_groups/description/...) is attached
    as '<col>_2' so the prompt can explain *why*/*what* the target does.

    Returns an empty DataFrame if metadata/DB are unavailable, so callers can fall
    back to the legacy llr->llr self-join.
    """
    rows = match_llr_to_vanilla(llr_df, require_improve=improve)
    if not rows:
        return pd.DataFrame()
    if top_k_per_group:
        rows = cap_per_group(rows, group_cols, top_k_per_group)
    for r in rows:
        r.pop('_delta', None)
    df = pd.DataFrame(rows)
    return df.head(max_rows) if max_rows else df


def build_join_pairs_without_sql(
    base_df: DataFrame,
    same_cols: tuple,
    diff_cols: tuple,
    improve: bool,
    max_rows: int | None,
) -> DataFrame:
    """
    Construct improvement pairs from a single-model DataFrame without a SQL JOIN.
    For each group (same_cols), pair every two rows where diff_cols differ and
    (if improve=True) the second has strictly higher accuracy than the first.
    Column naming mirrors JoinConf: base row keeps column names, partner columns
    get '_2' suffix.

    Fallback path used when build_vanilla_anchored_pairs() finds no vanilla
    baseline to anchor to (legacy llr -> llr self-join).
    """
    if base_df.empty:
        return base_df

    group_keys = [c for c in same_cols if c in base_df.columns]
    rows = []
    grouped = base_df.groupby(group_keys, sort=False) if group_keys else [(None, base_df)]
    for _, grp in grouped:
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i == j:
                    continue
                r1, r2 = grp.iloc[i], grp.iloc[j]
                # diff_cols must differ between the two rows
                if any(r1.get(c) == r2.get(c) for c in diff_cols if c in r1.index):
                    continue
                if improve and r2.get('accuracy', 0) <= r1.get('accuracy', 0):
                    continue
                combined = dict(r1)
                for col in r2.index:
                    combined[f'{col}_2'] = r2[col]
                rows.append(combined)
                if max_rows and len(rows) >= max_rows:
                    return pd.DataFrame(rows)
    result = pd.DataFrame(rows)
    if max_rows:
        result = result.head(max_rows)
    return result


def build_llr_join_dataset(lemur, key_dict: dict, use_join: bool, n_training_prompts: int | None) -> DataFrame:
    """
    LLR replacement for the `lemur.data(nn_prefixes=..., sql=JoinConf(...))` call:
    nn_prefixes always produces broken SQL once combined with a JOIN, so this
    fetches with only_best_accuracy=True (~15K rows, no full-table scan), filters
    to the llr-/llr2-/llr3- prefixes in Python, then builds vanilla-anchored
    training pairs (falling back to a plain llr->llr self-join if no vanilla
    baseline is found).
    """
    nn_prefixes = tuple(key_dict.get('nn_prefixes') or [])
    same_cols = tuple(key_dict.get('keep_same', []))
    diff_cols = tuple(key_dict.get('no_repeat', []))
    improve = key_dict.get('improve', False)

    base_data = lemur.data(only_best_accuracy=True, task=key_dict.get('task'), max_rows=None)
    if nn_prefixes:
        if 'nn' not in base_data.columns:
            print(f"[WARNING] lemur.data() returned no 'nn' column — DB may be empty. columns={base_data.columns.tolist()}")
            base_data = base_data.iloc[0:0]
        else:
            mask = base_data['nn'].apply(lambda v: any(str(v).startswith(p) for p in nn_prefixes))
            base_data = base_data[mask].reset_index(drop=True)
    print(f"[PREFETCH] {len(base_data)} prefix-filtered rows for nn_prefixes={nn_prefixes}")

    if use_join:
        # Primary: vanilla -> llr pairs (controlled before/after of the LR
        # injection, matched on arch+dataset+epoch). Fall back to the legacy
        # llr -> llr self-join only if baselines are unavailable. top_k_per_group
        # (Selection bucket) caps per (arch,dataset) so a few high-win
        # architectures don't dominate training; absent for the Mechanism
        # bucket (improve=False), which wants full coverage capped instead
        # per-architecture (see key config).
        data = build_vanilla_anchored_pairs(
            base_data, improve, n_training_prompts,
            top_k_per_group=key_dict.get('top_k_per_group'),
            group_cols=tuple(key_dict.get('top_k_group_cols', ('nn', 'dataset'))))
        if data.empty:
            print("[VANILLA] no vanilla-anchored pairs — falling back to llr→llr self-join")
            data = build_join_pairs_without_sql(base_data, same_cols, diff_cols, improve, n_training_prompts)
            data = enrich_pairs_with_llr_meta(data, load_llr_metadata())
    else:
        data = base_data.head(n_training_prompts) if n_training_prompts else base_data
    print(f"[FILTER] nn_prefixes={nn_prefixes}: {len(data)} rows after Python JOIN")
    return data


def count_available_llr_pairs(prompts_path) -> None:
    """
    Debug/preflight tool: print the number of available training pairs per key
    in an LLR prompt config, without running full prompt generation.
    """
    import ab.nn.api as lemur

    with open(prompts_path) as f:
        prompt_dict = json.load(f)
    for key, key_dict in prompt_dict.items():
        nn_prefixes = tuple(key_dict.get('nn_prefixes') or [])
        if not nn_prefixes:
            print(f"[PREFLIGHT] key={key}: no nn_prefixes filter, skipping count")
            continue
        use_join = key_dict.get('num_joint_nns', 1) >= 2
        data = build_llr_join_dataset(lemur, key_dict, use_join, None)
        print(f"[PREFLIGHT] key={key}, nn_prefixes={nn_prefixes}: {len(data)} available training pairs")

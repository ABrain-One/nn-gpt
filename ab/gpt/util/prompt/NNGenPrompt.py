import json
import re

import ab.nn.api as lemur
from overrides import override
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from ab.gpt.util.lemur_enrichment import patch_join_nn_query, enrich_dataframe
from tqdm import tqdm

from ab.nn.util.db.Query import JoinConf
from ab.gpt.util.Util import evaluate_delimited_formulas


from ab.gpt.brute.llr.llr_baselines import (
    LLR_META_COLS as _LLR_META_COLS,
    load_llr_metadata as _load_llr_metadata,
    load_vanilla_baselines as _load_vanilla_baselines,
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


def _match_llr_to_vanilla(llr_df: DataFrame, require_improve: bool) -> list[dict]:
    """
    Core matcher shared by the Selection and Mechanism bucket builders: for each
    evaluated llr model, recover its architecture + strategy spec from the
    metadata CSV, and pair it with its vanilla baseline at the SAME
    (dataset, epoch, transform) — a fair, confounder-free comparison.

    require_improve=True  -> keep only pairs where the llr model beat its baseline
    require_improve=False -> keep every matched pair regardless of outcome

    Returns a list of combined-row dicts (vanilla bare cols + '_2'-suffixed llr
    cols + strategy spec), each also carrying 'delta' (float) for ranking/capping.
    """
    meta = _load_llr_metadata()
    if not meta or llr_df is None or llr_df.empty or 'nn' not in llr_df.columns:
        return []
    arch_names = {m.get('architecture') for m in meta.values()}
    uniform_map = {nn: m.get('architecture') for nn, m in meta.items()
                   if 'uniform' in (m.get('strategy', '').lower()) and m.get('architecture')}
    code_lut, vanilla_acc_lut, uniform_acc_lut = _load_vanilla_baselines(arch_names, uniform_map)
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
        for c in _LLR_META_COLS:
            combined[f'{c}_2'] = m.get(c, '')
        rows.append(combined)
        matched += 1

    print(f"[VANILLA] matched={matched}, missing_arch={missing_arch}, "
          f"missing_baseline={missing_base}, below_vanilla={lost_improve}")
    return rows


def _cap_per_group(rows: list[dict], group_cols: tuple, top_k: int) -> list[dict]:
    """
    Keep only the top_k rows (by '_delta', descending) per group_cols key.
    Used to balance the dataset across architectures — without this, an
    architecture with many winning strategies (e.g. 34 for GoogLeNot) would
    dominate training relative to one with a single narrow win.
    """
    from collections import defaultdict
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
    rows = _match_llr_to_vanilla(llr_df, require_improve=improve)
    if not rows:
        return pd.DataFrame()
    if top_k_per_group:
        rows = _cap_per_group(rows, group_cols, top_k_per_group)
    for r in rows:
        r.pop('_delta', None)
    df = pd.DataFrame(rows)
    return df.head(max_rows) if max_rows else df


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def _build_join_pairs(
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


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @staticmethod
    def count_available(prompts_path) -> None:
        with open(prompts_path) as f:
            prompt_dict = json.load(f)
        for key, key_dict in prompt_dict.items():
            nn_prefixes = tuple(key_dict.get('nn_prefixes') or [])
            if not nn_prefixes:
                print(f"[PREFLIGHT] key={key}: no nn_prefixes filter, skipping count")
                continue
            num_joint_nns = key_dict.get('num_joint_nns', 1)
            use_join = num_joint_nns >= 2
            same_cols = tuple(key_dict.get('keep_same', []))
            diff_cols = tuple(key_dict.get('no_repeat', []))
            improve = key_dict.get('improve', False)
            # nn_prefixes in lemur.data() generates broken SQL — always pass empty and filter in Python
            base_data = lemur.data(only_best_accuracy=True, task=key_dict.get('task'))
            if nn_prefixes:
                if 'nn' not in base_data.columns:
                    print(f"[PREFLIGHT] key={key}: DB empty or missing 'nn' column, skipping count")
                    continue
                mask = base_data['nn'].apply(lambda v: any(str(v).startswith(p) for p in nn_prefixes))
                base_data = base_data[mask].reset_index(drop=True)
            if use_join:
                data = build_vanilla_anchored_pairs(
                    base_data, improve, max_rows=None,
                    top_k_per_group=key_dict.get('top_k_per_group'))
                if data.empty:
                    data = _build_join_pairs(base_data, same_cols, diff_cols, improve, max_rows=None)
            else:
                data = base_data
            print(f"[PREFLIGHT] key={key}, nn_prefixes={nn_prefixes}: {len(data)} available training pairs")

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        prompt_lists = []

        # /workspace/nn-gpt/ab/gpt/conf/prompt/train/NN_gen.json
        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            dataframe = DataFrame(
                columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            print('Preparing Data...', flush=True)
            key_dict = prompt_dict[key]
            num_joint_nns = key_dict.get('num_joint_nns') or 1

            # For JOIN queries, do NOT pass max_rows — the LIMIT applies before
            # the JOIN and causes an O(n²) correlated scan. Slice the result after.
            use_join = num_joint_nns >= 2
            system_text = '\n'.join(prompt_dict[key].get('system', []))
            prompt_template = '\n'.join(prompt_dict[key]['prompt'])
            print(f'Preparing Data for key: {key}...', flush=True)

            # ========== SMALL SWITCH TO DETECT PRUNING CONFIG ==========
            # Check if this is a pruning task (key starts with 'pruning' or contains 'pruning')
            is_pruning = (
                key.lower().startswith('pruning') or
                'pruning' in key.lower()
            )

            print(f"[DEBUG] Key: {key}, is_pruning: {is_pruning}")

            if is_pruning:
                # Use prun table for pruning statistics
                data = lemur.prun_data(max_rows=n_training_prompts)
                # Filter only successful pruning experiments
                if 'status' in data.columns:
                    data = data[data['status'] == 'success']
                print(
                    f"[PRUN] Fetched {len(data)} records from PRUN table for key: {key}")
            else:
                # for classification tasks: Patch LEMUR's join query before the data call so that dataset_2
                # and its siblings appear in the result set.

                classification = use_join and key_dict.get('output_type') == 'classification'
                if classification:
                    patch_join_nn_query() # # TODO: Generalize for all scenarios - SQL query implementation in the NN Dataset project
                nn_prefixes = tuple(key_dict.get('nn_prefixes') or [])
                same_cols = tuple(key_dict.get('keep_same', []))
                diff_cols = tuple(key_dict.get('no_repeat', []))
                improve = key_dict.get('improve', False)

                # nn_prefixes in lemur.data() always generates broken SQL — never pass it.
                # Fetch with only_best_accuracy=True (~15K rows), filter to llr- in Python,
                # then build improvement pairs. This avoids both the SQL bug and a full-table scan.
                if use_join or nn_prefixes:
                    base_data = lemur.data(
                        only_best_accuracy=True,
                        task=key_dict.get('task'),
                        max_rows=None,
                    )
                    if nn_prefixes:
                        if 'nn' not in base_data.columns:
                            print(f"[WARNING] lemur.data() returned no 'nn' column — DB may be empty. columns={base_data.columns.tolist()}")
                            base_data = base_data.iloc[0:0]
                        else:
                            mask = base_data['nn'].apply(lambda v: any(str(v).startswith(p) for p in nn_prefixes))
                            base_data = base_data[mask].reset_index(drop=True)
                    print(f"[PREFETCH] {len(base_data)} prefix-filtered rows for nn_prefixes={nn_prefixes}")
                    if use_join:
                        # Primary: vanilla -> llr pairs (controlled before/after of the
                        # LR injection, matched on arch+dataset+epoch). Fall back to the
                        # legacy llr -> llr self-join only if baselines are unavailable.
                        # top_k_per_group (Selection bucket) caps per (arch,dataset) so a
                        # few high-win architectures don't dominate training; absent for
                        # the Mechanism bucket (improve=False), which wants full coverage
                        # capped instead per-architecture (see key config below).
                        data = build_vanilla_anchored_pairs(
                            base_data, improve, n_training_prompts,
                            top_k_per_group=key_dict.get('top_k_per_group'),
                            group_cols=tuple(key_dict.get('top_k_group_cols', ('nn', 'dataset'))))
                        if data.empty:
                            print("[VANILLA] no vanilla-anchored pairs — falling back to llr→llr self-join")
                            data = _build_join_pairs(base_data, same_cols, diff_cols, improve, n_training_prompts)
                            data = enrich_pairs_with_llr_meta(data, _load_llr_metadata())
                    else:
                        data = base_data.head(n_training_prompts) if n_training_prompts else base_data
                    print(f"[FILTER] nn_prefixes={nn_prefixes}: {len(data)} rows after Python JOIN")
                else:
                    data = lemur.data(
                        only_best_accuracy=only_best_accuracy,
                        task=key_dict.get('task'),
                        max_rows=n_training_prompts,
                        sql=None if not use_join else JoinConf(
                            num_joint_nns=num_joint_nns,
                            same_columns=same_cols,
                            diff_columns=diff_cols,
                            enhance_nn=improve,
                        )
                    )
                # For classification tasks, enrich the DataFrame with normalised
                # accuracy and dataset-metadata columns needed for the prompt.
                if classification:
                    enrich_dataframe(data)  # TODO: Generalize for all scenarios based on the formula implementation (see evaluate_delimited_formulas(..))
                print(f"[STAT] Fetched {len(data)} records from STAT table for key: {key}")
            # ==========================================================

            print('Data acquisition complete', flush=True)

            # Check if this is delta mode
            use_delta = key_dict.get(
                'use_delta', False) or 'delta' in key.lower()

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break

                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    # Handle column name mapping gracefully
                    db_column = it['value']
                    try:
                        if db_column in row:
                            para_dict[it['para']] = row[db_column]
                        elif db_column == 'model_name' and 'nn' in row:
                            para_dict[it['para']] = row['nn']
                        elif db_column == 'nn' and 'model_name' in row:
                            para_dict[it['para']] = row['model_name']
                        else:
                            para_dict[it['para']] = row.get(
                                db_column, f"Missing: {db_column}")
                    except Exception as e:
                        print(
                            f"[WARNING] Could not get column '{db_column}': {e}")
                        para_dict[it['para']] = None

                # Cap only the PROMPT-context code (baseline). Never truncate the
                # response fields (addon_nn_code / addon_transform_code) — they are
                # the training target; clipping them produces broken Python labels.
                # Over-long responses are dropped later by the max_new_tokens filter.
                # Stash the untouched codes first: the training-target delta
                # must be computed between the FULL baseline and FULL improved
                # files (its hunk lives inside train_setup, which the shrunk
                # prompt still shows verbatim), never between shrunk/capped
                # views.
                full_nn_code = para_dict.get('nn_code')
                full_addon_nn_code = para_dict.get('addon_nn_code')

                if key_dict.get('shrink_nn_code') and isinstance(para_dict.get('nn_code'), str):
                    from ab.gpt.util.DeltaUtil import shrink_nn_code_for_prompt
                    para_dict['nn_code'] = shrink_nn_code_for_prompt(para_dict['nn_code'])

                nn_code_max_chars = key_dict.get('nn_code_max_chars')
                if nn_code_max_chars:
                    for _ck in ('nn_code', 'transform_code'):
                        if _ck in para_dict and isinstance(para_dict[_ck], str):
                            para_dict[_ck] = para_dict[_ck][:nn_code_max_chars]

                # Strip hardware/runtime stats from prm fields so the response stays short enough
                # to pass the max_new_tokens filter. DB prm blobs include gpu_memory_kb, cpu_type, etc.
                _HW_KEYS = {
                    'cpu_count', 'cpu_type', 'cpu_usage_percent',
                    'total_ram_kb', 'occupied_ram_kb', 'ram_usage_percent',
                    'gpu_type', 'gpu_memory_kb', 'gpu_total_memory_kb',
                    'occupied_gpu_memory_kb', 'gpu_memory_usage_percent',
                    'gradient_norm', 'samples_per_second', 'epoch_max',
                    'best_accuracy', 'best_epoch', 'metric_acc',
                    'train_loss', 'test_loss', 'train_accuracy',
                }
                for _pk in ('prm', 'addon_prm'):
                    if _pk not in para_dict:
                        continue
                    val = para_dict[_pk]
                    try:
                        import ast as _ast
                        if isinstance(val, dict):
                            _d = val
                        elif isinstance(val, str):
                            _d = _ast.literal_eval(val)
                        else:
                            continue
                        if isinstance(_d, dict):
                            para_dict[_pk] = str({k: v for k, v in _d.items() if k not in _HW_KEYS})
                    except Exception:
                        pass

                # Inject columns referenced in the output template but absent from input_list
                if key_dict.get('output_type') == 'classification':
                    output_template = '\n'.join(key_dict['output'])
                    for col in row.index:
                        if f'{{{col}}}' in output_template and col not in para_dict:
                            para_dict[col] = row[col]

                # ========== APPLY FORMULA EVALUATION TO PROMPT ==========
                inst = prompt.format(**para_dict)
                inst = evaluate_delimited_formulas(inst, para_dict)
                # ========================================================

                # Compute delta if delta mode is enabled
                if use_delta and 'addon_nn_code' in para_dict and 'nn_code' in para_dict:
                    try:
                        from ab.gpt.util.DeltaUtil import compute_delta
                        baseline_code = full_nn_code if isinstance(full_nn_code, str) else para_dict.get('nn_code', '')
                        improved_code = full_addon_nn_code if isinstance(full_addon_nn_code, str) else para_dict.get('addon_nn_code', '')

                        if baseline_code and improved_code:
                            computed_delta = compute_delta(
                                baseline_code, improved_code)
                            if not computed_delta:
                                computed_delta = ""
                        else:
                            computed_delta = ""

                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace(
                            '{computed_delta}', computed_delta)
                    except Exception as e:
                        print(
                            f'[WARNING] Failed to compute delta for key {key}: {e}. Using regular output.', flush=True)
                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_delta}', '')
                else:
                    # Regular mode: use output template as-is
                    output = '\n'.join(prompt_dict[key]['output'])
                    response = output.format(**para_dict)

                # ========== APPLY FORMULA EVALUATION TO RESPONSE ==========
                response = evaluate_delimited_formulas(response, para_dict)
                # ==========================================================

                # ========== PRINT FOR VERIFICATION (AFTER response EXISTS) ==========
                if len(dataframe) < 10:
                    print(f"\n[EXAMPLE {len(dataframe)+1}]:")
                    print(f"SYS: {system_text}")
                    print(f"USER: {inst[:1000]}...")
                    print(f"OUTPUT: {response[:500]}...")
                    print("-" * 50)
                # ================================================================

                text = self.tokenizer.apply_chat_template(
                    self._build_messages(
                        inst, response, system_prompt=system_text or None),
                    tokenize=False)

                # print(f"Prompt: {inst}", flush=True)
                # print(f"Output: {response}", flush=True)

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data
        return pd.concat(prompt_lists, ignore_index=True)

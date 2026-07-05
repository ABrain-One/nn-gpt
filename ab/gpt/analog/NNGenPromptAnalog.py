import json

import ab.nn.api as lemur
from datasets import Dataset
from overrides import override
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from tqdm import tqdm

from ab.nn.api import JoinConf
from ab.gpt.util.EditUtil import apply_source_edit_policy, build_source_edit_hint, infer_edit_spec, prune_edit_spec, prune_source_edit_hint, rank_source_edit_candidate, summarize_cifar_target
from ab.gpt.analog.UtilAnalog import is_cifar_unsafe_seed, is_small_rgb32_dataset


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def _stringify_prompt_value(value):
    if isinstance(value, dict):
        return json.dumps(value, separators=(',', ':'))
    return value


def _minify_block(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = []
    blank = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.strip().startswith('#'):
            continue
        if not stripped.strip():
            if blank:
                continue
            blank = True
            out.append('')
            continue
        blank = False
        out.append(stripped)
    return '\n'.join(out).strip()


def _compact_code_block(text: str, head_lines: int, tail_lines: int = 0) -> str:
    if not isinstance(text, str):
        return text
    compact = _minify_block(text)
    lines = compact.splitlines()
    if len(lines) <= head_lines + tail_lines + 1:
        return compact
    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines else []
    marker = ['# ... prompt-trimmed ...']
    return '\n'.join(head + marker + tail)


def _compact_unified_diff(diff_text: str, max_hunks: int = 2, max_lines_per_hunk: int = 18) -> str:
    if not isinstance(diff_text, str) or not diff_text.strip():
        return diff_text

    lines = diff_text.splitlines()
    if len(lines) <= 2:
        return diff_text

    header = lines[:2]
    body = lines[2:]
    compact = list(header)
    kept_hunks = 0
    i = 0

    while i < len(body) and kept_hunks < max_hunks:
        line = body[i]
        if not line.startswith('@@'):
            i += 1
            continue

        compact.append(line)
        i += 1
        kept_hunks += 1
        kept_lines = 0

        while i < len(body) and not body[i].startswith('@@'):
            if kept_lines < max_lines_per_hunk:
                compact.append(body[i])
            kept_lines += 1
            i += 1

    return '\n'.join(compact)


def _compact_prompt_fields(para_dict: dict, aggressive: bool = False, edit_schema=None, source_edit_policy=None) -> dict:
    compact = dict(para_dict)
    compact.setdefault('computed_edit', '')
    compact.setdefault('computed_delta', '')
    if 'source_edit' in compact and source_edit_policy and isinstance(compact['source_edit'], str):
        try:
            compact['source_edit'] = json.dumps(
                apply_source_edit_policy(json.loads(compact['source_edit']), source_edit_policy),
                separators=(',', ':'),
            )
        except Exception:
            pass
    if 'target_summary' not in compact and 'nn_code' in para_dict:
        try:
            compact['target_summary'] = json.dumps(
                summarize_cifar_target(
                    para_dict.get('nn_code', ''),
                    prm=para_dict.get('prm'),
                    model_name=para_dict.get('nn'),
                    accuracy=para_dict.get('accuracy'),
                ),
                separators=(',', ':'),
            )
        except Exception:
            compact['target_summary'] = '{"has_adaptive_avg":false,"has_classifier_dropout":false,"final_channels":0,"max_channels":0,"init_block_channels":0,"stem_kernel_size":0,"stem_stride":0,"has_stem_pool":false}'
    if 'source_edit' not in compact and 'source_nn_code' in para_dict and 'nn_code' in para_dict:
        try:
            compact['source_edit'] = json.dumps(
                apply_source_edit_policy(
                    prune_source_edit_hint(
                        build_source_edit_hint(
                            para_dict.get('nn_code', ''),
                            para_dict.get('source_nn_code', ''),
                            baseline_prm=para_dict.get('prm'),
                            improved_prm=para_dict.get('source_prm'),
                            baseline_name=para_dict.get('nn'),
                            improved_name=para_dict.get('source_nn'),
                            baseline_accuracy=para_dict.get('accuracy'),
                            improved_accuracy=para_dict.get('source_accuracy'),
                        ),
                        edit_schema,
                    ),
                    source_edit_policy,
                ),
                separators=(',', ':'),
            )
        except Exception:
            compact['source_edit'] = '{"suggested_edit":{"mode":"hp_transform_only","width":"same","init_width":"same","pool":"keep","stem_stride":"keep","stem_pool":"keep","classifier_dropout":0.0},"safe_edit":{"mode":"hp_transform_only","width":"same","init_width":"same","pool":"keep","stem_stride":"keep","stem_pool":"keep","classifier_dropout":0.0},"edit_bias":"safe_edit","source_traits":[],"transfer_focus":[],"target_already_has":[]}'
    for key, value in list(compact.items()):
        compact[key] = _stringify_prompt_value(value)

        if not isinstance(compact[key], str):
            continue

        if key.startswith('source_') and 'nn_code' in key:
            compact[key] = _compact_code_block(compact[key], 36 if not aggressive else 24, 16 if not aggressive else 8)
        elif key.startswith('source_') and 'transform_code' in key:
            compact[key] = _compact_code_block(compact[key], 10 if not aggressive else 6)
        elif aggressive and key == 'nn_code':
            compact[key] = _compact_code_block(compact[key], 100, 32)
        elif aggressive and key == 'transform_code':
            compact[key] = _compact_code_block(compact[key], 12)
    return compact


def _select_addon_row(available_addon, baseline_row, key_dict):
    if available_addon is None or available_addon.empty:
        return None

    strategy = key_dict.get('addon_selection', 'random')
    if strategy == 'best_accuracy' and 'accuracy' in available_addon.columns:
        return available_addon.sort_values('accuracy', ascending=False).iloc[0]

    if strategy in {'best_safe_edit', 'best_risky_edit'}:
        baseline_code = baseline_row.get('nn_code', '')
        baseline_prm = baseline_row.get('prm')
        best = None
        for _, addon_row in available_addon.iterrows():
            try:
                spec = infer_edit_spec(
                    baseline_code,
                    addon_row.get('nn_code', ''),
                    baseline_prm,
                    addon_row.get('prm'),
                )
                spec = prune_edit_spec(spec, key_dict.get('edit_schema'))
                candidate_score = rank_source_edit_candidate(
                    strategy,
                    spec,
                    baseline_code,
                    baseline_prm=baseline_prm,
                    baseline_accuracy=baseline_row.get('accuracy'),
                    baseline_name=baseline_row.get('nn'),
                    addon_name=addon_row.get('nn'),
                    addon_accuracy=addon_row.get('accuracy'),
                )
            except Exception:
                candidate_score = (float('-inf'), float('-inf'))
            candidate = (candidate_score, addon_row)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            return best[1]

    return available_addon.sample(n=1).iloc[0]


def _filter_cifar_safe_rows(df: DataFrame) -> DataFrame:
    if df is None or df.empty:
        return df
    if 'nn_code' not in df.columns:
        return df
    mask = df.apply(lambda row: not is_cifar_unsafe_seed(row.get('nn_code', ''), row.get('nn')), axis=1)
    return df.loc[mask].reset_index(drop=True)


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        prompt_lists = []

        with open(self.prompts_path) as prompt_file: # /workspace/nn-gpt/ab/gpt/conf/prompt/train/NN_gen.json
            prompt_dict = json.load(prompt_file)
            assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            print('Preparing Data...', flush=True)
            key_dict = prompt_dict[key]
            num_joint_nns = key_dict.get('num_joint_nns') or 1
            data = lemur.data(only_best_accuracy=only_best_accuracy, task=key_dict.get('task'),
                              dataset=key_dict.get('dataset'),
                              nn_prefixes=tuple(key_dict.get('nn_prefixes')), max_rows=n_training_prompts,
                              sql=None if num_joint_nns < 2 else JoinConf(num_joint_nns=num_joint_nns,
                                                                          same_columns=tuple(key_dict.get('keep_same', [])),
                                                                          diff_columns=tuple(key_dict.get('no_repeat', [])),
                                                                          enhance_nn=key_dict.get('improve', False)))
            if is_small_rgb32_dataset(key_dict.get('dataset')):
                before = len(data)
                data = _filter_cifar_safe_rows(data)
                filtered = before - len(data)
                if filtered:
                    print(f'[INFO] Filtered {filtered} CIFAR-unsafe training seeds for key {key}', flush=True)
            print('Data acquisition complete', flush=True)

            addon_task = key_dict.get('addon_task')
            addon_data = lemur.data(
                only_best_accuracy=True,
                task=addon_task,
                dataset=key_dict.get('addon_dataset', key_dict.get('dataset'))
            ) if addon_task else None
            if addon_data is not None and is_small_rgb32_dataset(key_dict.get('addon_dataset', key_dict.get('dataset'))):
                before = len(addon_data)
                addon_data = _filter_cifar_safe_rows(addon_data)
                filtered = before - len(addon_data)
                if filtered:
                    print(f'[INFO] Filtered {filtered} CIFAR-unsafe addon seeds for key {key}', flush=True)

            # Check generation target type
            use_delta = key_dict.get('use_delta', False) or 'delta' in key.lower()
            use_edit = key_dict.get('use_edit', False) or 'edit' in key.lower()

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                # print(f'Row: {row}')
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break
                para_dict = dict()
                para_dict['nn'] = row.get('nn')
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]

                if addon_data is not None and key_dict.get('addon_list'):
                    available_addon = addon_data
                    if 'nn' in addon_data.columns and 'nn' in row:
                        available_addon = available_addon.loc[available_addon.nn != row['nn']]
                    if key_dict.get('prefer_better_addon', True) and 'accuracy' in addon_data.columns and 'accuracy' in row:
                        better_addon = available_addon.loc[available_addon.accuracy > row['accuracy']]
                        if not better_addon.empty:
                            available_addon = better_addon
                    if not available_addon.empty:
                        addon_row = _select_addon_row(available_addon, row, key_dict)
                        para_dict['source_nn'] = addon_row.get('nn')
                        for it in key_dict['addon_list']:
                            para_dict[it['para']] = addon_row[it['value']]

                # Compute structured edit target if edit mode is enabled
                if use_edit and 'addon_nn_code' in para_dict and 'nn_code' in para_dict:
                    try:
                        baseline_code = para_dict.get('nn_code', '')
                        improved_code = para_dict.get('addon_nn_code', '')
                        computed_edit = infer_edit_spec(
                            baseline_code,
                            improved_code,
                            para_dict.get('prm'),
                            para_dict.get('addon_prm'),
                        )
                        computed_edit = prune_edit_spec(computed_edit, key_dict.get('edit_schema'))
                        computed_edit = json.dumps(computed_edit, separators=(',', ':'))
                        para_dict['computed_edit'] = computed_edit

                        output = '\n'.join(prompt_dict[key]['output'])
                        response_para_dict = {k: _stringify_prompt_value(v) for k, v in para_dict.items()}
                        try:
                            response = output.format(**response_para_dict)
                        except KeyError:
                            response = output
                            for k, v in response_para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_edit}', computed_edit)
                    except Exception as e:
                        print(f'[WARNING] Failed to compute structured edit for key {key}: {e}. Using regular output.', flush=True)
                        para_dict['computed_edit'] = ''
                        output = '\n'.join(prompt_dict[key]['output'])
                        response_para_dict = {k: _stringify_prompt_value(v) for k, v in para_dict.items()}
                        try:
                            response = output.format(**response_para_dict)
                        except KeyError:
                            response = output
                            for k, v in response_para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_edit}', '')
                # Compute delta if delta mode is enabled
                elif use_delta and 'addon_nn_code' in para_dict and 'nn_code' in para_dict:
                    try:
                        from ab.gpt.util.DeltaUtil import compute_delta
                        baseline_code = para_dict.get('nn_code', '')
                        improved_code = para_dict.get('addon_nn_code', '')

                        if baseline_code and improved_code:
                            computed_delta = compute_delta(baseline_code, improved_code)
                            # Ensure computed_delta is not None
                            if not computed_delta:
                                computed_delta = ""
                            else:
                                computed_delta = _compact_unified_diff(computed_delta)
                        else:
                            computed_delta = ""
                        para_dict['computed_delta'] = computed_delta

                        # Replace {computed_delta} placeholder in output template
                        # First format with para_dict, then replace placeholder
                        output = '\n'.join(prompt_dict[key]['output'])
                        response_para_dict = {k: _stringify_prompt_value(v) for k, v in para_dict.items()}
                        # Format with para_dict first (may contain other placeholders)
                        try:
                            response = output.format(**response_para_dict)
                        except KeyError:
                            # If formatting fails, use replace for all placeholders
                            response = output
                            for k, v in response_para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        # Always replace computed_delta placeholder (even if empty)
                        response = response.replace('{computed_delta}', computed_delta)
                    except Exception as e:
                        print(f'[WARNING] Failed to compute delta for key {key}: {e}. Using regular output.', flush=True)
                        para_dict['computed_delta'] = ''
                        # Fallback to regular output on error
                        output = '\n'.join(prompt_dict[key]['output'])
                        response_para_dict = {k: _stringify_prompt_value(v) for k, v in para_dict.items()}
                        try:
                            response = output.format(**response_para_dict)
                        except KeyError:
                            response = output
                            for k, v in response_para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        # Replace placeholder with empty string if delta computation failed
                        response = response.replace('{computed_delta}', '')
                else:
                    # Regular mode: use output template as-is
                    output = '\n'.join(prompt_dict[key]['output'])
                    try:
                        response = output.format(**{k: _stringify_prompt_value(v) for k, v in para_dict.items()})
                    except KeyError:
                        response = output
                        for k, v in {k: _stringify_prompt_value(v) for k, v in para_dict.items()}.items():
                            response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_edit}', para_dict.get('computed_edit', ''))
                        response = response.replace('{computed_delta}', para_dict.get('computed_delta', ''))

                prompt_para_dict = _compact_prompt_fields(
                    para_dict,
                    aggressive=False,
                    edit_schema=key_dict.get('edit_schema'),
                    source_edit_policy=key_dict.get('source_edit_policy'),
                )
                try:
                    inst = prompt.format(**prompt_para_dict)
                except KeyError:
                    inst = prompt
                    for k, v in prompt_para_dict.items():
                        inst = inst.replace(f'{{{k}}}', str(v))
                    inst = inst.replace('{computed_edit}', prompt_para_dict.get('computed_edit', ''))
                    inst = inst.replace('{computed_delta}', prompt_para_dict.get('computed_delta', ''))
                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False
                )

                if len(self.tokenizer(text, truncation=False).input_ids) >= self.max_len:
                    prompt_para_dict = _compact_prompt_fields(
                        para_dict,
                        aggressive=True,
                        edit_schema=key_dict.get('edit_schema'),
                        source_edit_policy=key_dict.get('source_edit_policy'),
                    )
                    try:
                        inst = prompt.format(**prompt_para_dict)
                    except KeyError:
                        inst = prompt
                        for k, v in prompt_para_dict.items():
                            inst = inst.replace(f'{{{k}}}', str(v))
                        inst = inst.replace('{computed_edit}', prompt_para_dict.get('computed_edit', ''))
                        inst = inst.replace('{computed_delta}', prompt_para_dict.get('computed_delta', ''))
                    text = self.tokenizer.apply_chat_template(
                        [
                            {'role': 'user', 'content': inst},
                            {'role': 'assistant', 'content': response}
                        ], tokenize=False
                    )

                # print(f"Prompt: {inst}")
                # print(f"Output: {response}")

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data
        return pd.concat(prompt_lists, ignore_index=True)

    @override
    def get_dataset(self, only_best_accuracy=False, seed=None, max_prompts=None, max_new_tokens=4096):
        """
        Return pre-rendered chat text so LoRA/SFT training can use the raw `text` column.
        We still length-filter examples here, but we do not strip `text` away.
        """
        raw_df = self.get_raw_dataset(only_best_accuracy, max_prompts)
        if raw_df.empty:
            return Dataset.from_pandas(raw_df, preserve_index=False)

        def keep_row(row):
            text_ids = self.tokenizer(row['text'], truncation=False).input_ids
            response_ids = self.tokenizer(row['response'], truncation=False).input_ids if 'response' in row else []
            return len(text_ids) < self.max_len and len(response_ids) < max_new_tokens

        filtered_df = raw_df[raw_df.apply(keep_row, axis=1)].reset_index(drop=True)
        dataset = Dataset.from_pandas(filtered_df, preserve_index=False)
        dataset = dataset.shuffle(seed=seed) if seed else dataset.shuffle()
        return dataset

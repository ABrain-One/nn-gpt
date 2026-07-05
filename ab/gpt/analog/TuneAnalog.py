import os
import random
import shutil
import json
import hashlib
import re
import ast
import importlib.util
from os import makedirs
from os.path import isfile
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import ab.nn.api as lemur
from ab.nn.util.Util import release_memory, create_file
from peft import (PeftModel)
from tqdm import tqdm

from ab.gpt.analog import NNEvalAnalog as NNEval
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *

from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.analog.UtilAnalog import (
    exists,
    extract_delta,
    extract_edit,
    extract_code,
    extract_hyperparam,
    extract_transform,
    get_dataset_prompt_defaults,
    get_dataset_smoke_profile,
    is_cifar_unsafe_seed,
    is_cifar_spatial_collapse_error,
    is_small_rgb32_dataset,
    normalize_generated_nn_code,
    parse_hyperparam_text,
    repair_cifar_spatial_collapse,
    validate_generated_nn_smoke,
)
from ab.gpt.analog.NNGenPromptAnalog import NNGenPrompt
from ab.gpt.util.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.EditUtil import (
    apply_source_edit_policy,
    apply_structured_edit,
    build_safe_edit_for_target,
    build_source_edit_hint,
    gate_structured_edit_spec,
    infer_edit_spec,
    parse_edit_text,
    prune_edit_spec,
    prune_source_edit_hint,
    rank_source_edit_candidate,
    score_edit_spec_for_target,
    summarize_cifar_target,
)
from ab.gpt.util.Const import nngpt_upload
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders

# from datasets import load_from_disk


ds_conf = conf_dir / 'DeepSpeed.json'

# Transform dir paths
TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'

# Delta mode constants
_MAX_DELTA_RETRIES = 2
_SKIP_POST_FINETUNE = os.environ.get('AB_GPT_SKIP_POST_FINETUNE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
_STRICT_NO_REPAIR = os.environ.get('AB_GPT_STRICT_NO_REPAIR', '').strip().lower() in {'1', 'true', 'yes', 'on'}
_REPAIR_MODE = os.environ.get('AB_GPT_REPAIR_MODE', '').strip().lower()
_SPATIAL_MINIMAL_REPAIR = _REPAIR_MODE in {'minimal_spatial', 'spatial_minimal'}
_MINIMAL_REPAIR = _REPAIR_MODE in {'minimal', 'mechanical'} or _SPATIAL_MINIMAL_REPAIR
_DISABLE_GENERATED_CODE_NORMALIZATION = _REPAIR_MODE in {'none', 'raw'}
_FULL_REPAIR = (not _STRICT_NO_REPAIR) and (not _MINIMAL_REPAIR) and (not _DISABLE_GENERATED_CODE_NORMALIZATION)
_ALLOW_CIFAR_SPATIAL_REPAIR = _FULL_REPAIR or _SPATIAL_MINIMAL_REPAIR
_GEOMETRY_GUARD = os.environ.get('AB_GPT_GEOMETRY_GUARD', '').strip().lower() in {'1', 'true', 'yes', 'on'}
_EVAL_EPOCH_LIMIT_MINUTES = int(os.environ['AB_GPT_EPOCH_LIMIT_MINUTES']) if os.environ.get('AB_GPT_EPOCH_LIMIT_MINUTES') else None
_ENABLE_EDIT_SAFETY_GATE = (
    os.environ.get('AB_GPT_ENABLE_EDIT_SAFETY_GATE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    and _FULL_REPAIR
)


def _prm_keys_used_by_code(tree) -> set[str]:
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'prm':
                key_node = node.slice
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    keys.add(key_node.value)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'prm'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
    return keys


def _literal_string_sequence(node):
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def _format_supported_hyperparameters(keys, original_node):
    quoted = ', '.join(repr(key) for key in keys)
    if isinstance(original_node, ast.Tuple):
        if len(keys) == 1:
            quoted += ','
        return f'({quoted})'
    if isinstance(original_node, ast.List):
        return f'[{quoted}]'
    if not keys:
        return '[]'
    return '{' + quoted + '}'


def _prune_unused_supported_hyperparameters(code: str) -> str:
    """Keep supported_hyperparameters() aligned with actual prm[...] reads.

    This is a declaration-only cleanup: it does not alter model layers, optimizer
    logic, or training behavior. It prevents the evaluator from rejecting code
    that declares a hyperparameter such as dropout but never consumes it.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    used_keys = _prm_keys_used_by_code(tree)
    if not used_keys:
        return code

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != 'supported_hyperparameters':
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Return):
                continue
            declared = _literal_string_sequence(stmt.value)
            if declared is None:
                return code
            kept = [key for key in declared if key in used_keys]
            if kept == declared:
                return code
            if not hasattr(stmt.value, 'end_lineno') or stmt.value.end_lineno is None:
                return code

            replacement = _format_supported_hyperparameters(kept, stmt.value)
            lines = code.splitlines(keepends=True)
            start_line = stmt.value.lineno - 1
            end_line = stmt.value.end_lineno - 1
            start_col = stmt.value.col_offset
            end_col = stmt.value.end_col_offset
            if start_line == end_line:
                lines[start_line] = (
                    lines[start_line][:start_col]
                    + replacement
                    + lines[start_line][end_col:]
                )
            else:
                lines[start_line] = lines[start_line][:start_col] + replacement + lines[end_line][end_col:]
                for i in range(start_line + 1, end_line + 1):
                    lines[i] = ''
            print(
                f"[INFO] Minimal repair pruned unused supported_hyperparameters: "
                f"{declared} -> {kept}"
            )
            return ''.join(lines)
    return code


def _minimal_repair_generated_nn_code(code: str) -> str:
    """Apply only mechanical runtime fixes used for the defensible repair ablation."""
    if not code:
        return code

    fixed = code

    needs_torch_import = "torch." in fixed or "torch.device" in fixed
    needs_nn_import = "nn." in fixed or "nn.Module" in fixed
    needs_optim_import = "optim." in fixed

    if needs_torch_import and "import torch" not in fixed:
        fixed = "import torch\n" + fixed
    if needs_nn_import and "import torch.nn as nn" not in fixed:
        prefix = "import torch\n" if fixed.startswith("import torch\n") else ""
        body = fixed[len(prefix):]
        fixed = f"{prefix}import torch.nn as nn\n{body}"
    if needs_optim_import and "import torch.optim as optim" not in fixed:
        header = ""
        body = fixed
        if body.startswith("import torch\n"):
            header += "import torch\n"
            body = body[len("import torch\n"):]
        if body.startswith("import torch.nn as nn\n"):
            header += "import torch.nn as nn\n"
            body = body[len("import torch.nn as nn\n"):]
        fixed = f"{header}import torch.optim as optim\n{body}"

    fixed = re.sub(r"\bc_in\s*=\s*in_shape\[0\]", "c_in = in_shape[1]", fixed)

    if "nn.LazyLinear" not in fixed and ("flatten" in fixed.lower() or ".view(" in fixed):
        fixed = re.sub(
            r"nn\.Linear\(\s*([0-9][0-9\s\*\+\-\/]*|[A-Za-z_][A-Za-z0-9_]*\s*[\*\+\-\/][^,\n]+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*|\d+)\s*\)",
            r"nn.LazyLinear(\2)",
            fixed,
            count=1,
        )

    fixed = _prune_unused_supported_hyperparameters(fixed)

    return fixed


def _prepare_generated_code_for_eval(code: str) -> str:
    if not code:
        return code
    if _DISABLE_GENERATED_CODE_NORMALIZATION:
        return code
    if _MINIMAL_REPAIR:
        return _minimal_repair_generated_nn_code(code)
    return normalize_generated_nn_code(code)


def _minify_prompt_block(text):
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


def _compact_prompt_block(text, head_lines, tail_lines=0):
    if not isinstance(text, str):
        return text
    compact = _minify_prompt_block(text)
    lines = compact.splitlines()
    if len(lines) <= head_lines + tail_lines + 1:
        return compact
    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines else []
    return '\n'.join(head + ['# ... prompt-trimmed ...'] + tail)


def _format_prompt_fields(para_dict, edit_schema=None, source_edit_policy=None):
    formatted = dict(para_dict)
    # Older analogical prompt templates use addon_* names, while the frozen
    # source-guided queue passes explicit source_* fields.
    source_to_addon = {
        'source_accuracy': 'addon_accuracy',
        'source_nn_code': 'addon_nn_code',
        'source_transform_code': 'addon_transform_code',
        'source_prm': 'addon_prm',
    }
    for source_key, addon_key in source_to_addon.items():
        if addon_key not in formatted and source_key in formatted:
            formatted[addon_key] = formatted[source_key]
    if 'source_edit' in formatted and source_edit_policy:
        try:
            formatted['source_edit'] = json.dumps(
                apply_source_edit_policy(json.loads(formatted['source_edit']), source_edit_policy),
                separators=(',', ':'),
            )
        except Exception:
            pass
    if 'target_summary' not in formatted and 'nn_code' in para_dict:
        try:
            formatted['target_summary'] = json.dumps(
                summarize_cifar_target(
                    para_dict.get('nn_code', ''),
                    prm=para_dict.get('prm'),
                    model_name=para_dict.get('nn'),
                    accuracy=para_dict.get('accuracy'),
                ),
                separators=(',', ':'),
            )
        except Exception:
            formatted['target_summary'] = '{"has_adaptive_avg":false,"has_classifier_dropout":false,"final_channels":0,"max_channels":0,"init_block_channels":0,"stem_kernel_size":0,"stem_stride":0,"has_stem_pool":false}'
    if 'source_edit' not in formatted and 'source_nn_code' in para_dict and 'nn_code' in para_dict:
        try:
            formatted['source_edit'] = json.dumps(
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
            formatted['source_edit'] = '{"suggested_edit":{"mode":"hp_transform_only","width":"same","init_width":"same","pool":"keep","stem_stride":"keep","stem_pool":"keep","classifier_dropout":0.0},"safe_edit":{"mode":"hp_transform_only","width":"same","init_width":"same","pool":"keep","stem_stride":"keep","stem_pool":"keep","classifier_dropout":0.0},"edit_bias":"safe_edit","source_traits":[],"transfer_focus":[],"target_already_has":[]}'
    for key, value in list(formatted.items()):
        if isinstance(value, dict):
            formatted[key] = json.dumps(value, separators=(',', ':'))
        if key.startswith('source_') and 'nn_code' in key:
            formatted[key] = _compact_prompt_block(formatted[key], 36, 16)
        elif key.startswith('source_') and 'transform_code' in key:
            formatted[key] = _compact_prompt_block(formatted[key], 10)
    return formatted


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _geometry_prm(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = parse_hyperparam_text(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def _filter_hp_copy_recipe(source_prm):
    """Mirror hp_transfer recipe handling: parse/normalize, but do not add a
    separate target-usage whitelist before the shared geometry guard.
    """
    parsed = _geometry_prm(source_prm)
    return _jsonable(parsed) if isinstance(parsed, dict) else {}


def _recipe_delta_payload(source_prm, raw_hp_str, final_hp_str):
    source_recipe = _filter_hp_copy_recipe(source_prm)
    llm_raw_output = _filter_hp_copy_recipe(raw_hp_str)
    candidate_final = _filter_hp_copy_recipe(final_hp_str)

    source_keys = set(source_recipe)
    raw_keys = set(llm_raw_output)
    final_keys = set(candidate_final)
    common_source_final = source_keys & final_keys

    fields_unchanged = sorted(
        key for key in common_source_final
        if source_recipe.get(key) == candidate_final.get(key)
    )
    fields_changed = {
        key: {
            'source': source_recipe.get(key),
            'final': candidate_final.get(key),
        }
        for key in sorted(common_source_final)
        if source_recipe.get(key) != candidate_final.get(key)
    }

    return {
        'source_recipe': source_recipe,
        'llm_raw_output': llm_raw_output,
        'candidate_final': candidate_final,
        'fields_unchanged': fields_unchanged,
        'fields_changed': fields_changed,
        'fields_added_by_llm': sorted(raw_keys - source_keys),
        'fields_dropped_by_validation': sorted(raw_keys - final_keys),
    }


def _write_source_recipe_delta(model_dir, source_prm, raw_hp_str, final_hp_str):
    try:
        payload = _recipe_delta_payload(source_prm, raw_hp_str, final_hp_str)
        create_file(model_dir, 'source_recipe_delta.json', json.dumps(payload, indent=2, sort_keys=True))
    except Exception:
        pass


def _format_hp_copy_output(hp_str, tr_str):
    return (
        '<hp>\n'
        f'{hp_str or "{}"}'
        '\n</hp><tr>\n'
        f'{tr_str or ""}'
        '\n</tr><edit>{"mode":"hp_transform_only"}</edit>'
    )


def _is_large_image_transform(name):
    return str(name or '').strip() in {
        'echo_128',
        'echo_128_flip',
        'echo_224',
        'echo_256',
        'echo_256_flip',
        'echo_299',
        'echo_299_flip',
        'echo_512',
        'echo_512_flip',
        'norm_224',
        'norm_224_flip',
        'norm_256',
        'norm_256_flip',
        'norm_299',
        'norm_299_flip',
        'norm_512',
        'norm_512_flip',
    }


def _apply_geometry_guard(hp_str, tr_str, origdf=None, para_context=None):
    if not _GEOMETRY_GUARD:
        return hp_str, tr_str

    dataset = None
    if origdf is not None:
        dataset = origdf.get('dataset')
    dataset_key = str(dataset or '').lower()

    para_context = para_context or {}
    generated_hp = parse_hyperparam_text(hp_str) or {}
    generated_transform = generated_hp.get('transform')

    target_prm = _geometry_prm(origdf.get('prm') if origdf is not None else None) or _geometry_prm(para_context.get('prm'))
    source_prm = _geometry_prm(para_context.get('source_prm'))
    target_transform = target_prm.get('transform') if isinstance(target_prm, dict) else None
    source_transform = source_prm.get('transform') if isinstance(source_prm, dict) else None

    preferred_prm = source_prm if _is_large_image_transform(source_transform) else target_prm
    preferred_tr = (
        para_context.get('source_transform_code')
        if preferred_prm is source_prm and para_context.get('source_transform_code')
        else para_context.get('transform_code')
    )

    fallback_by_dataset = {
        'celeba-gender': 'echo_224',
        'places365': 'echo_224',
        'imagenette': 'echo_224',
    }
    fallback_transform = fallback_by_dataset.get(dataset_key)

    if fallback_transform:
        fixed_prm = {'transform': fallback_transform}
        fixed_transform = fallback_transform
    elif isinstance(preferred_prm, dict) and _is_large_image_transform(preferred_prm.get('transform')):
        fixed_prm = _jsonable(preferred_prm)
        fixed_transform = preferred_prm.get('transform')
    else:
        return hp_str, tr_str

    compact_tr = str(tr_str or '').replace(' ', '')
    guard_needed = (
        not _is_large_image_transform(generated_transform)
        or str(generated_transform or '').startswith('echo_')
        or bool(tr_str and any(f'Resize(({size}' in compact_tr for size in ('32', '64')))
    )
    if not guard_needed:
        return hp_str, tr_str

    fixed_hp = dict(generated_hp)
    fixed_hp.update(fixed_prm)
    print(
        f"[INFO] Geometry guard replaced transform {generated_transform!r} "
        f"with {fixed_transform!r} for {dataset_key or 'unknown dataset'}"
    )
    return json.dumps(fixed_hp, separators=(',', ':')), preferred_tr or tr_str


def _transform_module_exists(name):
    name = str(name or '').strip()
    if not re.match(r'^[A-Za-z_]\w*$', name):
        return False
    return importlib.util.find_spec(f'ab.nn.transform.{name}') is not None


def _guard_available_transform(hp_str, tr_str, origdf=None):
    hp_obj = parse_hyperparam_text(hp_str)
    if not isinstance(hp_obj, dict):
        return hp_str, tr_str

    generated_transform = hp_obj.get('transform')
    if _transform_module_exists(generated_transform):
        return hp_str, tr_str

    dataset = origdf.get('dataset') if origdf is not None else None
    defaults = get_dataset_prompt_defaults(dataset) or {}
    fallback_transform = defaults.get('default_transform')
    if not fallback_transform or not _transform_module_exists(fallback_transform):
        return hp_str, tr_str

    hp_obj['transform'] = fallback_transform
    print(
        f"[INFO] Replaced unavailable transform {generated_transform!r} "
        f"with {fallback_transform!r} for {dataset or 'unknown dataset'}"
    )
    return json.dumps(hp_obj, separators=(',', ':')), defaults.get('transform_code') or tr_str


def _select_addon_row(available_addon, baseline_row, key_config):
    if available_addon is None or available_addon.empty:
        return None

    exact_addon = _filter_exact_rows(
        available_addon,
        ids=_selector_values(key_config, 'source_ids', 'AB_GPT_SOURCE_IDS')
            or _selector_values(key_config, 'addon_ids', 'AB_GPT_SOURCE_IDS'),
        nns=_selector_values(key_config, 'source_nns', 'AB_GPT_SOURCE_NNS')
            or _selector_values(key_config, 'addon_nns', 'AB_GPT_SOURCE_NNS'),
    )
    if exact_addon is not None and not exact_addon.empty:
        return exact_addon.sort_values('accuracy', ascending=False).iloc[0]

    strategy = key_config.get('addon_selection', 'random')
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
                spec = prune_edit_spec(spec, key_config.get('edit_schema'))
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

    fixed_seed = _fixed_generation_seed()
    if fixed_seed is not None:
        random_state = _stable_seed(fixed_seed, baseline_row.get('nn'), key_config.get('dataset'), key_config.get('task'))
        return available_addon.sample(n=1, random_state=random_state).iloc[0]

    return available_addon.sample(n=1).iloc[0]


def _fixed_generation_seed():
    raw_seed = os.environ.get('AB_GPT_FIXED_TEST_SEED')
    if raw_seed is None or raw_seed == '':
        return None
    try:
        return int(raw_seed)
    except Exception:
        return 0


def _candidate_seed_offset():
    raw_seed = os.environ.get('AB_GPT_CANDIDATE_SEED_OFFSET')
    if raw_seed is None or raw_seed == '':
        return 0
    try:
        return int(raw_seed)
    except Exception:
        return 0


def _stable_seed(base_seed, *parts):
    joined = '|'.join(str(part) for part in parts if part is not None)
    digest = hashlib.sha256(joined.encode('utf-8')).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32 - 1)


def _selector_values(key_config, config_key, env_key):
    raw = None
    if isinstance(key_config, dict):
        raw = key_config.get(config_key)
    if raw is None:
        raw = os.environ.get(env_key)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(v).strip() for v in raw if str(v).strip()]
    return [part.strip() for part in str(raw).split(',') if part.strip()]


def _filter_exact_rows(rows, ids=None, nns=None):
    if rows is None or rows.empty:
        return rows
    ids = ids or []
    nns = nns or []
    filtered = rows
    if ids:
        filtered = filtered.loc[filtered['id'].astype(str).isin(ids)]
    if nns:
        filtered = filtered.loc[filtered['nn'].astype(str).isin(nns)]
    if ids or nns:
        return filtered.reset_index(drop=True)
    return None


def _repeat_generation_rows(rows):
    raw = os.environ.get('AB_GPT_REPEAT_TARGET_N', '').strip()
    if not raw:
        return rows
    try:
        repeat_n = int(raw)
    except Exception:
        repeat_n = 1
    if repeat_n <= 1 or rows is None or rows.empty:
        return rows
    repeated = pd.concat([rows.copy() for _ in range(repeat_n)], ignore_index=True)
    print(f'[INFO] Repeating selected generation rows: {len(rows)} target(s) x {repeat_n} = {len(repeated)} candidate prompts')
    return repeated


def _sample_generation_rows(dataset_rows, test_nn, dataset_name, key_name, key_config=None):
    if dataset_rows is None or dataset_rows.empty:
        return dataset_rows

    exact_targets = _filter_exact_rows(
        dataset_rows,
        ids=_selector_values(key_config, 'target_ids', 'AB_GPT_TARGET_IDS'),
        nns=_selector_values(key_config, 'target_nns', 'AB_GPT_TARGET_NNS'),
    )
    if exact_targets is not None:
        selected = exact_targets.sort_values('accuracy').head(test_nn).reset_index(drop=True)
        if 'nn' in selected.columns:
            print(f'[INFO] Exact target selection for key {key_name}: {list(selected.nn)}')
        return _repeat_generation_rows(selected)

    fixed_seed = _fixed_generation_seed()
    sample_kwargs = {'random_state': fixed_seed} if fixed_seed is not None else {}
    grouped = (
        dataset_rows
        .sort_values(by='nn')
        .groupby(by='nn')
        .sample(n=1, **sample_kwargs)
        .reset_index(drop=True)
    )
    if is_small_rgb32_dataset(dataset_name):
        before = len(grouped)
        grouped = grouped.loc[
            ~grouped.apply(lambda row: is_cifar_unsafe_seed(row.get('nn_code', ''), row.get('nn')), axis=1)
        ].reset_index(drop=True)
        filtered = before - len(grouped)
        if filtered:
            print(f'[INFO] Filtered {filtered} CIFAR-unsafe test seeds for key {key_name}')

    if len(grouped) < test_nn:
        print(f'[WARNING] Requested {test_nn} safe test seeds for key {key_name}, but only {len(grouped)} available. Using all.')
        selected = grouped
        if fixed_seed is not None and 'nn' in selected.columns:
            print(f'[INFO] Fixed test seed {fixed_seed}; selected targets for key {key_name}: {list(selected.nn)}')
        return _repeat_generation_rows(selected)

    selected = grouped.sample(n=test_nn, **sample_kwargs).reset_index(drop=True)
    if fixed_seed is not None and 'nn' in selected.columns:
        print(f'[INFO] Fixed test seed {fixed_seed}; selected targets for key {key_name}: {list(selected.nn)}')
    return _repeat_generation_rows(selected)

def _dataset_prefill_parts(dataset_name=None):
    defaults = get_dataset_prompt_defaults(dataset_name) or {}
    return (
        defaults.get('hp_json', '{"batch":64,"transform":"echo_32","lr":0.01,"momentum":0.9}'),
        defaults.get(
            'transform_code',
            "import torchvision.transforms as transforms\n"
            "def transform(_):\n"
            "    return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])",
        ),
    )


def _delta_assistant_prefill(dataset_name=None):
    hp_json, transform_code = _dataset_prefill_parts(dataset_name)
    return (
        "<hp>\n"
        f"{hp_json}\n"
        "</hp>\n"
        "<tr>\n"
        f"{transform_code}\n"
        "</tr>\n"
        "<delta>\n"
        "--- baseline.py\n"
        "+++ improved.py\n"
    )


def _edit_assistant_prefill(dataset_name=None, mode_seed=False):
    edit_prefix = '{"mode":"' if mode_seed else '{'
    hp_json, transform_code = _dataset_prefill_parts(dataset_name)
    return (
        "<hp>\n"
        f"{hp_json}\n"
        "</hp>\n"
        "<tr>\n"
        f"{transform_code}\n"
        "</tr>\n"
        "<edit>\n"
        f"{edit_prefix}"
    )


def _chat_with_assistant_prefill(chat_bot, prompt_text, assistant_prefill, max_new_tokens=None):
    """Analog-only generation path that preserves a structured assistant prefix.

    The public ChatBot API does not accept assistant_prefill. For structured edit
    experiments, we render the normal user chat prompt, append the assistant
    prefix after the generation marker, and decode only the continuation.
    """
    if not assistant_prefill:
        return chat_bot.chat(prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

    if hasattr(chat_bot.model, "eval"):
        chat_bot.model.eval()

    messages = []
    system_prompt = getattr(chat_bot, "system_prompt", None)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    tokenizer = chat_bot.tokenizer
    if (
        not getattr(chat_bot, "disable_chat_template", False)
        and hasattr(tokenizer, "apply_chat_template")
    ):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
        )
        formatted_prompt = f"{formatted_prompt}\nAssistant:"

    formatted_prompt = formatted_prompt + assistant_prefill
    token_budget = max_new_tokens or 4096
    tokenizer_max_len = getattr(tokenizer, "model_max_length", 4096)
    try:
        tokenizer_max_len = int(tokenizer_max_len)
    except Exception:
        tokenizer_max_len = 4096
    if tokenizer_max_len <= 0 or tokenizer_max_len > 10**8:
        tokenizer_max_len = 4096
    max_input_len = max(1, tokenizer_max_len - token_budget)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(max_input_len, 128),
        add_special_tokens=False,
    )

    if 'input_ids' in inputs:
        input_ids = inputs['input_ids']
        vocab_size = tokenizer.vocab_size
        max_token_id = input_ids.max().item()
        if max_token_id >= vocab_size:
            clamp_value = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else vocab_size - 1
            inputs['input_ids'] = torch.clamp(input_ids, max=clamp_value)

    if hasattr(chat_bot.model, 'device') and chat_bot.model.device is not None:
        device = chat_bot.model.device
    elif getattr(chat_bot, "is_onnx", False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = next(chat_bot.model.parameters()).device
        except StopIteration:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[-1]

    with torch.no_grad():
        outputs = chat_bot.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or 4096,
            do_sample=True,
            temperature=chat_bot.temperature,
            top_k=chat_bot.top_k,
            top_p=chat_bot.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_out = assistant_prefill + generated
    return (
        extract_code(full_out),
        extract_hyperparam(full_out),
        extract_transform(full_out),
        full_out,
    )


def apply_sliding_window(example, max_length, stride, tokenizer):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']

    chunks = []
    for i in range(0, len(input_ids), stride):
        end = i + max_length
        if end <= len(input_ids):
            chunk_input_ids = input_ids[i:end]
            chunk_attention_mask = attention_mask[i:end]

            pad_len = max_length - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [tokenizer.pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len

            chunks.append({
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask
            })
    return {"chunks": chunks}


def flatten_chunks(data):
    all_chunks = sum(data["chunks"], [])  # flatten batched list
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


def _resolve_prompt_config(base_dir: Path, config_name: str) -> Path:
    """Resolve legacy flat prompt configs and isolated analog configs."""
    direct = base_dir / config_name
    if direct.exists():
        return direct
    analog = base_dir / "analog" / config_name
    if analog.exists():
        return analog
    return direct


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf, training_args, peft_config,
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None,
         onnx_run=False, trans_mode=False, prompt_batch=1, use_backbone=False, eval_save_to_db=True, context_length=None):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = False
    base_model_name = config['base_model_name']
    merged_candidate = nngpt_upload / Path(base_model_name).name

    if merged_candidate.exists():
        print(f"[EVOLUTION] Using merged model: {merged_candidate}")
        base_model_name = str(merged_candidate)
    else:
        print(f"[EVOLUTION] Using base model from config: {base_model_name}")


    llm_tune_epochs = int(training_args.num_train_epochs)
    use_deepspeed = training_args.deepspeed is not None
    only_best_accuracy = False
    if context_length is None:
        context_length = config.get('context_length') or config.get('default_context_length')
    unsloth_max_input_length = config.get('max_input_length', None)
    use_unsloth = config.get('use_unsloth', False)
    unsloth_load_in_4bit = config.get('load_in_4bit', True)
    use_backbone = config.get('backbone', use_backbone)

    access_token = None
    if token_from_file:
        with open(ab_root_path / 'token') as f:
            access_token = f.readline()

    print(f'[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {llm_path}')

    train_config_path = _resolve_prompt_config(conf_train_dir, llm_tune_conf)

    # Load test prompts
    with open(_resolve_prompt_config(conf_test_dir, nn_gen_conf)) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    copy_source_recipe_only = all(
        isinstance(prompt_dict.get(key), dict) and prompt_dict[key].get('copy_source_recipe', False)
        for key in conf_keys
    )
    if copy_source_recipe_only:
        print('[INFO] copy_source_recipe config detected; skipping LLM model/tokenizer load.')

    if copy_source_recipe_only:
        model_loader = None
        model = None
        tokenizer = None
        lora_tuner = None
        chat_bot = None
    else:
        from ab.gpt.util.LLM import LLM

        # Load model and tokenizer
        model_loader = LLM(
            base_model_name,
            quantization_config_4bit,
            access_token=access_token,
            use_deepspeed=use_deepspeed,
            context_length=context_length,
            training_args=training_args,
            use_unsloth=use_unsloth,
            load_in_4bit=unsloth_load_in_4bit
        )
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        tokenizer.model_max_length = model_loader.get_max_length()
        # print(model)
        if llm_path:
            print(f'Load saved LoRA layer from path: {llm_path}')
            model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
            model = model.merge_and_unload()

        # initialize deepspeed before we do infer in ChatBot
        if use_deepspeed:
            import deepspeed
            deepspeed.initialize(model=model, config_params=ds_conf)

        lora_tuner = LoRA(
            model,
            tokenizer,
            training_args=training_args,
            access_token=access_token,
            peft_config=peft_config,
            use_unsloth=use_unsloth)

        print('Using Max Length:', model_loader.get_max_length())

        # loop train and eval cycles
        chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p) # Only initialize ONCE

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped generation at epoch {epoch}')
        else:
            if trans_mode:
                trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
            else:
                nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=use_backbone, eval_save_to_db=eval_save_to_db)


        if copy_source_recipe_only or _SKIP_POST_FINETUNE:
            reason = 'copy_source_recipe config' if copy_source_recipe_only else 'AB_GPT_SKIP_POST_FINETUNE'
            print(f'[DEBUG]Skipping post-generation finetune at epoch {epoch} due to {reason}.')
            continue

        # fine tune model for 1 epoch / Using training_args and save copy
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')

        # Select data processor based on mode
        if trans_mode:
            data_processor = TransformGenPrompt(
                context_length if context_length else model_loader.get_max_length(),
                tokenizer,
                train_config_path,
                TRANSFORM_OUT_DIR,
                TRANSFORM_RES_DIR
            )
        elif use_backbone:
             from ab.gpt.util.prompt.SFTGenPrompt import SFTGenPrompt
             data_processor = SFTGenPrompt(
                context_length if context_length else model_loader.get_max_length(),
                tokenizer
             )
        else:
            if not use_unsloth:
                data_processor = NNGenPrompt(context_length if context_length else model_loader.get_max_length(), tokenizer, train_config_path)
            else:
                data_processor = NNGenPrompt(unsloth_max_input_length if unsloth_max_input_length else model_loader.get_max_length(), tokenizer, train_config_path)
        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts, max_new_tokens=max_new_tokens)

        print('Dataset length:', len(dataset))
        if len(dataset) < 2:
            print('[WARNING] Fine-tune dataset has fewer than 2 examples, skipping training for this epoch.')
            del dataset
            release_memory()
            continue
        model.train()
        model = lora_tuner.train(
            dataset,
            tokenizer,
            out_path / base_model_name,
            train_on_completions_only=True,
            response_template="<|im_start|>assistant\n",
        )
        del dataset
        release_memory()


def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=False, eval_save_to_db=True):
    print('Preparing prompts for generation, this might take a while...')
    # Anchor generation in the expected tag format for both raw-code and delta modes.
    assistant_prefill = '<hp>'

    # Detect structured generation mode from nn_name_prefix or config key
    use_delta = nn_name_prefix == 'delta'
    use_edit = nn_name_prefix == 'edit'
    copy_source_recipe = False
    prefill_dataset_name = None
    if isinstance(prompt_dict, dict) and conf_keys:
        first_key = conf_keys[0] if isinstance(conf_keys, (list, tuple)) else conf_keys
        key_config = prompt_dict.get(first_key, {})
        if isinstance(key_config, dict):
            copy_source_recipe = bool(key_config.get('copy_source_recipe', False))
            use_delta = use_delta or key_config.get('use_delta', False) or 'delta' in str(first_key).lower()
            use_edit = use_edit or copy_source_recipe or key_config.get('use_edit', False) or 'edit' in str(first_key).lower()
            prefill_dataset_name = key_config.get('dataset')
            assistant_prefill_mode = key_config.get('assistant_prefill_mode') or os.environ.get('AB_GPT_ASSISTANT_PREFILL_MODE')
        else:
            assistant_prefill_mode = os.environ.get('AB_GPT_ASSISTANT_PREFILL_MODE')
    else:
        assistant_prefill_mode = os.environ.get('AB_GPT_ASSISTANT_PREFILL_MODE')

    structured_mode = use_delta or use_edit

    if use_delta:
        assistant_prefill = _delta_assistant_prefill(prefill_dataset_name)
    elif use_edit:
        assistant_prefill = _edit_assistant_prefill(prefill_dataset_name, mode_seed=True)

    if assistant_prefill_mode == 'none':
        assistant_prefill = None
    elif assistant_prefill_mode == 'hp_open':
        assistant_prefill = '<hp>\n'

    prompts = []
    for key in conf_keys:
        prompt = ''
        key_config = prompt_dict[key]
        prompt_dict_key = key_config
        for pr in prompt_dict_key['prompt']:
            prompt += pr + '\n'
        data = _sample_generation_rows(
            lemur.data(
                only_best_accuracy=True,
                task=prompt_dict_key['task'],
                dataset=prompt_dict_key.get('dataset')
            ),
            test_nn,
            prompt_dict_key.get('dataset'),
            key,
            prompt_dict_key,
        )
        addon_task = prompt_dict_key.get('addon_task')
        addon_data = lemur.data(
            only_best_accuracy=True,
            task=addon_task,
            dataset=prompt_dict_key.get('addon_dataset', prompt_dict_key.get('dataset'))
        ) if addon_task else None
        if addon_data is not None and is_small_rgb32_dataset(prompt_dict_key.get('addon_dataset', prompt_dict_key.get('dataset'))):
            before = len(addon_data)
            addon_data = addon_data.loc[
                ~addon_data.apply(lambda row: is_cifar_unsafe_seed(row.get('nn_code', ''), row.get('nn')), axis=1)
            ].reset_index(drop=True)
            filtered = before - len(addon_data)
            if filtered:
                print(f'[INFO] Filtered {filtered} CIFAR-unsafe addon seeds for key {key}')
        from ab.gpt.analog.UtilAnalog import extract_str
        for _, row in data.iterrows():
            para_dict = dict()
            para_dict['nn'] = row.get('nn')
            for it in prompt_dict_key['input_list']:
                para_dict[it['para']] = row[it['value']]
            para_dict['_log_source_recipe_delta'] = bool(
                prompt_dict_key.get('log_source_recipe_delta')
                or (
                    'hp_transfer' in str(key).lower()
                    and not prompt_dict_key.get('copy_source_recipe', False)
                )
            )
            if addon_data is not None and not addon_data.empty:
                available_addon = addon_data.loc[addon_data.nn != row['nn']]
                if prompt_dict_key.get('prefer_better_addon', True) and 'accuracy' in available_addon.columns and 'accuracy' in row:
                    better_addon = available_addon.loc[available_addon.accuracy > row['accuracy']]
                    if not better_addon.empty:
                        available_addon = better_addon
                if not available_addon.empty:
                    addon_row = _select_addon_row(available_addon, row, prompt_dict_key)
                    para_dict.setdefault('source_nn', addon_row.get('nn'))
                    para_dict.setdefault('source_nn_code', addon_row.get('nn_code'))
                    para_dict.setdefault('source_accuracy', addon_row.get('accuracy'))
                    para_dict.setdefault('source_prm', addon_row.get('prm'))
                    para_dict.setdefault('source_transform_code', addon_row.get('transform_code'))
                    if prompt_dict_key.get('addon_list'):
                        for it in prompt_dict_key['addon_list']:
                            para_dict[it['para']] = addon_row[it['value']]
            prompt_text = prompt.format(**_format_prompt_fields(
                para_dict,
                key_config.get('edit_schema'),
                key_config.get('source_edit_policy'),
            ))
            if structured_mode:
                prompts.append((prompt_text, row, dict(para_dict)))
            else:
                prompts.append((prompt_text, row))

    if copy_source_recipe and len(prompts) > 1:
        print(f'[INFO] hp_copy is deterministic; using 1 candidate instead of {len(prompts)} prompts.')
        prompts = prompts[:1]

    models_dir = synth_dir(out_path)

    # Structured modes: per-sample processing with retry-and-feedback
    if structured_mode:
        for idx, prompt_data in tqdm(enumerate(prompts)):
            model_dir = models_dir / f'B{idx}'
            if len(prompt_data) == 3:
                prompt_text, origdf, para_context = prompt_data
            else:
                prompt_text, origdf = prompt_data
                para_context = {}
            smoke_profile = get_dataset_smoke_profile(origdf.get('dataset') if origdf is not None else None) or {}
            smoke_kwargs = {}
            if smoke_profile:
                smoke_kwargs = {
                    'in_shape': smoke_profile['in_shape'],
                    'out_shape': smoke_profile['out_shape'],
                }

            # Per-sample seed for reproducibility and diversity across epochs
            seed = epoch * 10000 + idx + _candidate_seed_offset()
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if unsloth_max_input_length and not copy_source_recipe:
                in_text = [{"role": "user", "content": prompt_text}]
                token_len = len(chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True))
                print(f'Sample prompt length: {token_len}, max_input_length: {unsloth_max_input_length}')
                if token_len > unsloth_max_input_length:
                    print(f'Prompt is too long, skipping...')
                    continue

            baseline_code = origdf.get('nn_code', '') if origdf is not None else ''
            code = None

            makedirs(model_dir, exist_ok=True)

            # Structured extraction + application with retry-and-feedback
            parsed_edit = None
            raw_parsed_edit = None
            gate_info = None
            if copy_source_recipe:
                source_prm = _geometry_prm(para_context.get('source_prm'))
                source_tr = para_context.get('source_transform_code')
                if not source_prm:
                    print(f'[ERROR] hp_copy requires source_prm for model B{idx}')
                    continue
                if source_tr is None or not str(source_tr).strip():
                    print(f'[ERROR] hp_copy requires source_transform_code for model B{idx}')
                    continue
                source_prm = _filter_hp_copy_recipe(source_prm)
                hp_str = json.dumps(_jsonable(source_prm), separators=(',', ':'))
                tr_str = str(source_tr)
                hp_str, tr_str = _apply_geometry_guard(hp_str, tr_str, origdf, para_context)
                final_out = _format_hp_copy_output(hp_str, tr_str)
                code = baseline_code
                parsed_edit = {'mode': 'hp_transform_only'}
                raw_parsed_edit = dict(parsed_edit)
                print(f'[INFO] hp_copy copied source recipe for B{idx} without LLM generation')
            else:
                # Initial generation
                _, hp, tr, full_out = _chat_with_assistant_prefill(
                    chat_bot,
                    prompt_text,
                    assistant_prefill,
                    max_new_tokens=max_new_tokens,
                )

                if use_backbone:
                    from ab.gpt.util.SFTUtil import skeleton_code
                    import textwrap

                    # Extract full blocks (including signatures)
                    block_code = extract_str(full_out, '<block>', '</block>')
                    init_code = extract_str(full_out, '<init>', '</init>')
                    forward_code = extract_str(full_out, '<forward>', '</forward>')

                    if block_code and init_code and forward_code:
                        code = skeleton_code

                        # Replace skeleton signatures with LLM-provided blocks (including signatures)
                        # Ensure correct indentation for internal methods
                        sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
                        code = code.replace(sig_block, textwrap.dedent(block_code))

                        sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
                        code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))

                        sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
                        code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
                    else:
                        code = extract_code(full_out)

                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)

                current_out = full_out
                current_prompt = prompt_text
                current_assistant_prefill = assistant_prefill
                for attempt in range(_MAX_DELTA_RETRIES + 1):
                    if attempt > 0:
                        _, _, _, current_out = _chat_with_assistant_prefill(
                            chat_bot,
                            current_prompt,
                            current_assistant_prefill,
                            max_new_tokens=max_new_tokens,
                        )

                    if use_edit:
                        edit_str = extract_edit(current_out)
                        if not edit_str:
                            error_msg = 'No valid <edit>...</edit> JSON object found in output.'
                        else:
                            parsed_edit = parse_edit_text(edit_str)
                            if parsed_edit is None:
                                error_msg = 'Structured edit JSON could not be parsed.'
                            else:
                                raw_parsed_edit = dict(parsed_edit)
                                if _ENABLE_EDIT_SAFETY_GATE and baseline_code:
                                    parsed_edit, gate_info = gate_structured_edit_spec(
                                        parsed_edit,
                                        baseline_code,
                                        baseline_prm=origdf.get('prm') if origdf is not None else None,
                                        baseline_accuracy=origdf.get('accuracy') if origdf is not None else None,
                                        baseline_name=origdf.get('nn') if origdf is not None else None,
                                    )
                                    if gate_info and gate_info.get('downgraded'):
                                        print(
                                            f"[INFO] Safety gate downgraded edit for B{idx}: "
                                            f"{gate_info['original_edit']} -> {gate_info['gated_edit']}"
                                        )
                                applied = apply_structured_edit(baseline_code, parsed_edit) if baseline_code else None
                                if applied:
                                    code = applied
                                    print(f'[INFO] Applied structured edit for B{idx} (attempt {attempt + 1})')
                                    break
                                error_msg = 'Structured edit failed to apply cleanly to the baseline code.'
                    else:
                        delta = extract_delta(current_out)
                        if not delta:
                            error_msg = 'No <delta>...</delta> block found in output.'
                        elif not validate_delta(delta):
                            error_msg = 'Delta format is invalid (must be unified diff with --- / +++ headers and @@ hunks).'
                        else:
                            applied = apply_delta(baseline_code, delta) if baseline_code else None
                            if applied:
                                code = applied
                                print(f'[INFO] Applied delta for B{idx} (attempt {attempt + 1})')
                                break
                            else:
                                error_msg = 'Delta patch failed to apply to the baseline code.'

                    if attempt < _MAX_DELTA_RETRIES:
                        mode_name = 'structured edit' if use_edit else 'delta'
                        print(f'[WARNING] {mode_name.capitalize()} attempt {attempt + 1} failed for B{idx}: {error_msg} Retrying with feedback...')
                        continuation_hint = (
                            (
                                '\nContinue directly after the existing {"mode":" prefix with one valid mode value only.'
                                '\nThen finish the same JSON object using only these keys: width, init_width, pool, stem_stride, stem_pool, residual_kernel, classifier_dropout.'
                                '\nDo not repeat <edit>, do not reopen {, and do not write prose.'
                                '\nDo not emit code, markdown, or explanations.'
                            ) if current_assistant_prefill and '{"mode":"' in current_assistant_prefill else (
                                '\nRegenerate exactly three XML blocks: <hp>...</hp>, <tr>...</tr>, and <edit>...</edit>.'
                                '\nThe <edit> block must contain one JSON object and no prose.'
                            )
                        ) if use_edit else (
                            '\nContinue directly after the existing <delta> header with unified diff hunks only.'
                        )
                        if use_edit:
                            current_assistant_prefill = _edit_assistant_prefill(
                                origdf.get('dataset') if origdf is not None else prefill_dataset_name,
                                mode_seed=True,
                            )
                        current_prompt = (
                            prompt_text
                            + f'\n\n[SYSTEM FEEDBACK - Attempt {attempt + 1} failed]: {error_msg}'
                            + '\nDo not write analysis or explanations.'
                            + '\nReuse the existing <hp> and <tr> blocks exactly as prefilled.'
                            + continuation_hint
                        )

                final_out = current_out

            # Syntax-repair fallback when structured generation still emits a raw model body.
            if code is None and _FULL_REPAIR:
                print(f'[WARNING] All structured attempts failed for B{idx}. Trying syntax repair on extracted code.')
                raw_code = extract_code(final_out)
                if raw_code:
                    repaired = repair_code(raw_code)
                    if repaired:
                        code = repaired
                        print(f'[INFO] Used syntax-repaired code fallback for B{idx}')

            # Re-parse hp/tr from saved output
            raw_hp_str = extract_hyperparam(final_out)
            tr_str = extract_transform(final_out)
            hp_str, tr_str = _apply_geometry_guard(raw_hp_str, tr_str, origdf, para_context)
            hp_str, tr_str = _guard_available_transform(hp_str, tr_str, origdf)
            if para_context.get('_log_source_recipe_delta'):
                _write_source_recipe_delta(
                    model_dir,
                    para_context.get('source_prm'),
                    raw_hp_str,
                    hp_str,
                )
            if use_edit and parsed_edit is None:
                parsed_edit = parse_edit_text(extract_edit(final_out))
                raw_parsed_edit = dict(parsed_edit) if parsed_edit is not None else None
                if _ENABLE_EDIT_SAFETY_GATE and parsed_edit is not None and baseline_code:
                    parsed_edit, gate_info = gate_structured_edit_spec(
                        parsed_edit,
                        baseline_code,
                        baseline_prm=origdf.get('prm') if origdf is not None else None,
                        baseline_accuracy=origdf.get('accuracy') if origdf is not None else None,
                        baseline_name=origdf.get('nn') if origdf is not None else None,
                    )
                    if gate_info and gate_info.get('downgraded'):
                        print(
                            f"[INFO] Safety gate downgraded late-parsed edit for B{idx}: "
                            f"{gate_info['original_edit']} -> {gate_info['gated_edit']}"
                        )

            if code is None and baseline_code and _FULL_REPAIR:
                hp_obj = parse_hyperparam_text(hp_str)
                has_transform = tr_str is not None and bool(tr_str.strip())
                if hp_obj is not None or has_transform:
                    code = baseline_code
                    print(f'[INFO] Using baseline code fallback for B{idx}; applying generated hp/transform without code delta')

            repaired_code = _prepare_generated_code_for_eval(code)
            if repaired_code != code:
                repair_label = 'minimal-repaired' if _MINIMAL_REPAIR else 'normalized'
                print(f'[INFO] {repair_label.capitalize()} generated code for B{idx} before evaluation')
            code = repaired_code

            if code:
                smoke_ok, smoke_error = validate_generated_nn_smoke(code, **smoke_kwargs)
                if not smoke_ok:
                    recovered = False
                    if is_cifar_spatial_collapse_error(smoke_error) and _ALLOW_CIFAR_SPATIAL_REPAIR:
                        repaired_collapse_code = repair_cifar_spatial_collapse(code)
                        if repaired_collapse_code:
                            if _FULL_REPAIR:
                                repaired_collapse_code = normalize_generated_nn_code(repaired_collapse_code)
                            else:
                                repaired_collapse_code = _minimal_repair_generated_nn_code(repaired_collapse_code)
                            repaired_collapse_ok, repaired_collapse_error = validate_generated_nn_smoke(repaired_collapse_code, **smoke_kwargs)
                            if repaired_collapse_ok:
                                code = repaired_collapse_code
                                recovered = True
                                print(f'[INFO] Applied CIFAR collapse repair for B{idx} after smoke failure: {smoke_error}')
                            else:
                                print(f'[WARNING] CIFAR collapse repair still failed smoke test for B{idx}: {repaired_collapse_error}')
                    if not recovered and use_edit and parsed_edit is not None and baseline_code and _ENABLE_EDIT_SAFETY_GATE and raw_parsed_edit is not None and _FULL_REPAIR:
                        safe_edit, safe_meta = build_safe_edit_for_target(
                            raw_parsed_edit,
                            baseline_code,
                            baseline_prm=origdf.get('prm') if origdf is not None else None,
                            baseline_accuracy=origdf.get('accuracy') if origdf is not None else None,
                            baseline_name=origdf.get('nn') if origdf is not None else None,
                        )
                        if safe_edit != parsed_edit:
                            safe_code = apply_structured_edit(baseline_code, safe_edit)
                            if safe_code:
                                safe_code = normalize_generated_nn_code(safe_code)
                                safe_ok, safe_error = validate_generated_nn_smoke(safe_code, **smoke_kwargs)
                                if safe_ok:
                                    code = safe_code
                                    parsed_edit = safe_edit
                                    gate_info = gate_info or {}
                                    gate_info.update({
                                        'downgraded': True,
                                        'downgrade_reason': f'smoke_failed:{smoke_error}',
                                        'target_already_has': safe_meta.get('target_already_has', []),
                                        'original_edit': raw_parsed_edit,
                                        'gated_edit': safe_edit,
                                        'original_score': round(float(score_edit_spec_for_target(
                                            raw_parsed_edit,
                                            baseline_code,
                                            baseline_prm=origdf.get('prm') if origdf is not None else None,
                                            baseline_accuracy=origdf.get('accuracy') if origdf is not None else None,
                                            baseline_name=origdf.get('nn') if origdf is not None else None,
                                        )), 4),
                                        'gated_score': round(float(score_edit_spec_for_target(
                                            safe_edit,
                                            baseline_code,
                                            baseline_prm=origdf.get('prm') if origdf is not None else None,
                                            baseline_accuracy=origdf.get('accuracy') if origdf is not None else None,
                                            baseline_name=origdf.get('nn') if origdf is not None else None,
                                        )), 4),
                                        'safe_edit': safe_edit,
                                    })
                                    print(f'[INFO] Safety gate downgraded smoke-failing edit for B{idx}: {raw_parsed_edit} -> {safe_edit}')
                                    recovered = True
                                else:
                                    print(f'[WARNING] Safe fallback edit also failed smoke test for B{idx}: {safe_error}')
                    if not recovered:
                        print(f'[WARNING] Structured edit smoke test failed for B{idx}: {smoke_error}')
                        if baseline_code and _FULL_REPAIR:
                            baseline_fallback_code = normalize_generated_nn_code(baseline_code)
                            baseline_ok, baseline_error = validate_generated_nn_smoke(baseline_fallback_code, **smoke_kwargs)
                            if not baseline_ok and is_cifar_spatial_collapse_error(baseline_error):
                                repaired_baseline = repair_cifar_spatial_collapse(baseline_fallback_code)
                                if repaired_baseline:
                                    repaired_baseline = normalize_generated_nn_code(repaired_baseline)
                                    repaired_baseline_ok, repaired_baseline_error = validate_generated_nn_smoke(repaired_baseline, **smoke_kwargs)
                                    if repaired_baseline_ok:
                                        baseline_fallback_code = repaired_baseline
                                        baseline_ok = True
                                        print(f'[INFO] Applied CIFAR collapse repair to baseline fallback for B{idx}')
                                    else:
                                        print(f'[WARNING] Repaired baseline fallback still failed smoke test for B{idx}: {repaired_baseline_error}')
                            if baseline_ok:
                                code = baseline_fallback_code
                                parsed_edit = None
                                print(f'[INFO] Falling back to baseline architecture for B{idx} after failed edit smoke test')

            try:
                print(f'Generated params: {hp_str}')
                hp_obj = parse_hyperparam_text(hp_str)
                if hp_obj is not None:
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp_obj, f)
                else:
                    print('[WARNING] No hyperparameters generated, skipping hp file')
            except Exception as e:
                print(f'[WARNING] Error processing hyperparameters: {e}')

            try:
                print(f'Generated transformer:\n\n{tr_str}\n----\n')
                if tr_str is not None and tr_str.strip():
                    create_file(model_dir, transformer_file, tr_str)
                else:
                    print('[WARNING] No transformer code generated')
            except Exception as e:
                print(f'[WARNING] Error saving transformer: {e}')

            if use_edit and parsed_edit is not None:
                try:
                    create_file(model_dir, 'edit.json', json.dumps(parsed_edit, indent=2, sort_keys=True))
                    if raw_parsed_edit is not None and raw_parsed_edit != parsed_edit:
                        create_file(model_dir, 'edit_raw.json', json.dumps(raw_parsed_edit, indent=2, sort_keys=True))
                    if gate_info is not None:
                        create_file(model_dir, 'edit_gate.json', json.dumps(gate_info, indent=2, sort_keys=True))
                except Exception as e:
                    print(f'[WARNING] Error saving structured edit: {e}')

            if code is not None and code.strip():
                create_file(model_dir, new_nn_file, code)
                print(f'[INFO] Saved code to {model_dir / new_nn_file}')
            else:
                print(f'[ERROR] No code generated for model B{idx}')
                continue

            create_file(model_dir, new_out_file, final_out)
            df_file = model_dir / 'dataframe.df'
            if origdf is None:
                if isfile(df_file):
                    os.remove(df_file)
                    print(f'[DEBUG]Removed unmatched file: {df_file}')
            else:
                create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                origdf.to_pickle(df_file)

    # Standard mode: batch processing
    else:
        pending = []
        for idx, prompt_data in tqdm(enumerate(prompts)):
            prompt, origdf = prompt_data

            if unsloth_max_input_length:
                in_text = [{"role": "user", "content": prompt}]
                output = chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
                print(f'Sample prompt length: {len(output)}, max_input_length: {unsloth_max_input_length}')
                if len(output) > unsloth_max_input_length:
                    print(f'Prompt is too long, skipping...')
                    continue

            pending.append((idx, prompt, origdf))

        if prompt_batch < 1:
            prompt_batch = 1
        if prompt_batch > 1:
            print(f'[INFO] Batch generation enabled: prompt_batch={prompt_batch}')

        for start in range(0, len(pending), prompt_batch):
            batch = pending[start:start + prompt_batch]
            batch_prompts = [item[1] for item in batch]

            if assistant_prefill:
                batch_outputs = [
                    _chat_with_assistant_prefill(
                        chat_bot,
                        p,
                        assistant_prefill,
                        max_new_tokens=max_new_tokens,
                    )
                    for p in batch_prompts
                ]
            elif prompt_batch > 1 and hasattr(chat_bot, 'chat_batch'):
                batch_outputs = chat_bot.chat_batch(batch_prompts, engineer_prompt=False, max_new_tokens=max_new_tokens)
            else:
                batch_outputs = [chat_bot.chat(p, engineer_prompt=False, max_new_tokens=max_new_tokens) for p in batch_prompts]

            for (idx, prompt, origdf), output in zip(batch, batch_outputs):
                model_dir = models_dir / f'B{idx}'
                code, hp, tr, full_out = output
                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)
                makedirs(model_dir, exist_ok=True)

                repaired_code = _prepare_generated_code_for_eval(code)
                if repaired_code != code:
                    repair_label = 'minimal-repaired' if _MINIMAL_REPAIR else 'normalized'
                    print(f'[INFO] {repair_label.capitalize()} generated code for B{idx} before evaluation')
                code = repaired_code
                hp, tr = _guard_available_transform(hp, tr, origdf)

                try:
                    print(f'Generated params: {hp}')
                    hp_obj = parse_hyperparam_text(hp)
                    if hp_obj is not None:
                        with open(model_dir / hp_file, 'w+') as f:
                            json.dump(hp_obj, f)
                    else:
                        print('[WARNING] No hyperparameters generated, skipping hp file')
                except Exception as e:
                    print(f'[WARNING] Error processing hyperparameters: {e}')

                try:
                    print(f'Generated transformer:\n\n{tr}\n----\n')
                    if tr is not None and tr.strip():
                        create_file(model_dir, transformer_file, tr)
                    else:
                        print('[WARNING] No transformer code generated')
                except Exception as e:
                    print(f'[WARNING] Error saving transformer: {e}')

                if code is not None and code.strip():
                    create_file(model_dir, new_nn_file, code)
                    print(f'[INFO] Saved code to {model_dir / new_nn_file}')
                else:
                    print(f'[ERROR] No code generated for model B{idx}')
                    continue
                create_file(model_dir, new_out_file, full_out)
                df_file = model_dir / 'dataframe.df'
                if origdf is None:
                    if isfile(df_file):
                        os.remove(df_file)
                        print(f'[DEBUG]Removed unmatched file: {df_file}')
                else:
                    create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                    origdf.to_pickle(df_file)

    print('[DEBUG] Release memory.')
    release_memory()
    if exists(models_dir):
        NNEval.main(
            nn_name_prefix,
            nn_train_epochs,
            epoch,
            save_to_db=eval_save_to_db,
            epoch_limit_minutes=_EVAL_EPOCH_LIMIT_MINUTES,
        )
        print('[DEBUG] Release_memory.')
        release_memory()
    print('Clear LEMUR query cache.')
    lemur.data.cache_clear()
    print('The cache has been cleared.')


def trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    """
    Transform Script Generation
    """
    print('Running Transform Generation...')

    out_gen_dir = str(TRANSFORM_OUT_DIR)
    result_gen_dir = str(TRANSFORM_RES_DIR)

    prompts = []

    # Load all data from folders to be used for seed prompts
    all_data = load_data_from_folders(out_gen_dir, result_gen_dir, only_best_accuracy=True)
    if len(all_data) == 0:
        print("Warning: No data loaded from folders for generation. Skipping.", flush=True)
        return

    for key in conf_keys:
        prompt_config = prompt_dict_global[key]
        prompt = ''
        for pr in prompt_config['prompt']:
            prompt += pr + '\n'

        # Get seed data
        if len(all_data) < test_nn:
            print(f"Warning: Requested {test_nn} samples, but only {len(all_data)} available. Using all.", flush=True)
            data_sample = all_data.sample(n=len(all_data))
        else:
            data_sample = all_data.sample(n=test_nn)

        addon_data = all_data

        for _, row in data_sample.iterrows():
            para_dict = dict()
            row_dict = row.to_dict()
            for it in prompt_config['input_list']:
                para_dict[it['para']] = row_dict.get(it['value'])

            # Avoid sampling the same transform
            filtered_addon_data = addon_data.loc[addon_data.id_name != row['id_name']]
            if len(filtered_addon_data) > 0:
                addon_row = filtered_addon_data.sample(n=1).iloc[0].to_dict()
                if prompt_config.get('addon_list'):
                    for it in prompt_config['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])
                prompts.append((prompt.format(**para_dict), row))
            else:
                print(f"Warning: Could not find addon data for {row['id_name']}. Skipping prompt.", flush=True)

    models_dir = synth_dir(out_path)

    for idx, prompt_data in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt_data

        code, hp, tr, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)

        if save_llm_output: create_file(model_dir, new_out_file, full_out)
        makedirs(model_dir, exist_ok=True)

        if tr is not None and tr.strip():
            print(f'Generated transformer:\n\n{tr}\n----\n')
            create_file(model_dir, transformer_file, tr)
        else:
            print(f'[ERROR] No code generated for model B{idx}')
            continue

        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):
                os.remove(df_file)
        else:
            create_file(model_dir, f"original_{origdf['id_name']}.py", origdf['transform_code'])
            origdf.to_pickle(df_file)

    print('[DEBUG] Release memory.')
    release_memory()

    # Evaluate produced CV models
    if exists(models_dir):
        try:
            run_eval(epoch_num=epoch, FT_MODE=True)
        except Exception as e:
            print(f"Error running evaluation main(): {e}", flush=True)

        print('[DEBUG] Release_memory.')
        release_memory()

    print('Folder data reload will occur next epoch.')

import ast
import json
import re


def _round_channels(value: float) -> int:
    value = max(16, int(round(value / 8.0) * 8))
    return value


def _flatten_channel_values(value):
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_channel_values(item))
        return out
    return []


def _scale_channel_values(value, scale: float):
    if isinstance(value, int):
        return _round_channels(value * scale)
    if isinstance(value, list):
        return [_scale_channel_values(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_channel_values(item, scale) for item in value)
    raise ValueError(f'Unsupported channel container: {type(value)!r}')


def _extract_channel_list(code: str):
    match = re.search(r'(?m)^\s*channels\s*=\s*(\[[^\n]+\])\s*$', code)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return None
    values = _flatten_channel_values(parsed)
    return values if values else None


def _extract_init_block_channels(code: str):
    match = re.search(r'init_block_channels\s*=\s*(\d+)', code or '')
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _net_init_region(code: str):
    if not code:
        return ''
    start = code.find('class Net')
    if start < 0:
        return code
    end = code.find('\n    def forward', start)
    if end < 0:
        end = min(len(code), start + 4000)
    return code[start:end]


def _extract_first_net_conv_config(code: str):
    region = _net_init_region(code)
    match = re.search(r'nn\.Conv2d\([^\n]*?kernel_size\s*=\s*(\d+)[^\n]*?stride\s*=\s*(\d+)', region)
    if not match:
        return {'kernel_size': 0, 'stride': 0}
    try:
        kernel_size = int(match.group(1))
    except Exception:
        kernel_size = 0
    try:
        stride = int(match.group(2))
    except Exception:
        stride = 0
    return {'kernel_size': kernel_size, 'stride': stride}


def _has_stem_pool(code: str):
    region = _net_init_region(code)
    return re.search(r'nn\.(?:MaxPool2d|AvgPool2d)\(', region) is not None


def model_family(model_name):
    if not model_name:
        return None
    return str(model_name).split('-', 1)[0]


def summarize_cifar_target(code: str, prm=None, model_name=None, accuracy=None):
    channels = _extract_channel_list(code or '') or []
    stem = _extract_first_net_conv_config(code or '')
    summary = {
        'has_adaptive_avg': 'AdaptiveAvgPool2d' in (code or ''),
        'has_adaptive_max': 'AdaptiveMaxPool2d' in (code or ''),
        'has_classifier_dropout': 'Dropout' in (code or ''),
        'has_batchnorm': 'BatchNorm' in (code or ''),
        'has_attention': ('attention' in (code or '').lower()) or ('Multihead' in (code or '')),
        'has_se': ('Squeeze-and-Excitation' in (code or '')) or ('SE attention' in (code or '')),
        'has_residual': ('residual' in (code or '').lower()) or ('downsample' in (code or '')) or ('identity' in (code or '')),
        'final_channels': channels[-1] if channels else 0,
        'max_channels': max(channels) if channels else 0,
        'init_block_channels': _extract_init_block_channels(code or ''),
        'stem_kernel_size': stem['kernel_size'],
        'stem_stride': stem['stride'],
        'has_stem_pool': _has_stem_pool(code or ''),
    }
    family = model_family(model_name)
    if family:
        summary['family'] = family
    try:
        if accuracy is not None:
            summary['target_accuracy'] = round(float(accuracy), 4)
    except Exception:
        pass
    try:
        if isinstance(prm, dict) and 'dropout' in prm:
            summary['hp_dropout'] = round(float(prm['dropout']), 4)
    except Exception:
        pass
    return summary


def _summary_traits(summary: dict):
    traits = []
    if summary.get('has_adaptive_avg') and summary.get('has_adaptive_max'):
        traits.append('dual_pool')
    elif summary.get('has_adaptive_avg'):
        traits.append('adaptive_avg')
    if summary.get('has_classifier_dropout'):
        traits.append('dropout')
    if summary.get('has_batchnorm'):
        traits.append('batchnorm')
    if summary.get('has_attention'):
        traits.append('attention')
    if summary.get('has_se'):
        traits.append('se')
    if summary.get('has_residual'):
        traits.append('residual')
    stem_stride = summary.get('stem_stride') or 0
    if stem_stride and stem_stride <= 1:
        traits.append('gentle_stem')
    elif stem_stride >= 2:
        traits.append('aggressive_stem')
    if summary.get('has_stem_pool'):
        traits.append('stem_pool')
    else:
        traits.append('no_stem_pool')
    init_channels = summary.get('init_block_channels') or 0
    if init_channels >= 96:
        traits.append('wide_stem')
    final_channels = summary.get('final_channels') or summary.get('max_channels') or 0
    if final_channels >= 512:
        traits.append('wide_tail')
    elif final_channels and final_channels <= 256:
        traits.append('narrow_tail')
    return traits


def _finalize_transfer_edit(spec: dict):
    spec = normalize_edit_spec(spec or {})
    residual_kernel = spec.get('residual_kernel')
    structural = (
        spec['width'] != 'same'
        or spec['init_width'] != 'same'
        or spec['pool'] != 'keep'
        or spec['stem_stride'] != 'keep'
        or spec['stem_pool'] != 'keep'
        or residual_kernel in (1, 2, 3)
        or spec.get('classifier_dropout', 0.0) > 0
    )
    if not structural:
        return {
            'mode': 'hp_transform_only',
            'width': 'same',
            'init_width': 'same',
            'pool': 'keep',
            'stem_stride': 'keep',
            'stem_pool': 'keep',
            'classifier_dropout': 0.0,
        }

    spec['mode'] = 'structured_cifar_edit'
    return spec


def _build_safe_transfer_edit(suggested_edit: dict, target_summary: dict):
    safe_edit = dict(normalize_edit_spec(suggested_edit or {}))
    target_already_has = []
    final_channels = target_summary.get('final_channels') or target_summary.get('max_channels') or 0
    init_channels = target_summary.get('init_block_channels') or 0
    stem_stride = target_summary.get('stem_stride') or 0

    if target_summary.get('has_adaptive_avg') and safe_edit.get('pool') == 'adaptive_avg':
        safe_edit['pool'] = 'keep'
        target_already_has.append('adaptive_avg')

    if final_channels >= 512 and safe_edit.get('width') in {'wide', 'xwide'}:
        safe_edit['width'] = 'same'
        target_already_has.append('wide_tail')

    if init_channels >= 96 and safe_edit.get('init_width') == 'wide':
        safe_edit['init_width'] = 'same'
        target_already_has.append('wide_stem')

    if stem_stride and stem_stride <= 1 and safe_edit.get('stem_stride') == 'preserve':
        safe_edit['stem_stride'] = 'keep'
        target_already_has.append('gentle_stem')

    if not target_summary.get('has_stem_pool') and safe_edit.get('stem_pool') == 'remove':
        safe_edit['stem_pool'] = 'keep'
        target_already_has.append('no_stem_pool')

    if target_summary.get('has_classifier_dropout') and safe_edit.get('classifier_dropout', 0.0) > 0:
        safe_edit['classifier_dropout'] = 0.0
        target_already_has.append('classifier_dropout')

    return _finalize_transfer_edit(safe_edit), target_already_has


def build_source_edit_hint(
    baseline_code: str,
    improved_code: str,
    baseline_prm=None,
    improved_prm=None,
    baseline_name=None,
    improved_name=None,
    baseline_accuracy=None,
    improved_accuracy=None,
):
    suggested_edit = infer_edit_spec(baseline_code, improved_code, baseline_prm, improved_prm)
    target_summary = summarize_cifar_target(
        baseline_code,
        prm=baseline_prm,
        model_name=baseline_name,
        accuracy=baseline_accuracy,
    )
    source_summary = summarize_cifar_target(
        improved_code,
        prm=improved_prm,
        model_name=improved_name,
        accuracy=improved_accuracy,
    )
    safe_edit, target_already_has = _build_safe_transfer_edit(suggested_edit, target_summary)

    transfer_focus = []
    target_traits = _summary_traits(target_summary)
    source_traits = _summary_traits(source_summary)

    if model_family(baseline_name) and model_family(baseline_name) == model_family(improved_name):
        transfer_focus.append('same_family_upgrade')
    if source_summary.get('has_classifier_dropout') and not target_summary.get('has_classifier_dropout'):
        transfer_focus.append('add_head_regularization')
    if (source_summary.get('stem_stride') or 0) < (target_summary.get('stem_stride') or 0):
        transfer_focus.append('gentler_stem')
    if target_summary.get('has_stem_pool') and not source_summary.get('has_stem_pool'):
        transfer_focus.append('remove_stem_pool')
    if source_summary.get('has_batchnorm') and not target_summary.get('has_batchnorm'):
        transfer_focus.append('more_normalization')
    if source_summary.get('has_attention') and not target_summary.get('has_attention'):
        transfer_focus.append('stronger_feature_mixing')
    if source_summary.get('has_adaptive_max') and source_summary.get('has_adaptive_avg') and not target_summary.get('has_adaptive_max'):
        transfer_focus.append('richer_pooling_head')

    try:
        b_acc = float(baseline_accuracy)
        s_acc = float(improved_accuracy)
        if s_acc - b_acc >= 0.15:
            transfer_focus.append('large_accuracy_gap')
    except Exception:
        pass

    aggressive_score = score_edit_spec_for_target(
        suggested_edit,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
        addon_name=improved_name,
    )
    safe_score = score_edit_spec_for_target(
        safe_edit,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
        addon_name=improved_name,
    )

    hint = {
        'suggested_edit': suggested_edit,
        'safe_edit': safe_edit,
        'edit_bias': 'safe_edit' if safe_score > aggressive_score else 'suggested_edit',
        'target_family': target_summary.get('family'),
        'source_family': source_summary.get('family'),
        'source_accuracy': source_summary.get('target_accuracy'),
        'source_traits': source_traits[:4],
        'transfer_focus': transfer_focus[:4],
        'target_already_has': target_already_has[:3],
    }
    if target_summary.get('family'):
        hint['target_accuracy'] = target_summary.get('target_accuracy')
        hint['target_traits'] = target_traits[:4]
    return hint


def apply_source_edit_policy(hint: dict, policy: str | None = None):
    if not isinstance(hint, dict) or not policy:
        return hint

    updated = dict(hint)
    if policy == 'prefer_safe':
        updated['edit_bias'] = 'safe_edit'
    elif policy == 'prefer_suggested':
        updated['edit_bias'] = 'suggested_edit'
    return updated


def _strip_edit_wrapper(text: str):
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*(?:<\|im_start\|>\s*)?assistant\s*>?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*assistant\s*:?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```json', '').replace('```python', '').replace('```', '').strip()
    return cleaned.strip()


def _iter_json_object_strings(text: str):
    start_positions = [idx for idx, ch in enumerate(text) if ch == '{']
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield text[start:idx + 1]
                    break


def _parse_edit_candidate(text: str):
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    return normalize_edit_spec(parsed)


def extract_valid_edit_json(text: str):
    cleaned = _strip_edit_wrapper(text)
    if not cleaned:
        return None

    direct = _parse_edit_candidate(cleaned)
    if direct is not None:
        return json.dumps(direct, separators=(',', ':'))

    for candidate in _iter_json_object_strings(cleaned):
        parsed = _parse_edit_candidate(candidate)
        if parsed is not None:
            return json.dumps(parsed, separators=(',', ':'))

    return None


def normalize_edit_text(text: str):
    return extract_valid_edit_json(text)


def parse_edit_text(text: str):
    normalized = extract_valid_edit_json(text)
    if not normalized:
        return None
    return _parse_edit_candidate(normalized)


def normalize_edit_spec(spec: dict):
    normalized = {
        'mode': str(spec.get('mode', 'hp_transform_only')),
        'width': str(spec.get('width', 'same')),
        'init_width': str(spec.get('init_width', 'same')),
        'pool': str(spec.get('pool', 'keep')),
        'stem_stride': str(spec.get('stem_stride', 'keep')),
        'stem_pool': str(spec.get('stem_pool', 'keep')),
    }

    residual_kernel = spec.get('residual_kernel')
    if residual_kernel in (1, 2, 3):
        normalized['residual_kernel'] = int(residual_kernel)

    classifier_dropout = spec.get('classifier_dropout', 0.0)
    try:
        classifier_dropout = float(classifier_dropout)
    except Exception:
        classifier_dropout = 0.0
    normalized['classifier_dropout'] = max(0.0, min(0.5, classifier_dropout))

    if normalized['mode'] not in {'hp_transform_only', 'structured_cifar_edit'}:
        normalized['mode'] = 'hp_transform_only'
    if normalized['width'] not in {'same', 'narrow', 'wide', 'xwide'}:
        normalized['width'] = 'same'
    if normalized['init_width'] not in {'same', 'narrow', 'wide'}:
        normalized['init_width'] = 'same'
    if normalized['pool'] not in {'keep', 'adaptive_avg'}:
        normalized['pool'] = 'keep'
    if normalized['stem_stride'] not in {'keep', 'preserve'}:
        normalized['stem_stride'] = 'keep'
    if normalized['stem_pool'] not in {'keep', 'remove'}:
        normalized['stem_pool'] = 'keep'

    return normalized


def prune_edit_spec(spec: dict, edit_schema=None):
    normalized = normalize_edit_spec(spec or {})
    if edit_schema not in {'v1_5', 'v1_core'}:
        return normalized

    pruned = {
        'mode': normalized['mode'],
        'width': 'wide' if normalized['width'] == 'xwide' else normalized['width'],
        'pool': normalized['pool'],
        'classifier_dropout': normalized['classifier_dropout'],
    }
    if edit_schema == 'v1_5':
        pruned['stem_stride'] = normalized['stem_stride']
        pruned['stem_pool'] = normalized['stem_pool']
    if normalized.get('residual_kernel') in (1, 2, 3):
        pruned['residual_kernel'] = normalized['residual_kernel']
    return normalize_edit_spec(pruned)


def prune_source_edit_hint(hint: dict, edit_schema=None):
    if not isinstance(hint, dict):
        return hint
    if edit_schema not in {'v1_5', 'v1_core'}:
        return hint

    pruned = dict(hint)
    if 'suggested_edit' in pruned:
        pruned['suggested_edit'] = prune_edit_spec(pruned['suggested_edit'], edit_schema)
    if 'safe_edit' in pruned:
        pruned['safe_edit'] = prune_edit_spec(pruned['safe_edit'], edit_schema)
    return pruned


def rank_source_edit_candidate(
    strategy: str,
    spec: dict,
    baseline_code: str,
    baseline_prm=None,
    baseline_accuracy=None,
    baseline_name=None,
    addon_name=None,
    addon_accuracy=None,
):
    spec = normalize_edit_spec(spec or {})
    target_score = score_edit_spec_for_target(
        spec,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
        addon_name=addon_name,
    )

    try:
        source_accuracy = float(addon_accuracy)
    except Exception:
        source_accuracy = float('-inf')

    if strategy == 'best_safe_edit':
        return (target_score, source_accuracy)

    if strategy == 'best_risky_edit':
        try:
            baseline_acc = float(baseline_accuracy)
            accuracy_gap = max(0.0, source_accuracy - baseline_acc)
        except Exception:
            accuracy_gap = 0.0

        aggressive_score = score_edit_spec(spec)
        structural_bonus = 0.0
        if spec.get('mode') == 'structured_cifar_edit':
            structural_bonus += 1.5
        else:
            structural_bonus -= 1.0
        if spec.get('width') == 'wide':
            structural_bonus += 0.5
        elif spec.get('width') == 'xwide':
            structural_bonus += 1.0
        if spec.get('pool') == 'adaptive_avg':
            structural_bonus += 0.5
        if spec.get('stem_stride') == 'preserve':
            structural_bonus += 0.5
        if spec.get('stem_pool') == 'remove':
            structural_bonus += 0.5
        if spec.get('residual_kernel') in {1, 2, 3}:
            structural_bonus += 0.25

        risky_score = aggressive_score + structural_bonus + 6.0 * accuracy_gap
        return (risky_score, target_score, source_accuracy)

    return (source_accuracy,)


def score_edit_spec(spec: dict) -> float:
    spec = normalize_edit_spec(spec or {})
    score = 0.0

    if spec['mode'] == 'structured_cifar_edit':
        score += 2.0
    if spec['width'] == 'xwide':
        score += 2.5
    elif spec['width'] == 'wide':
        score += 2.0
    elif spec['width'] == 'same':
        score += 0.5
    else:
        score -= 0.5

    if spec['init_width'] == 'wide':
        score += 1.0
    elif spec['init_width'] == 'narrow':
        score -= 0.25

    if spec['pool'] == 'adaptive_avg':
        score += 1.5
    elif spec['width'] == 'same':
        score -= 1.0

    if spec['stem_stride'] == 'preserve':
        score += 1.0
    if spec['stem_pool'] == 'remove':
        score += 1.0

    dropout = spec.get('classifier_dropout', 0.0)
    if 0.05 <= dropout <= 0.2:
        score += 1.5
    elif 0.0 <= dropout < 0.05:
        score += 0.5
    elif dropout > 0.35:
        score -= 1.0

    residual_kernel = spec.get('residual_kernel')
    if residual_kernel is None:
        score += 0.5
    elif residual_kernel == 1:
        score += 0.25
    else:
        score -= 0.5

    if (
        spec['mode'] == 'structured_cifar_edit'
        and spec['width'] == 'same'
        and spec['init_width'] == 'same'
        and spec['pool'] == 'keep'
        and spec['stem_stride'] == 'keep'
        and spec['stem_pool'] == 'keep'
    ):
        score -= 1.5

    return score


def score_edit_spec_for_target(spec: dict, baseline_code: str, baseline_prm=None, baseline_accuracy=None, baseline_name=None, addon_name=None) -> float:
    spec = normalize_edit_spec(spec or {})
    score = score_edit_spec(spec)
    summary = summarize_cifar_target(
        baseline_code,
        prm=baseline_prm,
        model_name=baseline_name,
        accuracy=baseline_accuracy,
    )
    final_channels = summary.get('final_channels') or summary.get('max_channels') or 0
    has_adaptive_avg = bool(summary.get('has_adaptive_avg'))
    stem_stride = summary.get('stem_stride') or 0
    has_stem_pool = bool(summary.get('has_stem_pool'))
    init_channels = summary.get('init_block_channels') or 0

    if has_adaptive_avg and final_channels >= 512:
        if spec['mode'] == 'structured_cifar_edit' and spec['width'] in {'wide', 'xwide'}:
            score -= 1.5 if spec['width'] == 'wide' else 2.25
        if spec['mode'] == 'structured_cifar_edit' and spec['pool'] == 'adaptive_avg':
            score -= 0.75
        if spec['mode'] == 'hp_transform_only':
            score += 0.75
    elif (not has_adaptive_avg) or (final_channels and final_channels <= 256):
        if spec['mode'] == 'structured_cifar_edit':
            score += 0.75
        if spec['width'] in {'wide', 'xwide'}:
            score += 0.5 if spec['width'] == 'wide' else 0.75
        if spec['pool'] == 'adaptive_avg':
            score += 0.5

    if stem_stride >= 2:
        if spec['stem_stride'] == 'preserve':
            score += 1.25
        if spec['stem_pool'] == 'remove' and has_stem_pool:
            score += 1.0
    elif stem_stride and stem_stride <= 1:
        if spec['stem_stride'] == 'preserve':
            score -= 1.0
        if spec['stem_pool'] == 'remove' and not has_stem_pool:
            score -= 0.75

    if has_stem_pool and spec['stem_pool'] == 'remove':
        score += 0.5
    elif not has_stem_pool and spec['stem_pool'] == 'remove':
        score -= 0.5

    if init_channels >= 96 and spec['init_width'] == 'wide':
        score -= 0.75
    elif init_channels and init_channels <= 64 and spec['init_width'] == 'wide':
        score += 0.5

    try:
        accuracy = float(baseline_accuracy)
    except Exception:
        accuracy = None

    if accuracy is not None:
        if accuracy >= 0.75:
            if spec['mode'] == 'hp_transform_only':
                score += 0.75
            if spec['mode'] == 'structured_cifar_edit' and spec['width'] in {'wide', 'xwide'}:
                score -= 0.75 if spec['width'] == 'wide' else 1.25
            if spec['init_width'] == 'wide':
                score -= 0.5
        elif accuracy <= 0.6:
            if spec['mode'] == 'structured_cifar_edit':
                score += 0.75
            if spec['width'] in {'wide', 'xwide'}:
                score += 0.25 if spec['width'] == 'wide' else 0.5
            if spec['init_width'] == 'wide':
                score += 0.25

    if model_family(addon_name) and model_family(addon_name) == model_family(baseline_name):
        score += 0.75

    return score


def build_safe_edit_for_target(
    spec: dict,
    baseline_code: str,
    baseline_prm=None,
    baseline_accuracy=None,
    baseline_name=None,
):
    normalized = normalize_edit_spec(spec or {})
    target_summary = summarize_cifar_target(
        baseline_code,
        prm=baseline_prm,
        model_name=baseline_name,
        accuracy=baseline_accuracy,
    )
    safe_edit, target_already_has = _build_safe_transfer_edit(normalized, target_summary)
    return safe_edit, {
        'target_summary': target_summary,
        'target_already_has': target_already_has,
    }


def gate_structured_edit_spec(
    spec: dict,
    baseline_code: str,
    baseline_prm=None,
    baseline_accuracy=None,
    baseline_name=None,
):
    normalized = normalize_edit_spec(spec or {})
    safe_edit, safe_meta = build_safe_edit_for_target(
        normalized,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
    )
    target_already_has = safe_meta['target_already_has']

    original_score = score_edit_spec_for_target(
        normalized,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
    )
    safe_score = score_edit_spec_for_target(
        safe_edit,
        baseline_code,
        baseline_prm=baseline_prm,
        baseline_accuracy=baseline_accuracy,
        baseline_name=baseline_name,
    )

    downgrade = (
        normalized.get('mode') == 'structured_cifar_edit'
        and safe_edit != normalized
        and (
            safe_score >= original_score + 0.75
            or original_score < 2.5
        )
    )

    gated = safe_edit if downgrade else normalized
    gate_info = {
        'downgraded': downgrade,
        'original_score': round(float(original_score), 4),
        'gated_score': round(float(safe_score if downgrade else original_score), 4),
        'target_already_has': target_already_has,
        'original_edit': normalized,
        'gated_edit': gated,
        'safe_edit': safe_edit,
    }
    return gated, gate_info


def infer_edit_spec(baseline_code: str, improved_code: str, baseline_prm=None, improved_prm=None):
    if not baseline_code or not improved_code:
        return {
            'mode': 'hp_transform_only',
            'width': 'same',
            'init_width': 'same',
            'pool': 'keep',
            'stem_stride': 'keep',
            'stem_pool': 'keep',
            'classifier_dropout': 0.0,
        }

    edit = {
        'mode': 'hp_transform_only',
        'width': 'same',
        'init_width': 'same',
        'pool': 'keep',
        'stem_stride': 'keep',
        'stem_pool': 'keep',
        'classifier_dropout': 0.0,
    }

    baseline_channels = _extract_channel_list(baseline_code)
    improved_channels = _extract_channel_list(improved_code)
    if baseline_channels and improved_channels and baseline_channels[-1] > 0:
        ratio = improved_channels[-1] / baseline_channels[-1]
        if ratio >= 1.4:
            edit['width'] = 'xwide'
        elif ratio >= 1.2:
            edit['width'] = 'wide'
        elif ratio <= 0.85:
            edit['width'] = 'narrow'

    baseline_init = _extract_init_block_channels(baseline_code)
    improved_init = _extract_init_block_channels(improved_code)
    if baseline_init > 0 and improved_init > 0:
        init_ratio = improved_init / baseline_init
        if init_ratio >= 1.2:
            edit['init_width'] = 'wide'
        elif init_ratio <= 0.85:
            edit['init_width'] = 'narrow'

    if 'AdaptiveAvgPool2d' in improved_code and 'AdaptiveAvgPool2d' not in baseline_code:
        edit['pool'] = 'adaptive_avg'

    baseline_stem = _extract_first_net_conv_config(baseline_code)
    improved_stem = _extract_first_net_conv_config(improved_code)
    if baseline_stem['stride'] and improved_stem['stride'] and improved_stem['stride'] < baseline_stem['stride']:
        edit['stem_stride'] = 'preserve'

    if _has_stem_pool(baseline_code) and not _has_stem_pool(improved_code):
        edit['stem_pool'] = 'remove'

    if 'self.downsample' in baseline_code and 'self.downsample' in improved_code:
        if 'kernel_size=1' in improved_code and 'kernel_size=1' not in baseline_code:
            edit['residual_kernel'] = 1

    improved_dropout = None
    if isinstance(improved_prm, dict) and 'dropout' in improved_prm:
        try:
            improved_dropout = float(improved_prm['dropout'])
        except Exception:
            improved_dropout = None

    if 'Dropout' in improved_code and 'Dropout' not in baseline_code:
        edit['classifier_dropout'] = max(edit['classifier_dropout'], improved_dropout or 0.2)
    elif improved_dropout is not None and improved_dropout > 0:
        edit['classifier_dropout'] = max(edit['classifier_dropout'], min(0.5, improved_dropout))

    if any(
        edit.get(key) not in ('same', 'keep', 0.0, None)
        for key in ('width', 'init_width', 'pool', 'stem_stride', 'stem_pool', 'classifier_dropout', 'residual_kernel')
    ):
        edit['mode'] = 'structured_cifar_edit'

    return normalize_edit_spec(edit)


def _replace_channel_scale(code: str, scale: float):
    changed = False

    def repl_channels(match):
        nonlocal changed
        try:
            parsed = ast.literal_eval(match.group(2))
            scaled = _scale_channel_values(parsed, scale)
        except Exception:
            return match.group(0)
        changed = True
        return f"{match.group(1)}{repr(scaled)}"

    code = re.sub(r'(?m)^(\s*channels\s*=\s*)(\[[^\n]+\])\s*$', repl_channels, code, count=1)

    def repl_init(match):
        nonlocal changed
        value = int(match.group(1))
        scaled = _round_channels(value * scale)
        changed = True
        return f"init_block_channels = {scaled}"

    code = re.sub(r'init_block_channels\s*=\s*(\d+)', repl_init, code, count=1)
    return code, changed


def _replace_init_scale(code: str, scale: float):
    changed = False

    def repl_init(match):
        nonlocal changed
        value = int(match.group(1))
        scaled = _round_channels(value * scale)
        changed = True
        return f"init_block_channels = {scaled}"

    code = re.sub(r'init_block_channels\s*=\s*(\d+)', repl_init, code, count=1)
    return code, changed


def _replace_pool(code: str):
    changed = False

    def repl(match):
        nonlocal changed
        changed = True
        return f"{match.group(1)} = nn.AdaptiveAvgPool2d((1, 1))"

    code = re.sub(
        r'(self\.\w*pool\w*)\s*=\s*nn\.[A-Za-z]+Pool2d\([^\n]*\)',
        repl,
        code,
    )
    return code, changed


def _replace_first_net_conv_stride(code: str, stride: int):
    start = code.find('class Net')
    if start < 0:
        return code, False
    region = code[start:]
    match = re.search(r'nn\.Conv2d\([^\n]*?stride\s*=\s*\d+[^\n]*\)', region)
    if not match:
        return code, False
    updated, n = re.subn(r'stride\s*=\s*\d+', f'stride={stride}', match.group(0), count=1)
    if n == 0:
        return code, False
    abs_start = start + match.start()
    abs_end = start + match.end()
    return code[:abs_start] + updated + code[abs_end:], True


def _remove_stem_pool(code: str):
    start = code.find('class Net')
    if start < 0:
        return code, False
    region = code[start:]
    match = re.search(r'nn\.(?:MaxPool2d|AvgPool2d)\([^\n]*\)', region)
    if not match:
        return code, False
    abs_start = start + match.start()
    abs_end = start + match.end()
    return code[:abs_start] + 'nn.Identity()' + code[abs_end:], True


def _replace_residual_kernel(code: str, kernel_size: int):
    match = re.search(r'self\.downsample\s*=\s*\((?P<body>.*?)\)\s*if', code, flags=re.DOTALL)
    if not match:
        return code, False

    body = match.group('body')
    new_body, n = re.subn(r'kernel_size\s*=\s*\d+', f'kernel_size={kernel_size}', body, count=1)
    if n == 0:
        return code, False

    if kernel_size == 1:
        if 'padding=' in new_body:
            new_body = re.sub(r'padding\s*=\s*\d+', 'padding=0', new_body, count=1)
    elif 'padding=' in new_body:
        new_body = re.sub(r'padding\s*=\s*\d+', f'padding={kernel_size // 2}', new_body, count=1)

    return code[:match.start('body')] + new_body + code[match.end('body'):], True


def _insert_classifier_dropout(code: str, dropout: float):
    if dropout <= 0:
        return code, False

    changed = False
    dropout_line = f"        self.classifier_dropout = nn.Dropout({dropout:.3f})\n"

    if 'self.classifier_dropout' in code:
        code, n = re.subn(
            r'self\.classifier_dropout\s*=\s*nn\.Dropout\([^\n]*\)',
            f'self.classifier_dropout = nn.Dropout({dropout:.3f})',
            code,
            count=1,
        )
        return code, n > 0

    classifier_match = re.search(r'(?m)^(?P<indent>\s*)self\.(classifier|output|fc\d*)\s*=\s*nn\.(Linear|Sequential)', code)
    if classifier_match:
        insert_at = classifier_match.start()
        code = code[:insert_at] + dropout_line + code[insert_at:]
        changed = True

    patterns = [
        r'(?m)^(?P<indent>\s*)return\s+self\.(classifier|output|fc\d*)\((?P<arg>[A-Za-z_][A-Za-z0-9_]*)\)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match and 'self.classifier_dropout(' not in code:
            indent = match.group('indent')
            arg = match.group('arg')
            repl = (
                f"{indent}{arg} = self.classifier_dropout({arg})\n"
                f"{indent}return {match.group(0).strip()[7:]}"
            )
            code = code[:match.start()] + repl + code[match.end():]
            changed = True
            break

    if not changed:
        match = re.search(r'(?m)^(?P<indent>\s*)(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*torch\.flatten\((?P<expr>.*)\)\s*$', code)
        if match and 'self.classifier_dropout(' not in code:
            indent = match.group('indent')
            var = match.group('var')
            insertion = f"{match.group(0)}\n{indent}{var} = self.classifier_dropout({var})"
            code = code[:match.start()] + insertion + code[match.end():]
            changed = True

    return code, changed


def apply_structured_edit(baseline_code: str, edit_spec: dict):
    if not baseline_code:
        return None

    spec = normalize_edit_spec(edit_spec)
    if spec['mode'] == 'hp_transform_only':
        return baseline_code

    code = baseline_code
    changed = False

    width_scale = {'narrow': 0.75, 'same': 1.0, 'wide': 1.25, 'xwide': 1.5}[spec['width']]
    if width_scale != 1.0:
        code, local_changed = _replace_channel_scale(code, width_scale)
        changed = changed or local_changed

    init_scale = {'narrow': 0.75, 'same': 1.0, 'wide': 1.25}[spec['init_width']]
    if init_scale != 1.0:
        code, local_changed = _replace_init_scale(code, init_scale)
        changed = changed or local_changed

    if spec['pool'] == 'adaptive_avg':
        code, local_changed = _replace_pool(code)
        changed = changed or local_changed

    if spec['stem_stride'] == 'preserve':
        code, local_changed = _replace_first_net_conv_stride(code, 1)
        changed = changed or local_changed

    if spec['stem_pool'] == 'remove':
        code, local_changed = _remove_stem_pool(code)
        changed = changed or local_changed

    if spec.get('residual_kernel') in (1, 2, 3):
        code, local_changed = _replace_residual_kernel(code, spec['residual_kernel'])
        changed = changed or local_changed

    if spec['classifier_dropout'] > 0:
        code, local_changed = _insert_classifier_dropout(code, spec['classifier_dropout'])
        changed = changed or local_changed

    if not changed:
        return baseline_code

    try:
        ast.parse(code)
    except Exception:
        return None

    return code

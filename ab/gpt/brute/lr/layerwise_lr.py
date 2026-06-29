"""
Layerwise Learning Rate Model Generator — ~1000 CV Models

Reads actual nn-dataset architecture source files and injects layerwise
learning rate grouping into train_setup(), replacing self.parameters() with
parameter groups that carry different lr multipliers per depth slice.

Generates models for 3 datasets: imagenette, cifar-10, cifar-100
Target: ~1000 trainable model directories.

Math: 29 archs × 12 strategies × 3 datasets = 1044 potential; ~1000 after skips.

Usage:
    python -m ab.gpt.brute.lr.layerwise_lr
"""

import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from ab.gpt.brute.lr.schedulers import (
    _find_method_range,
    read_architecture_source,
)


def find_nn_source_dir() -> Path:
    """
    Locate the nn-dataset architecture source directory using package metadata
    only — no module import, so torchvision is never touched.
    """
    spec = importlib.util.find_spec('ab.nn.nn')
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    # Fallback: derive from the ab.nn package __init__ path
    spec = importlib.util.find_spec('ab.nn')
    if spec is not None and spec.origin:
        nn_dir = Path(spec.origin).parent / 'nn'
        if nn_dir.exists():
            return nn_dir
    raise RuntimeError(
        "Cannot locate ab.nn.nn source directory. Is nn-dataset installed?"
    )


def get_supported_hp_from_source(source_code: str):
    """
    Extract the return value of supported_hyperparameters() by static AST
    parsing — no module import, no torch/torchvision dependency.
    Returns a set/frozenset/list, or None if parsing fails.
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == 'supported_hyperparameters'
            ):
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        try:
                            return ast.literal_eval(stmt.value)
                        except (ValueError, TypeError):
                            pass
                        # Handle frozenset([...]) / set([...]) calls
                        val = stmt.value
                        if isinstance(val, ast.Call) and val.args:
                            try:
                                inner = ast.literal_eval(val.args[0])
                                if isinstance(inner, (list, tuple, set)):
                                    return set(inner)
                            except (ValueError, TypeError):
                                pass
                # Found the function but couldn't parse its return — return
                # an empty set so the arch is not skipped for this reason alone.
                return set()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Architectures — mirrors schedulers.py exclusions:
#   RLFN, SwinIR          – no source files
#   ConvNeXtTransformer   – very custom learn() that doesn't use self.parameters()
#   MoE-hetero4-*         – complex / composite model
#   ComplexNet            – complex model (user-excluded)
# ---------------------------------------------------------------------------
ARCHITECTURES = [
    'AirNet', 'AirNext', 'AlexNet', 'BagNet', 'BayesianNet-1',
    'ConvNeXt', 'DPN107', 'DPN131', 'DPN68',
    'DarkNet', 'DenseNet', 'Diffuser', 'EfficientNet', 'FractalNet',
    'GoogLeNet', 'ICNet', 'InceptionV3-1', 'MNASNet', 'MaxVit',
    'MobileNetV2', 'MobileNetV3',
    'RegNet', 'ResNet', 'ShuffleNet', 'SqueezeNet-1', 'SwinTransformer',
    'UNet2D', 'VGG', 'VisionTransformer',
]  # 29 architectures

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = [
    {
        'name': 'imagenette',
        'task': 'img-classification',
        'metric': 'acc',
        'transform': 'norm_256_flip',
    },
    {
        'name': 'cifar-10',
        'task': 'img-classification',
        'metric': 'acc',
        'transform': 'norm_256_flip',
    },
    {
        'name': 'cifar-100',
        'task': 'img-classification',
        'metric': 'acc',
        'transform': 'norm_256_flip',
    },
]

# ---------------------------------------------------------------------------
# Layerwise LR strategies
#
# 'type' == 'uniform' → write original source unchanged (control group).
# 'type' == 'n_group' → split named_parameters() by split_ratios and apply
#                        per-group lr = base_lr * multiplier.
#
# split_ratios must sum to 1.0; len(split_ratios) == len(multipliers) == n_groups.
# ---------------------------------------------------------------------------
LAYERWISE_STRATEGIES = [
    {
        'name': 'llr_uniform',
        'type': 'uniform',
        'n_groups': 1,
        'multipliers': [1.0],
        'split_ratios': [1.0],
        'description': 'Control: all layers share the same LR.',
    },
    # ── 2-group strategies ──────────────────────────────────────────────────
    {
        'name': 'llr_2grp_01x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.5, 0.5],
        'description': '50/50 split: early layers at 0.1× LR, later at 1.0×.',
    },
    {
        'name': 'llr_2grp_001x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.01, 1.0],
        'split_ratios': [0.5, 0.5],
        'description': '50/50 split: early layers at 0.01× LR, later at 1.0×.',
    },
    {
        'name': 'llr_2grp_75pct',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.75, 0.25],
        'description': '75/25 split: first 75% of params at 0.1×, top 25% at 1.0×.',
    },
    {
        'name': 'llr_2grp_25pct',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.25, 0.75],
        'description': '25/75 split: first 25% of params at 0.1×, rest 75% at 1.0×.',
    },
    # ── 3-group strategies ──────────────────────────────────────────────────
    {
        'name': 'llr_3grp_lin',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.1, 0.3, 1.0],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups: 0.1×, 0.3×, 1.0× (linear decay from head).',
    },
    {
        'name': 'llr_3grp_exp',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.01, 0.1, 1.0],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups: 0.01×, 0.1×, 1.0× (exponential decay).',
    },
    {
        'name': 'llr_3grp_top20',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.01, 0.1, 1.0],
        'split_ratios': [0.4, 0.4, 0.2],
        'description': '40/40/20 split: top-20% head gets full LR, deep layers 0.01×.',
    },
    {
        'name': 'llr_3grp_warm',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.1, 0.5, 1.0],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups: 0.1×, 0.5×, 1.0× (gentle linear ramp).',
    },
    # ── 4-group strategies ──────────────────────────────────────────────────
    {
        'name': 'llr_4grp_lin',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.1, 0.3, 0.6, 1.0],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups: 0.1×, 0.3×, 0.6×, 1.0×.',
    },
    {
        'name': 'llr_4grp_exp',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.001, 0.01, 0.1, 1.0],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups: 0.001×, 0.01×, 0.1×, 1.0× (steep decay).',
    },
    # ── 5-group strategy ────────────────────────────────────────────────────
    {
        'name': 'llr_5grp_exp',
        'type': 'n_group',
        'n_groups': 5,
        'multipliers': [0.001, 0.005, 0.01, 0.1, 1.0],
        'split_ratios': [0.2, 0.2, 0.2, 0.2, 0.2],
        'description': '5 equal groups: exponential LR decay backbone→head.',
    },
]  # 12 strategies × 29 archs × 3 datasets = 1044 potential model dirs

# ---------------------------------------------------------------------------
# Architecture-specific HP defaults (copied from schedulers.py for consistency)
# ---------------------------------------------------------------------------
ARCH_EXTRA_HP_DEFAULTS = {
    'ConvNeXt': {'stochastic_depth_prob': 0.1, 'norm_eps': 1e-6, 'norm_std': 0.02},
    'GoogLeNet': {'dropout_aux': 0.7},
    'MaxVit': {'attention_dropout': 0.0, 'stochastic_depth_prob': 0.1},
    'MobileNetV3': {'norm_eps': 0.001, 'norm_momentum': 0.01},
    'SwinTransformer': {'attention_dropout': 0.0, 'stochastic_depth_prob': 0.1},
    'VisionTransformer': {'attention_dropout': 0.0, 'patch_size': 0.5},
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_grouping_code(strategy: dict) -> str:
    """
    Return the Python source lines (indented for a method body at 8-space indent)
    that build _llr_groups — the per-group parameter list passed to the optimizer.
    """
    name = strategy['name']
    split_ratios = strategy['split_ratios']
    multipliers = strategy['multipliers']

    def _fmt_float(v):
        s = f'{v:.10g}'
        return s

    ratios_str = '[' + ', '.join(_fmt_float(r) for r in split_ratios) + ']'
    mults_str = '[' + ', '.join(_fmt_float(m) for m in multipliers) + ']'

    lines = [
        f"        # Layerwise LR strategy: {name}",
        f"        _llr_params = list(self.named_parameters())",
        f"        _llr_n = len(_llr_params)",
        f"        _llr_ratios = {ratios_str}",
        f"        _llr_mults = {mults_str}",
        f"        _llr_groups = []",
        f"        _llr_start = 0",
        f"        for _llr_i, (_llr_r, _llr_m) in enumerate(zip(_llr_ratios, _llr_mults)):",
        f"            if _llr_i < len(_llr_ratios) - 1:",
        f"                _llr_size = max(1, round(_llr_n * _llr_r))",
        f"            else:",
        f"                _llr_size = _llr_n - _llr_start",
        f"            _llr_end = min(_llr_start + _llr_size, _llr_n)",
        f"            if _llr_start < _llr_n:",
        f"                _llr_groups.append({{'params': [p for _, p in _llr_params[_llr_start:_llr_end]], 'lr': prm.get('lr', 0.01) * _llr_m}})",
        f"            _llr_start = _llr_end",
    ]
    return '\n'.join(lines)


def _find_optimizer_block(lines, ts_start, ts_end):
    """
    Within a train_setup method, find the line range of the self.optimizer = ...
    assignment (which may span multiple lines due to parentheses).

    Returns (opt_line_idx, opt_end_idx) or None.
    """
    for i in range(ts_start, ts_end):
        if 'self.optimizer' in lines[i] and '=' in lines[i]:
            # Track parenthesis depth to find end of multi-line call
            paren_depth = 0
            found_open = False
            opt_end_idx = i
            for k in range(i, ts_end):
                for ch in lines[k]:
                    if ch == '(':
                        paren_depth += 1
                        found_open = True
                    elif ch == ')':
                        paren_depth -= 1
                if found_open and paren_depth == 0:
                    opt_end_idx = k
                    break
            return i, opt_end_idx
    return None


def inject_layerwise_lr(source_code: str, strategy: dict):
    """
    Inject layerwise LR grouping into train_setup().

    For 'uniform' strategies, return source_code unchanged.
    For 'n_group' strategies:
      1. Find train_setup method.
      2. Find self.optimizer = ... block.
      3. Verify self.parameters() is in that block.
      4. Insert _llr_groups construction code before the optimizer line.
      5. Replace self.parameters() with _llr_groups.

    Returns modified source string, or None if injection is not possible.
    """
    if strategy['type'] == 'uniform':
        return source_code

    lines = source_code.split('\n')

    # Find class Net
    class_indent = 0
    found_class = False
    for line in lines:
        m = re.match(r'^(\s*)class Net\b', line)
        if m:
            class_indent = len(m.group(1))
            found_class = True
            break
    if not found_class:
        return None

    # Find train_setup method
    ts_range = _find_method_range(lines, 'train_setup', class_indent)
    if ts_range is None:
        return None
    ts_start, ts_end = ts_range

    # Find optimizer block
    result = _find_optimizer_block(lines, ts_start, ts_end)
    if result is None:
        return None
    opt_line_idx, opt_end_idx = result

    # Optimizer block as a single string
    opt_block_lines = lines[opt_line_idx:opt_end_idx + 1]
    opt_block = '\n'.join(opt_block_lines)

    # Must contain self.parameters() to substitute
    if 'self.parameters()' not in opt_block:
        return None

    # Build grouping code lines
    grouping_code = _build_grouping_code(strategy)

    # Replace self.parameters() → _llr_groups (first occurrence only)
    new_opt_block = opt_block.replace('self.parameters()', '_llr_groups', 1)

    # Assemble new source
    new_lines = (
        lines[:opt_line_idx]
        + grouping_code.split('\n')
        + new_opt_block.split('\n')
        + lines[opt_end_idx + 1:]
    )
    code = '\n'.join(new_lines)

    # Validate Python syntax
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    return code


def build_hp_dict(arch: str, dataset_cfg: dict) -> dict:
    """Build the hp.txt JSON payload for a model."""
    hp = {
        'lr': 0.01,
        'batch': 64,
        'dropout': 0.2,
        'momentum': 0.9,
        'transform': dataset_cfg['transform'],
        'epoch_max': 50,
    }
    if arch in ARCH_EXTRA_HP_DEFAULTS:
        hp.update(ARCH_EXTRA_HP_DEFAULTS[arch])
    return hp


def write_dataframe_df(model_dir: Path, arch: str, strategy: dict, dataset_cfg: dict, hp_dict: dict):
    """
    Write dataframe.df so NNEval resolves the correct task/dataset/metric
    for this model directory without requiring CLI overrides.
    """
    series = pd.Series({
        'nn': f"{arch}-{strategy['name']}",
        'task': dataset_cfg['task'],
        'dataset': dataset_cfg['name'],
        'metric': dataset_cfg['metric'],
        'prm': hp_dict,
    })
    series.to_pickle(str(model_dir / 'dataframe.df'))


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_models(output_base_dir: str, prefix: str = 'llr') -> int:
    """
    Generate layerwise-LR model variants and write them to output_base_dir.

    Directory layout per model:
        <prefix>_NNNN/
            new_nn.py       – modified architecture code
            hp.txt          – JSON hyperparameter dict
            model_meta.txt  – human-readable metadata
            dataframe.df    – pandas Series pickle (task/dataset/metric for NNEval)

    Returns the number of model directories successfully written.
    """
    src_dir = find_nn_source_dir()
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    model_idx = 1
    total_generated = 0
    total_skipped = 0
    arch_stats: dict = {}

    print(f"Source directory  : {src_dir}")
    print(f"Output directory  : {output_base}")
    print(f"Architectures     : {len(ARCHITECTURES)}")
    print(f"Strategies        : {len(LAYERWISE_STRATEGIES)}")
    print(f"Datasets          : {len(DATASETS)}")
    max_possible = len(ARCHITECTURES) * len(LAYERWISE_STRATEGIES) * len(DATASETS)
    print(f"Max model dirs    : {max_possible}")
    print()

    for arch in ARCHITECTURES:
        source_code = read_architecture_source(src_dir, arch)
        if source_code is None:
            print(f"[SKIP] {arch}: source file not found")
            total_skipped += len(LAYERWISE_STRATEGIES) * len(DATASETS)
            arch_stats[arch] = {'generated': 0, 'skipped': len(LAYERWISE_STRATEGIES) * len(DATASETS)}
            continue

        # Validate supported_hyperparameters() exists via AST — no module import needed.
        arch_hp = get_supported_hp_from_source(source_code)
        if arch_hp is None:
            print(f"[SKIP] {arch}: supported_hyperparameters() not found in source")
            total_skipped += len(LAYERWISE_STRATEGIES) * len(DATASETS)
            arch_stats[arch] = {'generated': 0, 'skipped': len(LAYERWISE_STRATEGIES) * len(DATASETS)}
            continue

        arch_gen = 0
        arch_skip = 0

        for strategy in LAYERWISE_STRATEGIES:
            # Produce modified code (shared across all 3 datasets)
            model_code = inject_layerwise_lr(source_code, strategy)
            if model_code is None:
                reason = 'no self.parameters() or injection failed'
                print(f"  [SKIP] {arch} / {strategy['name']}: {reason}")
                arch_skip += len(DATASETS)
                total_skipped += len(DATASETS)
                continue

            # One model directory per dataset
            for dataset_cfg in DATASETS:
                model_name = f"{prefix}_{model_idx:04d}"
                model_dir = output_base / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                # new_nn.py
                (model_dir / 'new_nn.py').write_text(model_code, encoding='utf-8')

                # hp.txt
                hp_dict = build_hp_dict(arch, dataset_cfg)
                (model_dir / 'hp.txt').write_text(
                    json.dumps(hp_dict, indent=2), encoding='utf-8'
                )

                # model_meta.txt
                meta_lines = [
                    f"architecture: {arch}",
                    f"strategy: {strategy['name']}",
                    f"strategy_type: {strategy['type']}",
                    f"n_groups: {strategy['n_groups']}",
                    f"multipliers: {strategy['multipliers']}",
                    f"split_ratios: {[round(r, 6) for r in strategy['split_ratios']]}",
                    f"dataset: {dataset_cfg['name']}",
                    f"task: {dataset_cfg['task']}",
                    f"metric: {dataset_cfg['metric']}",
                    f"transform: {dataset_cfg['transform']}",
                    f"description: {strategy['description']}",
                ]
                (model_dir / 'model_meta.txt').write_text(
                    '\n'.join(meta_lines) + '\n', encoding='utf-8'
                )

                # dataframe.df (NNEval reads this for task/dataset/metric)
                write_dataframe_df(model_dir, arch, strategy, dataset_cfg, hp_dict)

                model_idx += 1
                arch_gen += 1
                total_generated += 1

        arch_stats[arch] = {'generated': arch_gen, 'skipped': arch_skip}
        status = 'OK' if arch_gen > 0 else 'FAIL'
        print(f"[{status}] {arch}: {arch_gen} dirs generated, {arch_skip} skipped")

    print(f"\n{'=' * 64}")
    print(f"TOTAL: {total_generated} model directories generated, {total_skipped} skipped")
    print(f"Models saved to: {output_base}")
    print(f"{'=' * 64}")
    print()
    print("Per-architecture summary:")
    for arch, stats in arch_stats.items():
        print(f"  {arch:40s}  gen={stats['generated']:4d}  skip={stats['skipped']:4d}")

    return total_generated


def main():
    project_root = Path(__file__).resolve().parents[4]  # nn-gpt root
    output_dir = (
        project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / 'A0' / 'synth_nn'
    )

    # Remove stale llr_ model directories
    if output_dir.exists():
        import shutil
        removed = 0
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith('llr_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing llr_ model directories.\n")

    total = generate_models(str(output_dir), prefix='llr')
    print(f"\nDone. Generated {total} model directories ready for NNEval.")
    print()
    print("To evaluate on cifar-10 (metadata overrides dataset per model):")
    print(
        "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python -m ab.gpt.NNEval --only_epoch 0 --nn_train_epochs 5 "
        "--nn_name_prefix llr"
    )
    print()
    print("Each model dir contains dataframe.df which NNEval uses to resolve")
    print("the correct dataset (imagenette / cifar-10 / cifar-100) automatically.")


if __name__ == '__main__':
    main()

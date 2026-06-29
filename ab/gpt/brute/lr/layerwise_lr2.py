"""
Layerwise Learning Rate Model Generator — Batch 2 (~1000 CV Models)

Second batch of layerwise LR strategies extending layerwise_lr.py:
  - Inverted decay    : head gets LOWER LR than backbone (anti-freeze)
  - Asymmetric splits : 90/10, 80/20 — very small head gets full LR
  - Cosine-spaced     : multipliers follow a cosine curve backbone→head
  - More groups       : 6-group linear / exponential decay shapes
  - Cyclic grouping   : even/odd layers alternate between two LR values

Math: 29 archs × 13 strategies × 3 datasets = 1131 potential; ~1000 after skips.

Usage:
    python -m ab.gpt.brute.lr.layerwise_lr2
"""

import ast
import json
import re
from pathlib import Path

from ab.gpt.brute.lr.schedulers import (
    _find_method_range,
    read_architecture_source,
)
from ab.gpt.brute.lr.layerwise_lr import (
    ARCHITECTURES,
    ARCH_EXTRA_HP_DEFAULTS,
    DATASETS,
    _find_optimizer_block,
    build_hp_dict,
    find_nn_source_dir,
    get_supported_hp_from_source,
    write_dataframe_df,
)


# ---------------------------------------------------------------------------
# Layerwise LR strategies — batch 2
#
# 'type' == 'n_group' → contiguous parameter slices by split_ratios.
# 'type' == 'cyclic'  → even/odd layer indices cycle through multipliers;
#                       split_ratios is None for cyclic strategies.
#
# Cosine-spaced multiplier formula (min=0.1, max=1.0, i = 0..n−1):
#   m_i = 0.1 + 0.9 * (1 − cos(π·i / (n−1))) / 2
# ---------------------------------------------------------------------------
LAYERWISE_STRATEGIES = [
    # ── Inverted decay: head gets LOWER LR (anti-freeze) ───────────────────
    {
        'name': 'llr2_2grp_inv_01x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [1.0, 0.1],
        'split_ratios': [0.5, 0.5],
        'description': 'Inverted 50/50: backbone 1.0×, head 0.1× (anti-freeze).',
    },
    {
        'name': 'llr2_2grp_inv_001x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [1.0, 0.01],
        'split_ratios': [0.5, 0.5],
        'description': 'Inverted 50/50: backbone 1.0×, head 0.01× (strong anti-freeze).',
    },
    {
        'name': 'llr2_3grp_inv_exp',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [1.0, 0.1, 0.01],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups inverted-exp: backbone 1.0×, mid 0.1×, head 0.01×.',
    },
    {
        'name': 'llr2_4grp_inv_lin',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [1.0, 0.6, 0.3, 0.1],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups inverted-linear: 1.0×, 0.6×, 0.3×, 0.1× (head slowest).',
    },
    # ── Asymmetric splits: very small head ──────────────────────────────────
    {
        'name': 'llr2_2grp_90_10',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.9, 0.1],
        'description': '90/10 split: 90% backbone at 0.1×, tiny 10% head at 1.0×.',
    },
    {
        'name': 'llr2_2grp_80_20',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.8, 0.2],
        'description': '80/20 split: 80% backbone at 0.1×, 20% head at 1.0×.',
    },
    {
        'name': 'llr2_3grp_80_10_10',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.01, 0.1, 1.0],
        'split_ratios': [0.8, 0.1, 0.1],
        'description': '80/10/10: large backbone 0.01×, small mid 0.1×, tiny head 1.0×.',
    },
    # ── Cosine-spaced multipliers (backbone→head) ────────────────────────────
    {
        'name': 'llr2_3grp_cos',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.1, 0.55, 1.0],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups cosine-spaced: 0.1×, 0.55×, 1.0×.',
    },
    {
        'name': 'llr2_4grp_cos',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.1, 0.325, 0.775, 1.0],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups cosine-spaced: 0.1×, 0.325×, 0.775×, 1.0×.',
    },
    # ── More groups: 6 ──────────────────────────────────────────────────────
    {
        'name': 'llr2_6grp_lin',
        'type': 'n_group',
        'n_groups': 6,
        'multipliers': [0.1, 0.28, 0.46, 0.64, 0.82, 1.0],
        'split_ratios': [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        'description': '6 equal groups linear-spaced 0.1→1.0 (step 0.18).',
    },
    {
        'name': 'llr2_6grp_exp',
        'type': 'n_group',
        'n_groups': 6,
        'multipliers': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'split_ratios': [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        'description': '6 equal groups log-spaced: 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0.',
    },
    # ── Cyclic grouping: even/odd layer indices alternate ────────────────────
    {
        'name': 'llr2_cyclic_01x',
        'type': 'cyclic',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': None,
        'description': 'Cyclic 2-way: even-indexed layers 0.1×, odd-indexed layers 1.0×.',
    },
    {
        'name': 'llr2_cyclic_001x',
        'type': 'cyclic',
        'n_groups': 2,
        'multipliers': [0.01, 1.0],
        'split_ratios': None,
        'description': 'Cyclic 2-way: even-indexed layers 0.01×, odd-indexed layers 1.0×.',
    },
]  # 13 strategies × 29 archs × 3 datasets = 1131 potential model dirs


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def _build_grouping_code(strategy: dict) -> str:
    """
    Return Python source lines (8-space indent for a method body) that build
    _llr_groups — the per-group parameter list passed to the optimizer.

    Supports 'n_group' (contiguous slices via split_ratios) and
    'cyclic' (even/odd alternation through multipliers).
    """
    name = strategy['name']
    multipliers = strategy['multipliers']

    def _fmt(v):
        return f'{v:.10g}'

    if strategy['type'] == 'cyclic':
        n_cycle = len(multipliers)
        mults_str = '[' + ', '.join(_fmt(m) for m in multipliers) + ']'
        lines = [
            f"        # Layerwise LR strategy: {name} (cyclic)",
            f"        _llr_params = list(self.named_parameters())",
            f"        _llr_cycle = {mults_str}",
            f"        _llr_buckets = {{}}",
            f"        for _llr_i, (_, _llr_p) in enumerate(_llr_params):",
            f"            _llr_m = _llr_cycle[_llr_i % {n_cycle}]",
            f"            _llr_buckets.setdefault(_llr_m, []).append(_llr_p)",
            f"        _llr_groups = [{{'params': _llr_ps, 'lr': prm.get('lr', 0.01) * _llr_m}} for _llr_m, _llr_ps in _llr_buckets.items()]",
        ]
        return '\n'.join(lines)

    # n_group: contiguous slices by split_ratios
    split_ratios = strategy['split_ratios']
    ratios_str = '[' + ', '.join(_fmt(r) for r in split_ratios) + ']'
    mults_str  = '[' + ', '.join(_fmt(m) for m in multipliers) + ']'
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


def inject_layerwise_lr(source_code: str, strategy: dict):
    """
    Inject layerwise LR grouping into train_setup() of the architecture source.

    For 'uniform' strategies returns source unchanged.
    For 'n_group' and 'cyclic':
      1. Locate class Net and its train_setup method.
      2. Find the self.optimizer = ... block.
      3. Confirm self.parameters() appears in that block.
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

    opt_block = '\n'.join(lines[opt_line_idx:opt_end_idx + 1])
    if 'self.parameters()' not in opt_block:
        return None

    grouping_code = _build_grouping_code(strategy)
    new_opt_block = opt_block.replace('self.parameters()', '_llr_groups', 1)

    new_lines = (
        lines[:opt_line_idx]
        + grouping_code.split('\n')
        + new_opt_block.split('\n')
        + lines[opt_end_idx + 1:]
    )
    code = '\n'.join(new_lines)

    try:
        ast.parse(code)
    except SyntaxError:
        return None

    return code


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_models(output_base_dir: str, prefix: str = 'llr2') -> int:
    """
    Generate batch-2 layerwise-LR model variants and write to output_base_dir.

    Directory layout per model (identical to layerwise_lr.py):
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

        arch_hp = get_supported_hp_from_source(source_code)
        if arch_hp is None:
            print(f"[SKIP] {arch}: supported_hyperparameters() not found in source")
            total_skipped += len(LAYERWISE_STRATEGIES) * len(DATASETS)
            arch_stats[arch] = {'generated': 0, 'skipped': len(LAYERWISE_STRATEGIES) * len(DATASETS)}
            continue

        arch_gen = 0
        arch_skip = 0

        for strategy in LAYERWISE_STRATEGIES:
            model_code = inject_layerwise_lr(source_code, strategy)
            if model_code is None:
                print(f"  [SKIP] {arch} / {strategy['name']}: no self.parameters() or injection failed")
                arch_skip += len(DATASETS)
                total_skipped += len(DATASETS)
                continue

            for dataset_cfg in DATASETS:
                model_name = f"{prefix}_{model_idx:04d}"
                model_dir = output_base / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                (model_dir / 'new_nn.py').write_text(model_code, encoding='utf-8')

                hp_dict = build_hp_dict(arch, dataset_cfg)
                (model_dir / 'hp.txt').write_text(
                    json.dumps(hp_dict, indent=2), encoding='utf-8'
                )

                split_ratios_repr = (
                    [round(r, 6) for r in strategy['split_ratios']]
                    if strategy['split_ratios'] is not None
                    else None
                )
                meta_lines = [
                    f"architecture: {arch}",
                    f"strategy: {strategy['name']}",
                    f"strategy_type: {strategy['type']}",
                    f"n_groups: {strategy['n_groups']}",
                    f"multipliers: {strategy['multipliers']}",
                    f"split_ratios: {split_ratios_repr}",
                    f"dataset: {dataset_cfg['name']}",
                    f"task: {dataset_cfg['task']}",
                    f"metric: {dataset_cfg['metric']}",
                    f"transform: {dataset_cfg['transform']}",
                    f"description: {strategy['description']}",
                ]
                (model_dir / 'model_meta.txt').write_text(
                    '\n'.join(meta_lines) + '\n', encoding='utf-8'
                )

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

    if output_dir.exists():
        import shutil
        removed = 0
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith('llr2_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing llr2_ model directories.\n")

    total = generate_models(str(output_dir), prefix='llr2')
    print(f"\nDone. Generated {total} model directories ready for NNEval.")
    print()
    print("To evaluate (dataframe.df resolves dataset per model automatically):")
    print(
        "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python -m ab.gpt.NNEval --only_epoch 0 --nn_train_epochs 5 "
        "--nn_name_prefix llr2"
    )


if __name__ == '__main__':
    main()

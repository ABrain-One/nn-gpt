"""
Layerwise Learning Rate Model Generator — Batch 3 (~1000 CV Models)

Third batch of layerwise LR strategies — new mathematical families not
covered by batches 1 & 2:
  - Geometric multipliers   : power-of-2 decay / inverted
  - Non-monotone            : V-shaped (middle fastest), plateau
  - Step functions          : hard freeze at backbone or head
  - Skewed split ratios     : backbone-heavy 60/30/10, head-heavy 10/30/60
  - Top-N full LR           : only last 2 groups at 1.0×
  - Geometric split ratios  : group sizes halve each step
  - Gentle 2-group          : 0.5× backbone (mild regularisation)
  - Very tiny head          : 95/5 split
  - Inverted cosine 6-group : cosine curve with backbone-high direction

Math: 29 archs × 14 strategies × 3 datasets = 1218 potential; ~400 after skips.

Usage:
    python -m ab.gpt.brute.llr.layerwise_lr3
"""

import ast
import json
import re
from pathlib import Path

from ab.gpt.brute.lr.schedulers import (
    _find_method_range,
    read_architecture_source,
)
from ab.gpt.brute.llr.layerwise_lr import (
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
# Layerwise LR strategies — batch 3
#
# New mathematical families vs batches 1 & 2:
#   geometric multipliers, non-monotone shapes, step functions,
#   skewed split ratios, geometric split ratios, gentle / extreme splits.
#
# Inverted cosine 6-group multipliers (backbone-high, head-low):
#   m_i = 0.1 + 0.9 * (1 − cos(π·(n−1−i) / (n−1))) / 2
#   → [1.0, 0.914, 0.689, 0.411, 0.186, 0.1]
# ---------------------------------------------------------------------------
LAYERWISE_STRATEGIES = [
    # ── Gentle 2-group (mild regularisation) ────────────────────────────────
    {
        'name': 'llr3_2grp_05x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.5, 1.0],
        'split_ratios': [0.5, 0.5],
        'description': '50/50 split: backbone 0.5×, head 1.0× (gentle freeze).',
    },
    {
        'name': 'llr3_2grp_inv_05x',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [1.0, 0.5],
        'split_ratios': [0.5, 0.5],
        'description': '50/50 inverted: backbone 1.0×, head 0.5× (gentle anti-freeze).',
    },
    # ── Very tiny head ───────────────────────────────────────────────────────
    {
        'name': 'llr3_2grp_95_5',
        'type': 'n_group',
        'n_groups': 2,
        'multipliers': [0.1, 1.0],
        'split_ratios': [0.95, 0.05],
        'description': '95/5 split: 95% backbone at 0.1×, tiny 5% head at 1.0×.',
    },
    # ── Step functions: hard freeze except one region ────────────────────────
    {
        'name': 'llr3_3grp_step_head',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.1, 0.1, 1.0],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups step-at-head: backbone+mid 0.1×, head 1.0×.',
    },
    {
        'name': 'llr3_3grp_step_back',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [1.0, 0.1, 0.1],
        'split_ratios': [1/3, 1/3, 1/3],
        'description': '3 equal groups step-at-backbone: backbone 1.0×, mid+head 0.1×.',
    },
    # ── Skewed split ratios ──────────────────────────────────────────────────
    {
        'name': 'llr3_3grp_60_30_10',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.01, 0.1, 1.0],
        'split_ratios': [0.6, 0.3, 0.1],
        'description': '60/30/10 backbone-heavy: large frozen backbone, small full-LR head.',
    },
    {
        'name': 'llr3_3grp_10_30_60',
        'type': 'n_group',
        'n_groups': 3,
        'multipliers': [0.01, 0.1, 1.0],
        'split_ratios': [0.1, 0.3, 0.6],
        'description': '10/30/60 head-heavy: small backbone group, large full-LR head.',
    },
    # ── Geometric multipliers (power-of-2) ───────────────────────────────────
    {
        'name': 'llr3_4grp_geo_mults',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.125, 0.25, 0.5, 1.0],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups geometric (×2 each step): 0.125×, 0.25×, 0.5×, 1.0×.',
    },
    {
        'name': 'llr3_4grp_geo_mults_inv',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [1.0, 0.5, 0.25, 0.125],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups inverted geometric: 1.0×, 0.5×, 0.25×, 0.125× (head slowest).',
    },
    # ── Geometric split ratios (groups halve in size) ────────────────────────
    {
        'name': 'llr3_4grp_geo_splits',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.1, 0.3, 0.6, 1.0],
        'split_ratios': [0.5, 0.25, 0.125, 0.125],
        'description': '4 groups geometric-sized (50/25/12.5/12.5): linear mults 0.1→1.0.',
    },
    # ── Top-2 groups at full LR ──────────────────────────────────────────────
    {
        'name': 'llr3_4grp_top2_full',
        'type': 'n_group',
        'n_groups': 4,
        'multipliers': [0.01, 0.01, 1.0, 1.0],
        'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'description': '4 equal groups: bottom-2 at 0.01×, top-2 at 1.0× (step at midpoint).',
    },
    # ── Non-monotone shapes ──────────────────────────────────────────────────
    {
        'name': 'llr3_5grp_vshaped',
        'type': 'n_group',
        'n_groups': 5,
        'multipliers': [0.1, 0.5, 1.0, 0.5, 0.1],
        'split_ratios': [0.2, 0.2, 0.2, 0.2, 0.2],
        'description': '5 equal groups V-shaped: extremes 0.1×, centre 1.0× (middle layers fastest).',
    },
    {
        'name': 'llr3_5grp_plateau',
        'type': 'n_group',
        'n_groups': 5,
        'multipliers': [0.1, 1.0, 1.0, 1.0, 0.1],
        'split_ratios': [0.2, 0.2, 0.2, 0.2, 0.2],
        'description': '5 equal groups plateau: extremes 0.1×, inner 3 at 1.0× (plateau shape).',
    },
    # ── Inverted cosine 6-group (backbone-high direction) ───────────────────
    {
        'name': 'llr3_6grp_cos_inv',
        'type': 'n_group',
        'n_groups': 6,
        'multipliers': [1.0, 0.914, 0.689, 0.411, 0.186, 0.1],
        'split_ratios': [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        'description': '6 equal groups cosine-spaced inverted: 1.0→0.1× (backbone-high).',
    },
]  # 14 strategies × 29 archs × 3 datasets = 1218 potential model dirs


# ---------------------------------------------------------------------------
# Code generation — identical to layerwise_lr2.py (n_group only in batch 3)
# ---------------------------------------------------------------------------

def _build_grouping_code(strategy: dict) -> str:
    name = strategy['name']
    split_ratios = strategy['split_ratios']
    multipliers = strategy['multipliers']

    def _fmt(v):
        return f'{v:.10g}'

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
    Returns modified source string, or None if injection is not possible.
    """
    lines = source_code.split('\n')

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

    ts_range = _find_method_range(lines, 'train_setup', class_indent)
    if ts_range is None:
        return None
    ts_start, ts_end = ts_range

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

def generate_models(output_base_dir: str, prefix: str = 'llr3') -> int:
    """
    Generate batch-3 layerwise-LR model variants and write to output_base_dir.

    Same layout as batches 1 & 2:
        <prefix>_NNNN/
            new_nn.py       – modified architecture code
            hp.txt          – JSON hyperparameter dict
            model_meta.txt  – human-readable metadata
            dataframe.df    – pandas Series pickle for NNEval

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
                arch_skip += 1
                total_skipped += 1
                continue

            # ONE dir per (arch, strategy). Code is dataset-independent, so the
            # evaluator fans it out over datasets via `NNEval --datasets ...`
            # (one stat row each) instead of duplicating the dir per dataset.
            for dataset_cfg in DATASETS[:1]:
                model_name = f"{prefix}_{model_idx:04d}"
                model_dir = output_base / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                (model_dir / 'new_nn.py').write_text(model_code, encoding='utf-8')

                hp_dict = build_hp_dict(arch, dataset_cfg)
                (model_dir / 'hp.txt').write_text(
                    json.dumps(hp_dict, indent=2), encoding='utf-8'
                )

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
            if d.is_dir() and d.name.startswith('llr3_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing llr3_ model directories.\n")

    total = generate_models(str(output_dir), prefix='llr3')
    print(f"\nDone. Generated {total} model directories ready for NNEval.")
    print()
    print("To evaluate (dataframe.df resolves dataset per model automatically):")
    print(
        "  PYTORCH_CUDA_ALLOC_ALLOC_CONF=expandable_segments:True "
        "python -m ab.gpt.NNEval --nn_name_prefix llr3 --nn_train_epochs 1 "
        "--prm_json '{\"batch\": 32}'"
    )


if __name__ == '__main__':
    main()

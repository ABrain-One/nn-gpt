"""
Layerwise LR Results Analyzer

Reads eval_info.json + model_meta.txt from every llr_* model directory and
produces:
  - results.csv        : flat table of all evaluated models
  - strategy_rank.csv  : per-(architecture, dataset) strategy ranking
  - group_rank.csv     : accuracy by n_groups, averaged across all archs
  - report.txt         : human-readable summary

Usage:
    python -m ab.gpt.brute.lr.analyze_llr
    python -m ab.gpt.brute.lr.analyze_llr --synth_dir /path/to/synth_nn --out_dir /path/to/out
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _default_synth_dir() -> Path:
    return Path(__file__).resolve().parents[4] / 'out' / 'nngpt' / 'llm' / 'epoch' / 'A0' / 'synth_nn'

def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[4] / 'out' / 'nngpt' / 'llr_analysis'


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_meta(meta_path: Path) -> dict:
    meta = {}
    for line in meta_path.read_text(encoding='utf-8').splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip()
    return meta


def _parse_eval(eval_path: Path) -> dict | None:
    try:
        data = json.loads(eval_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    accuracy = None
    er = data.get('eval_results', {})
    if isinstance(er, dict):
        accuracy = er.get('accuracy')
    if accuracy is None:
        return None
    prm = data.get('eval_args', {}).get('prm', {})
    return {
        'accuracy': float(accuracy),
        'epochs_trained': int(prm.get('epoch', prm.get('epoch_max', 1))),
        'train_loss': prm.get('train_loss'),
        'test_loss': prm.get('test_loss'),
        'samples_per_second': prm.get('samples_per_second'),
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(synth_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(synth_dir.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith('llr_'):
            continue

        meta_path = model_dir / 'model_meta.txt'
        eval_path = model_dir / 'eval_info.json'
        error_path = model_dir / 'error.txt'

        if not meta_path.exists():
            continue

        meta = _parse_meta(meta_path)

        row = {
            'model_id': model_dir.name,
            'architecture': meta.get('architecture', ''),
            'strategy': meta.get('strategy', ''),
            'strategy_type': meta.get('strategy_type', ''),
            'n_groups': int(meta.get('n_groups', 1)),
            'dataset': meta.get('dataset', ''),
            'task': meta.get('task', ''),
            'multipliers': meta.get('multipliers', ''),
            'split_ratios': meta.get('split_ratios', ''),
            'description': meta.get('description', ''),
            'status': 'pending',
            'accuracy': None,
            'epochs_trained': None,
            'train_loss': None,
            'test_loss': None,
            'samples_per_second': None,
        }

        if eval_path.exists():
            result = _parse_eval(eval_path)
            if result:
                row.update(result)
                row['status'] = 'success'
            else:
                row['status'] = 'error'
        elif error_path.exists():
            row['status'] = 'error'

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def strategy_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (architecture, dataset), rank strategies by mean accuracy.
    Returns a wide table: rows = (arch, dataset), columns = strategies.
    """
    success = df[df['status'] == 'success'].copy()
    if success.empty:
        return pd.DataFrame()

    pivot = (
        success
        .groupby(['architecture', 'dataset', 'strategy'])['accuracy']
        .mean()
        .reset_index()
        .pivot_table(index=['architecture', 'dataset'], columns='strategy', values='accuracy')
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # Add a 'best_strategy' column
    strat_cols = [c for c in pivot.columns if c not in ('architecture', 'dataset')]
    if strat_cols:
        pivot['best_strategy'] = pivot[strat_cols].idxmax(axis=1)
        pivot['best_accuracy'] = pivot[strat_cols].max(axis=1)
        pivot['uniform_accuracy'] = pivot.get('llr_uniform', None)
        pivot['best_vs_uniform'] = pivot['best_accuracy'] - pivot.get('uniform_accuracy', 0)
    return pivot


def group_count_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Average accuracy by n_groups across all archs/datasets."""
    success = df[df['status'] == 'success'].copy()
    if success.empty:
        return pd.DataFrame()

    grp = (
        success
        .groupby(['n_groups', 'dataset'])['accuracy']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_accuracy', 'std': 'std_accuracy', 'count': 'n_models'})
        .sort_values(['dataset', 'mean_accuracy'], ascending=[True, False])
    )
    return grp


def arch_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each architecture: what is the gain from using the best layerwise
    strategy vs the uniform baseline?
    """
    success = df[df['status'] == 'success'].copy()
    if success.empty:
        return pd.DataFrame()

    uniform = (
        success[success['strategy'] == 'llr_uniform']
        .groupby(['architecture', 'dataset'])['accuracy']
        .mean()
        .rename('uniform_acc')
    )
    best = (
        success[success['strategy'] != 'llr_uniform']
        .groupby(['architecture', 'dataset'])['accuracy']
        .max()
        .rename('best_llr_acc')
    )
    comb = pd.concat([uniform, best], axis=1).dropna().reset_index()
    comb['llr_gain'] = comb['best_llr_acc'] - comb['uniform_acc']
    comb = comb.sort_values('llr_gain', ascending=False)
    return comb


def multiplier_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 2-group strategies only, show how backbone multiplier affects accuracy.
    """
    success = df[(df['status'] == 'success') & (df['n_groups'] == 2)].copy()
    if success.empty:
        return pd.DataFrame()

    grp = (
        success
        .groupby(['strategy', 'multipliers', 'dataset'])['accuracy']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_accuracy', 'count': 'n_models'})
        .sort_values(['dataset', 'mean_accuracy'], ascending=[True, False])
    )
    return grp


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def build_report(df: pd.DataFrame, strat_rank: pd.DataFrame,
                 grp_rank: pd.DataFrame, arch_sens: pd.DataFrame) -> str:
    lines = []

    success_df = df[df['status'] == 'success']
    total = len(df)
    n_success = len(success_df)
    n_error = (df['status'] == 'error').sum()
    n_pending = (df['status'] == 'pending').sum()

    lines += [
        '=' * 72,
        'LAYERWISE LR ANALYSIS REPORT',
        '=' * 72,
        f'Total model dirs  : {total}',
        f'  Evaluated       : {n_success}',
        f'  Errors          : {n_error}',
        f'  Pending         : {n_pending}',
        '',
    ]

    if success_df.empty:
        lines.append('No successful evaluations yet.')
        return '\n'.join(lines)

    # ── Overall best accuracy ────────────────────────────────────────────
    lines += ['─' * 72, 'TOP-10 MODELS BY ACCURACY', '─' * 72]
    top10 = (
        success_df
        .nlargest(10, 'accuracy')
        [['model_id', 'architecture', 'strategy', 'dataset', 'accuracy', 'epochs_trained']]
    )
    lines.append(top10.to_string(index=False))
    lines.append('')

    # ── Best strategy per architecture × dataset ─────────────────────────
    if not arch_sens.empty:
        lines += ['─' * 72, 'LLR GAIN OVER UNIFORM BASELINE (best layerwise - uniform)', '─' * 72]
        top_gain = arch_sens.head(20)
        for _, r in top_gain.iterrows():
            gain_str = f'+{r["llr_gain"]:.4f}' if r['llr_gain'] >= 0 else f'{r["llr_gain"]:.4f}'
            lines.append(
                f'  {r["architecture"]:25s}  {r["dataset"]:12s}'
                f'  uniform={_fmt(r["uniform_acc"])}  best_llr={_fmt(r["best_llr_acc"])}'
                f'  gain={gain_str}'
            )
        lines.append('')

    # ── Strategy win counts ──────────────────────────────────────────────
    if not strat_rank.empty and 'best_strategy' in strat_rank.columns:
        lines += ['─' * 72, 'STRATEGY WIN COUNT (how many arch×dataset slots does each strategy win?)', '─' * 72]
        win_counts = strat_rank['best_strategy'].value_counts()
        for strat, cnt in win_counts.items():
            lines.append(f'  {strat:30s}  {cnt:3d} wins')
        lines.append('')

    # ── Group count analysis ─────────────────────────────────────────────
    if not grp_rank.empty:
        lines += ['─' * 72, 'ACCURACY BY N_GROUPS (mean over all architectures)', '─' * 72]
        for dataset in grp_rank['dataset'].unique():
            lines.append(f'  Dataset: {dataset}')
            sub = grp_rank[grp_rank['dataset'] == dataset]
            for _, r in sub.iterrows():
                lines.append(
                    f'    n_groups={int(r["n_groups"])}  '
                    f'mean={_fmt(r["mean_accuracy"])}  '
                    f'std={_fmt(r["std_accuracy"])}  '
                    f'n={int(r["n_models"])}'
                )
            lines.append('')

    # ── Per-dataset best strategies ──────────────────────────────────────
    lines += ['─' * 72, 'BEST STRATEGY PER DATASET (averaged across all architectures)', '─' * 72]
    for dataset in sorted(success_df['dataset'].unique()):
        sub = success_df[success_df['dataset'] == dataset]
        top_strat = (
            sub.groupby('strategy')['accuracy']
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append(f'  {dataset}:')
        for strat, acc in top_strat.items():
            lines.append(f'    {strat:30s}  mean_acc={acc:.4f}')
        lines.append('')

    # ── Architectures most/least sensitive to LLR ───────────────────────
    if not arch_sens.empty:
        lines += ['─' * 72, 'ARCHITECTURES MOST RESPONSIVE TO LAYERWISE LR', '─' * 72]
        by_arch = arch_sens.groupby('architecture')['llr_gain'].mean().sort_values(ascending=False)
        lines.append('  Most improved:')
        for arch, gain in by_arch.head(5).items():
            lines.append(f'    {arch:30s}  avg_gain={gain:+.4f}')
        lines.append('  Least improved (uniform already optimal):')
        for arch, gain in by_arch.tail(5).items():
            lines.append(f'    {arch:30s}  avg_gain={gain:+.4f}')
        lines.append('')

    lines += ['=' * 72, 'END OF REPORT', '=' * 72]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(synth_dir: str | None = None, out_dir: str | None = None):
    synth = Path(synth_dir) if synth_dir else _default_synth_dir()
    out = Path(out_dir) if out_dir else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    print(f'Scanning {synth} ...')
    df = collect(synth)
    print(f'  {len(df)} model dirs  |  {(df.status == "success").sum()} evaluated  |  {(df.status == "error").sum()} errors')

    # Save flat results table
    results_csv = out / 'results.csv'
    df.to_csv(results_csv, index=False)
    print(f'  Saved {results_csv}')

    strat_rank = strategy_ranking(df)
    grp_rank = group_count_ranking(df)
    arch_sens = arch_sensitivity(df)
    mult_eff = multiplier_effect(df)

    if not strat_rank.empty:
        path = out / 'strategy_rank.csv'
        strat_rank.to_csv(path, index=False)
        print(f'  Saved {path}')

    if not grp_rank.empty:
        path = out / 'group_rank.csv'
        grp_rank.to_csv(path, index=False)
        print(f'  Saved {path}')

    if not arch_sens.empty:
        path = out / 'arch_sensitivity.csv'
        arch_sens.to_csv(path, index=False)
        print(f'  Saved {path}')

    if not mult_eff.empty:
        path = out / 'multiplier_effect.csv'
        mult_eff.to_csv(path, index=False)
        print(f'  Saved {path}')

    report = build_report(df, strat_rank, grp_rank, arch_sens)
    report_path = out / 'report.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f'  Saved {report_path}')
    print()
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth_dir', type=str, default=None,
                        help='Path to synth_nn directory (default: auto-detected)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for CSVs and report (default: out/nngpt/llr_analysis)')
    args = parser.parse_args()
    main(synth_dir=args.synth_dir, out_dir=args.out_dir)

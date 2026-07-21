"""
Generation-time candidate sampling for the layerwise-LR (LLR) fine-tune.

Problem this solves: Tune.py's generic nn_gen() picks the baseline+"addon"
example to show the model each generation cycle via a raw self-join over the
ENTIRE LEMUR stat table (lemur.data(sql=JoinConf(...))) — no nn_prefixes
filter, no vanilla-anchoring. For LLR this means the model is likely shown an
arbitrary, unrelated pair of models rather than "vanilla arch + a layerwise-LR
strategy spec to apply", which is what it was actually fine-tuned on. That
train/generation distribution mismatch would undercut everything upstream.

This module builds generation candidates in the SAME shape as the SFT training
pairs (see NNGenPrompt.build_vanilla_anchored_pairs), split by prompt key:

  - Selection key (key_dict['improve'] is True): replay real, proven
    (architecture, dataset) -> winning-strategy combos from our evaluated data
    — validates the model reproduces known-good transformations faithfully.
  - Mechanism key (key_dict['improve'] is False): EXPLORE — sample a vanilla
    architecture from the full DB (not just the ~26 we've brute-forced LLR on)
    crossed with a strategy spec from the full 39-strategy catalog. This is the
    actual generalization target: applying LLR to architectures/strategy
    combinations we've never evaluated, using the model's learned injection
    skill rather than a deterministic generator.
"""

import random

from ab.gpt.brute.llr.llr_baselines import (
    LLR_META_COLS,
    load_all_strategies,
    load_all_vanilla_archs,
    load_arch_best_accuracy,
    load_llr_metadata,
    load_vanilla_baselines,
)

# Must match the transforms actually used by the LLR brute-force re-evaluation
# (nngpt-llr-reeval.yaml's --datasets flag): imagenette benefits from the higher
# resolution, cifar-10/100 (32x32 native) do not.
CANONICAL_DATASETS = (
    ('imagenette', 'norm_256_flip'),
    ('cifar-10', 'norm_64_flip'),
    ('cifar-100', 'norm_128_flip'),
)
CANONICAL_TASK = 'img-classification'
CANONICAL_METRIC = 'acc'
CANONICAL_EPOCH = 1  # matches the brute-force re-evaluation's --nn_train_epochs 1


# Raw LAYERWISE_STRATEGIES dicts use different keys (name/type) than the
# metadata CSV / training-pair columns (strategy/strategy_type). Map explicitly
# rather than relying on LLR_META_COLS, whose names match the CSV schema.
_SPEC_KEY_MAP = {
    'strategy': 'name',
    'strategy_type': 'type',
    'n_groups': 'n_groups',
    'multipliers': 'multipliers',
    'split_ratios': 'split_ratios',
    'description': 'description',
}


def _row_from_spec(arch: str, code: str, dataset: str, transform: str,
                    accuracy, spec: dict) -> dict:
    row = {
        'nn': arch,
        'nn_code': code,
        'accuracy': accuracy if accuracy is not None else '',
        'epoch': CANONICAL_EPOCH,
        'dataset': dataset,
        'task': CANONICAL_TASK,
        'metric': CANONICAL_METRIC,
        'transform': transform,
        'transform_code': '',
        'prm': {},
    }
    for csv_col, spec_key in _SPEC_KEY_MAP.items():
        val = spec.get(spec_key, '')
        row[f'{csv_col}_2'] = str(val) if isinstance(val, (list, tuple)) else val
    row['architecture_2'] = arch
    row['accuracy_2'] = ''  # unknown at generation time — no real result yet
    row['nn_code_2'] = ''   # the model generates this; nothing to show as ground truth
    row['transform_code_2'] = ''
    row['prm_2'] = {}
    return row


def sample_selection_candidates(test_nn: int, seed: int | None = None) -> list[dict]:
    """
    Replay real, proven (architecture, dataset) -> winning-strategy combinations.
    Sampled from the same matched-pairs pool the Selection training bucket uses,
    so generation-time input distribution matches what the model was trained on.
    """
    from ab.gpt.util.prompt.NNGenPrompt import _match_llr_to_vanilla
    import pandas as pd
    import ab.nn.api as lemur

    meta = load_llr_metadata()
    if not meta:
        return []
    llr_names = [nn for nn, m in meta.items() if 'uniform' not in m.get('strategy', '').lower()]
    if not llr_names:
        return []
    # Best-accuracy row per evaluated llr model, across all its (dataset,epoch,transform) runs.
    df = lemur.data(only_best_accuracy=True, task=CANONICAL_TASK, max_rows=None)
    if df.empty or 'nn' not in df.columns:
        return []
    df = df[df['nn'].isin(llr_names)]
    rows = _match_llr_to_vanilla(df, require_improve=True)
    if not rows:
        return []
    rng = random.Random(seed)
    picks = rng.sample(rows, k=min(test_nn, len(rows))) if len(rows) > test_nn \
        else [rng.choice(rows) for _ in range(test_nn)]
    out = []
    for r in picks:
        # Start from the full matched-pair dict (same shape _match_llr_to_vanilla
        # gives training) rather than a curated subset — Tune.py's generation loop
        # does an unconditional row[it["value"]] for every input_list entry
        # (including addon_prm/addon_transform_code/addon_nn_code, even though
        # generation doesn't render them into the prompt), so any input_list
        # column missing from this row would raise a KeyError and crash generation.
        row = dict(r)
        row.setdefault('task', CANONICAL_TASK)
        row.setdefault('metric', CANONICAL_METRIC)
        for k in ('nn_code_2', 'transform_code_2', 'prm_2', 'accuracy_2'):
            row.setdefault(k, '')
        out.append(row)
    return out


def sample_mechanism_candidates(test_nn: int, seed: int | None = None) -> list[dict]:
    """
    Exploratory candidates: a vanilla architecture (from the FULL DB, not just
    the ~26 architectures the LLR generators have brute-forced) crossed with a
    strategy spec (from the full 39-strategy catalog). Real "current accuracy"
    is looked up when available; otherwise left blank (the mechanism prompt
    framing never claims a target accuracy, so this is safe).
    """
    archs = load_all_vanilla_archs()
    # Exclude the uniform control by NAME, not type: llr_uniform's 'type' field
    # is 'n_group' (it's implemented as a real single-group injection, not a
    # no-op passthrough — see layerwise_lr.py) so it doesn't match type=='uniform'.
    strategies = [s for s in load_all_strategies() if 'uniform' not in s.get('name', '').lower()]
    if not archs or not strategies:
        return []
    arch_acc = load_arch_best_accuracy(list(archs.keys()))

    rng = random.Random(seed)
    arch_names = list(archs.keys())
    out = []
    for _ in range(test_nn):
        arch = rng.choice(arch_names)
        code = archs[arch]
        spec = rng.choice(strategies)
        dataset, transform = rng.choice(CANONICAL_DATASETS)
        acc = arch_acc.get((arch, dataset, CANONICAL_EPOCH, transform))
        out.append(_row_from_spec(arch, code, dataset, transform, acc, spec))
    return out


def build_llr_generation_data(key_dict: dict, test_nn: int, epoch: int = 0):
    """
    Entry point called from Tune.py's nn_gen(). Returns a pandas DataFrame with
    the same bare + '_2'-suffixed column shape as the SFT training pairs, sized
    to test_nn rows (best-effort; may return fewer if data is sparse).
    """
    import pandas as pd
    improve = bool(key_dict.get('improve', False))
    seed = epoch * 7919  # any fixed, epoch-varying seed for reproducible-but-changing sampling
    if improve:
        rows = sample_selection_candidates(test_nn, seed=seed)
    else:
        rows = sample_mechanism_candidates(test_nn, seed=seed)
    return pd.DataFrame(rows)

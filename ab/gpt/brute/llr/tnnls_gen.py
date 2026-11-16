"""
TNNLS-revision experiment generator (Route C: self-contained, no shared-pipeline edits).

Extends epoch_ablation_gen with the conditions the reviewer round asked for:

  control    llr_uniform at the base LR                          (the ablation's control)
  winning    the architecture's epoch-1 winning strategy         (the ablation's treatment)
  matched    the winning strategy's groups collapsed to ONE group at its parameter-mass-
             weighted mean multiplier (same mean step size, no layerwise structure)
  ulmfit     discriminative LR: 4 depth groups, ratio 2.6 per group, top group at base LR
  strat=<n>  any named strategy from the three llr modules (selection / comparison arms)

plus a configurable base LR (hp.txt + dataframe.df `lr`), train-epoch budget (used only
for the manifest; the budget itself is set by NNEval --nn_train_epochs) and seed range.

Why the code differs per condition/seed: NNEval dedups on uuid4(code), so each replicate
carries inert trailing assignments (`_replicate_seed`, and `_base_lr_tag` when the base LR
is not the sweep's 0.01) that change the checksum without changing behaviour. Use seed
indices that were not used before (the original 60-run ablation used 0-4).

Usage (pod-side; needs ab.nn for find_nn_source_dir):
    python -m ab.gpt.brute.llr.tnnls_gen --epoch-idx 970 --prefix llrtn \
        --archs GoogLeNet,ConvNeXt --conditions control,winning,matched,ulmfit \
        --base-lr 0.1 --seeds 1 --seed-start 200 --train-epochs 3
Then evaluate:
    python -m ab.gpt.NNEval --only_epoch 970 --nn_train_epochs 3 --dataset cifar-10 \
        --transform norm_64_flip --task img-classification --metric acc \
        --nn_name_prefix llrtn --save_to_db
"""

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from ab.gpt.brute.llr.epoch_ablation_gen import (
    ABLATION_ARCHS,
    CONTROL_STRATEGY,
    DATASET_CFG,
    _build_strategy_index,
)
from ab.gpt.brute.llr.layerwise_lr import (
    _find_method_range,
    _find_optimizer_block,
    inject_layerwise_lr as inject_generic,
    find_nn_source_dir,
    get_supported_hp_from_source,
    build_hp_dict,
    read_architecture_source,
)

SWEEP_BASE_LR = 0.01
ULMFIT_RATIO = 2.6

# Canonical discriminative fine-tuning (Howard & Ruder 2018): each earlier group's LR is
# the next group's divided by 2.6, top group at the base LR. Only the discriminative-LR
# rule is used (no slanted-triangular schedule, no gradual unfreezing).
ULMFIT_STRATEGY = {
    'name': 'ulmfit_4grp_26',
    'type': 'n_group',
    'n_groups': 4,
    'multipliers': [round(ULMFIT_RATIO ** -3, 6), round(ULMFIT_RATIO ** -2, 6),
                    round(ULMFIT_RATIO ** -1, 6), 1.0],
    'split_ratios': [0.25, 0.25, 0.25, 0.25],
    'description': 'ULMFiT discriminative LR: 4 equal groups, ratio 2.6 per group, head at 1.0x.',
}

# Inserted immediately before the optimizer line (8-space method-body indent). Collapses
# whatever groups the strategy built into a single group at the mass-weighted mean LR.
_MATCHED_SNIPPET = [
    "        # Matched-effective-LR control: ONE group at the parameter-mass-weighted mean",
    "        # multiplier of the strategy above (same mean step size, no layerwise structure).",
    "        _llr_base = prm.get('lr', 0.01)",
    "        _llr_numel = [sum(p.numel() for p in g['params']) for g in _llr_groups]",
    "        _llr_meff = sum(n * g['lr'] for n, g in zip(_llr_numel, _llr_groups)) / max(1, sum(_llr_numel)) / _llr_base",
    "        print(f'[llr-matched] m_eff={_llr_meff:.6f}', flush=True)",
    "        _llr_groups = [{'params': [p for g in _llr_groups for p in g['params']], 'lr': _llr_base * _llr_meff}]",
]


# Mean-preserving variant: keep the strategy's grouping and its multiplier RATIOS, but
# rescale every group so the parameter-mass-weighted mean multiplier is 1.0 (the same
# mean step size as llr_uniform). Some groups then run above the base LR. Together with
# `matched` (mean reduced, no structure) this gives a 2x2 of structure x mean rate.
_MEANPRES_SNIPPET = [
    "        # Mean-preserving control: same grouping and ratios, rescaled so the",
    "        # parameter-mass-weighted mean multiplier is 1.0 (groups may exceed the base LR).",
    "        _llr_base = prm.get('lr', 0.01)",
    "        _llr_numel = [sum(p.numel() for p in g['params']) for g in _llr_groups]",
    "        _llr_meff = sum(n * g['lr'] for n, g in zip(_llr_numel, _llr_groups)) / max(1, sum(_llr_numel)) / _llr_base",
    "        print(f'[llr-meanpres] m_eff={_llr_meff:.6f} max_mult={max(g[\"lr\"] for g in _llr_groups) / _llr_base / _llr_meff:.4f}', flush=True)",
    "        _llr_groups = [dict(g, lr=g['lr'] / _llr_meff) for g in _llr_groups]",
]

# Tuned recipe, appended after the optimizer is built: weight decay on every group, then a
# per-epoch LambdaLR (linear warmup, cosine decay to ~0 at the final epoch). LambdaLR scales
# each group's own initial LR, so layerwise ratios are kept. learn() is wrapped on the
# instance so the scheduler steps once per epoch (Train calls learn() once per epoch).
_RECIPE_SNIPPET = """\
# Tuned recipe (TNNLS revision): weight decay {wd}, {warmup}-epoch linear warmup, cosine over {epochs} epochs.
import math as _llr_math
for _llr_g in self.optimizer.param_groups:
    _llr_g['weight_decay'] = {wd}
_llr_E, _llr_W = {epochs}, {warmup}
self._llr_sched = torch.optim.lr_scheduler.LambdaLR(
    self.optimizer,
    lambda e: (e + 1) / _llr_W if e < _llr_W else 0.5 * (1 + _llr_math.cos(_llr_math.pi * (e - _llr_W) / max(1, _llr_E - _llr_W))))
_llr_inner_learn = self.learn
def _llr_learn(train_data, _inner=_llr_inner_learn):
    _out = _inner(train_data)
    self._llr_sched.step()
    return _out
self.learn = _llr_learn"""


def _insert_before_optimizer(code: str, snippet):
    """Insert the snippet before the optimizer line that consumes _llr_groups."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'self.optimizer' in line and '=' in line and '_llr_groups' in '\n'.join(lines[i:i + 4]):
            new = '\n'.join(lines[:i] + snippet + lines[i:])
            try:
                ast.parse(new)
            except SyntaxError:
                return None
            return new
    return None


def collapse_to_matched_uniform(code: str):
    return _insert_before_optimizer(code, _MATCHED_SNIPPET)


def rescale_to_mean_one(code: str):
    return _insert_before_optimizer(code, _MEANPRES_SNIPPET)


def swap_to_adamw(code: str):
    """Rewrite the train_setup() optimizer call from SGD to AdamW (momentum becomes beta1).
    ConvNeXt fallback when SGD + cosine misses the tuned bar; weight decay comes from the recipe."""
    lines = code.split('\n')
    m = next((re.match(r'^(\s*)class Net\b', l) for l in lines if re.match(r'^(\s*)class Net\b', l)), None)
    ts = _find_method_range(lines, 'train_setup', len(m.group(1))) if m else None
    blk = _find_optimizer_block(lines, *ts) if ts else None
    if blk is None:
        return None
    opt_i, opt_end = blk
    block = '\n'.join(lines[opt_i:opt_end + 1])
    if 'torch.optim.SGD(' not in block:
        return None
    # nn-dataset rejects a model whose code ignores a prm key, so momentum becomes AdamW's beta1
    # (0.9 -> betas (0.9, 0.999), the AdamW defaults) instead of being dropped.
    block = re.sub(r'momentum\s*=\s*([^,)]+)', r'betas=(\1, 0.999)', block.replace('torch.optim.SGD(', 'torch.optim.AdamW(', 1))
    new = '\n'.join(lines[:opt_i] + block.split('\n') + lines[opt_end + 1:])
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


def add_tuned_recipe(code: str, epochs: int, warmup: int, wd: float):
    """Append the recipe snippet right after the self.optimizer block in train_setup()."""
    lines = code.split('\n')
    m = next((re.match(r'^(\s*)class Net\b', l) for l in lines if re.match(r'^(\s*)class Net\b', l)), None)
    if m is None:
        return None
    ts = _find_method_range(lines, 'train_setup', len(m.group(1)))
    blk = _find_optimizer_block(lines, *ts) if ts else None
    if blk is None:
        return None
    opt_i, opt_end = blk
    indent = re.match(r'^(\s*)', lines[opt_i]).group(1)
    body = _RECIPE_SNIPPET.format(wd=wd, warmup=warmup, epochs=epochs)
    new = '\n'.join(lines[:opt_end + 1] + [indent + l for l in body.split('\n')] + lines[opt_end + 1:])
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


def _tags(seed: int, base_lr: float) -> str:
    tag = (
        "\n\n# TNNLS-revision replicate - behaviourally inert, distinct checksum only\n"
        f"_replicate_seed = {seed}\n"
    )
    if base_lr != SWEEP_BASE_LR:
        tag += f"_base_lr_tag = {base_lr}\n"
    return tag


def _resolve_condition(token: str, win_strategy: str, strat_index: dict):
    """-> (condition_label, strategy_name, strategy_dict, inject_fn, variant) or None.

    variant is '' (as defined), 'matched' (one group at the mean rate) or 'meanpres'
    (same groups rescaled to mean multiplier 1.0). 'meanpres' alone uses the winning
    strategy; 'meanpres=<name>' / 'strat=<name>' take any named strategy or ulmfit_4grp_26.
    """
    variant = ''
    if token == 'control':
        name = CONTROL_STRATEGY
    elif token in ('winning', 'matched', 'meanpres'):
        name = win_strategy
        variant = '' if token == 'winning' else token
    elif token == 'ulmfit':
        name = ULMFIT_STRATEGY['name']
    elif token.startswith(('strat=', 'meanpres=')):
        kind, name = token.split('=', 1)
        variant = 'meanpres' if kind == 'meanpres' else ''
    else:
        raise SystemExit(f"Unknown condition {token!r}")
    if name == ULMFIT_STRATEGY['name']:
        strat, inject_fn = ULMFIT_STRATEGY, inject_generic
    elif name in strat_index:
        strat, inject_fn = strat_index[name]
    else:
        print(f"[SKIP] strategy {name} not found in any module")
        return None
    if variant and strat['type'] == 'uniform':
        print(f"[SKIP] {token}: {name} has one group, nothing to rescale")
        return None
    label = token.split('=', 1)[0] if '=' in token else token
    return label, name, strat, inject_fn, variant


def generate(epoch_idx, archs, conditions, base_lr, seeds, seed_start, train_epochs,
             prefix, clean, force, extra=(), recipe='const', warmup=5, wd=5e-4, optimizer='sgd'):
    project_root = Path(__file__).resolve().parents[4]
    out_dir = project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / f'A{epoch_idx}' / 'synth_nn'
    out_dir.mkdir(parents=True, exist_ok=True)

    # NNEval evaluates EVERY dir under synth_nn, not just ours: a stray dir from an earlier
    # local dry run silently queues extra models onto the worker (cost us a test job before).
    strays = [d.name for d in out_dir.iterdir() if d.is_dir() and not d.name.startswith(f'{prefix}_')]
    if strays and not force:
        raise SystemExit(f"{out_dir} has non-{prefix}_ dirs {strays[:5]}; remove them or pass --force")
    existing = sorted(d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith(f'{prefix}_'))
    if clean:
        for d in existing:
            shutil.rmtree(d)
        if existing:
            print(f"Cleaned {len(existing)} existing {prefix}_ dirs")
        existing = []
    model_idx = len(existing) + 1   # --no-clean appends: keep names unique across invocations

    strat_index = _build_strategy_index()
    src_dir = find_nn_source_dir()
    wanted = {a.strip() for a in archs} if archs else None
    plan = [row for row in list(ABLATION_ARCHS) + list(extra) if wanted is None or row[0] in wanted]
    if not plan:
        raise SystemExit(f"No architectures matched --archs {archs!r}")

    manifest = []
    print(f"Output dir : {out_dir}\nArchs      : {[r[0] for r in plan]}\nConditions : {conditions}")
    print(f"Base LR    : {base_lr}   seeds {seed_start}..{seed_start + seeds - 1}   "
          f"train epochs {train_epochs}   recipe {recipe}" +
          (f" (warmup {warmup}, wd {wd})" if recipe == 'cosine' else '') + "\n")

    for arch, win_strategy, role in plan:
        source_code = read_architecture_source(src_dir, arch)
        if source_code is None or get_supported_hp_from_source(source_code) is None:
            print(f"[SKIP] {arch}: source/hyperparameters not found")
            continue
        for token in conditions:
            res = _resolve_condition(token, win_strategy, strat_index)
            if res is None:
                continue
            label, strat_name, strat, inject_fn, variant = res
            base_code = inject_fn(source_code, strat)
            if base_code is not None and variant == 'matched':
                base_code = collapse_to_matched_uniform(base_code)
            elif base_code is not None and variant == 'meanpres':
                base_code = rescale_to_mean_one(base_code)
            if base_code is not None and optimizer == 'adamw':
                base_code = swap_to_adamw(base_code)
            if base_code is not None and recipe == 'cosine':
                # The epoch count is baked into the schedule, so here --train-epochs changes
                # behaviour and must equal NNEval's --nn_train_epochs.
                base_code = add_tuned_recipe(base_code, train_epochs, warmup, wd)
            if base_code is None:
                print(f"[SKIP] {arch}/{token}: injection failed")
                continue
            shown_name = strat_name + (f'_{variant}' if variant else '')
            for seed in range(seed_start, seed_start + seeds):
                name = f"{prefix}_{model_idx:04d}"
                mdir = out_dir / name
                mdir.mkdir(parents=True, exist_ok=True)
                (mdir / 'new_nn.py').write_text(base_code + _tags(seed, base_lr), encoding='utf-8')
                hp = build_hp_dict(arch, DATASET_CFG)
                hp['lr'] = base_lr
                (mdir / 'hp.txt').write_text(json.dumps(hp, indent=2), encoding='utf-8')
                pd.Series({
                    'nn': f"{arch}-{shown_name}", 'task': DATASET_CFG['task'],
                    'dataset': DATASET_CFG['name'], 'metric': DATASET_CFG['metric'], 'prm': hp,
                }).to_pickle(str(mdir / 'dataframe.df'))
                (mdir / 'model_meta.txt').write_text('\n'.join([
                    f"architecture: {arch}", f"strategy: {shown_name}", f"condition: {label}",
                    f"base_lr: {base_lr}", f"seed: {seed}", f"train_epochs: {train_epochs}",
                    f"recipe: {recipe}", f"optimizer: {optimizer}", f"multipliers: {strat.get('multipliers')}",
                ]) + '\n', encoding='utf-8')
                manifest.append({
                    'model_dir': name, 'architecture': arch, 'strategy': shown_name,
                    'condition': label, 'role': role, 'base_lr': base_lr, 'seed': seed,
                    'train_epochs': train_epochs, 'recipe': recipe + ('+adamw' if optimizer == 'adamw' else ''), 'dataset': DATASET_CFG['name'],
                    'transform': DATASET_CFG['transform'], 'prefix': prefix,
                })
                model_idx += 1
        print(f"[OK] {arch} ({role})")

    manifest_path = project_root / 'out' / 'nngpt' / f'tnnls_manifest_A{epoch_idx}.csv'
    mode = 'a' if manifest_path.exists() and not clean else 'w'
    with open(manifest_path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        if mode == 'w':
            w.writeheader()
        w.writerows(manifest)
    print(f"\nGenerated {len(manifest)} model dirs into {out_dir}\nManifest : {manifest_path}")
    return len(manifest)


def main():
    p = argparse.ArgumentParser(description="Generate TNNLS-revision model dirs.")
    p.add_argument('--epoch-idx', type=int, required=True, help="A{N}/synth_nn slot (use >=970).")
    p.add_argument('--archs', type=str, default='', help="Comma-separated subset of the 6 ablation archs.")
    p.add_argument('--conditions', type=str, default='control,winning',
                   help="Comma list of: control, winning, matched, meanpres, ulmfit, "
                        "strat=<name>, meanpres=<name>.")
    p.add_argument('--base-lr', type=float, default=SWEEP_BASE_LR)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=100,
                   help="First seed index (0-4 were used by the original ablation).")
    p.add_argument('--train-epochs', type=int, default=10,
                   help="Manifest only for --recipe const; the cosine horizon for --recipe cosine.")
    p.add_argument('--recipe', choices=['const', 'cosine'], default='const',
                   help="const = the paper's constant-rate SGD; cosine = + weight decay, warmup, cosine.")
    p.add_argument('--warmup', type=int, default=5, help="Warmup epochs (cosine recipe).")
    p.add_argument('--wd', type=float, default=5e-4, help="Weight decay (cosine recipe).")
    p.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd',
                   help="adamw rewrites the SGD call (ConvNeXt fallback); use with --wd 0.05.")
    p.add_argument('--prefix', type=str, default='llrtn')
    p.add_argument('--no-clean', dest='clean', action='store_false',
                   help="Append to an existing slot instead of clearing our own prefix dirs.")
    p.add_argument('--force', action='store_true', help="Allow non-prefix stray dirs in the slot.")
    p.add_argument('--extra-arch', action='append', default=[], metavar='ARCH:STRATEGY:ROLE',
                   help="Add an architecture outside the six-arch cohort with its pre-specified "
                        "'winning' strategy (held-out tests). Repeatable.")
    a = p.parse_args()
    archs = [x for x in a.archs.split(',') if x.strip()]
    conds = [x.strip() for x in a.conditions.split(',') if x.strip()]
    extra = [tuple(x.split(':')) for x in a.extra_arch]
    if any(len(x) != 3 for x in extra):
        raise SystemExit('--extra-arch takes ARCH:STRATEGY:ROLE')
    generate(a.epoch_idx, archs, conds, a.base_lr, a.seeds, a.seed_start, a.train_epochs,
             a.prefix, a.clean, a.force, extra, a.recipe, a.warmup, a.wd, a.optimizer)


if __name__ == '__main__':
    main()

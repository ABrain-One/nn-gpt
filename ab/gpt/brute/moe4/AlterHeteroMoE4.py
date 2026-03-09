"""
AlterHeteroMoE4.py — Programmatic 4-Expert Heterogeneous MoE Generator

Generates heterogeneous Mixture-of-Experts neural networks by combining
quartets of UUID-variant architectures from nn-dataset into a single
MoE model with a learned gating network, mixup augmentation, and
warmup + cosine LR scheduling.

No LLM involved — pure deterministic code assembly.

Expert pool:  UUID-variant models (e.g. AlexNet-355f849c-…) filtered to
              img-classification / cifar-10 / acc from the LEMUR dataset API.
Constraint:   All 4 experts must come from **different** base-model families
              (cross-family only) to maximise architectural diversity.

Pattern follows AlterNNFN.py / AlterHeteroMoE.py:
  combinatorial generation → compile check → write new_nn.py
Evaluation via existing NNEval.py:
  python -m ab.gpt.NNEval --custom_synth_dir <output> --nn_name_prefix MoE4
"""

import re
import itertools
from pathlib import Path

import torch
import ab.nn.api as api
from ab.nn.util.Util import uuid4
from ab.gpt.util.Const import new_nn_file, nngpt_dir

# ── Reuse pipeline functions from the 2-expert generator ─────────────────────
from ab.gpt.brute.moe.AlterHeteroMoE import (
    transform_expert,
    _extract_prm_keys,
    _prm_default,
    _helper_classes,
)

# ── Output directory (separate from 2-expert generator) ──────────────────────
_DEFAULT_OUT_DIR = nngpt_dir / 'hetero_moe4' / 'synth_nn'

# ── Task / dataset constants ─────────────────────────────────────────────────
TASK = 'img-classification'
DATASET = 'cifar-10'
METRIC = 'acc'

# ── Families to exclude from the expert pool ─────────────────────────────────
_EXCLUDE_FAMILIES = frozenset({'FractalNet', 'MoE', 'MoEv3'})


# ══════════════════════════════════════════════════════════════════════════════
# Multi-expert import merging  (extends 2-expert _merge_imports)
# ══════════════════════════════════════════════════════════════════════════════

def _merge_imports_multi(import_lists: list[list[str]]) -> str:
    """Deduplicate and sort import lines from *N* experts."""
    seen: dict[str, str] = {}
    for imp_list in import_lists:
        for ln in imp_list:
            key = ln.strip()
            if key:
                seen[key] = key
    return '\n'.join(sorted(seen.values()))


# ══════════════════════════════════════════════════════════════════════════════
# Multi-expert class-collision detection
# ══════════════════════════════════════════════════════════════════════════════

def _has_any_class_collision(codes: list[str]) -> bool:
    """True if **any** pair of expert codes shares a helper class name."""
    helper_sets = [_helper_classes(c) for c in codes]
    for i in range(len(helper_sets)):
        for j in range(i + 1, len(helper_sets)):
            if helper_sets[i] & helper_sets[j]:
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 4-Expert MoE Wrapper Template
#
# Enhanced over the 2-expert wrapper with features extracted from the
# reference model  MoE-hetero4-Alex-Dense-Air-Bag.py :
#   • n_experts = 4
#   • Mixup augmentation in learn()
#   • Warmup (LinearLR) + Cosine Annealing LR scheduling
# ══════════════════════════════════════════════════════════════════════════════

_MOE4_WRAPPER = '''\


# ============================================================================
# HETEROGENEOUS MOE GATE
# ============================================================================
class HeterogeneousGate(nn.Module):
    """Lightweight CNN-based gating network that routes inputs to 4 experts."""
    def __init__(self, input_channels, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x):
        features = self.gate_features(x).flatten(1)
        logits = self.gate(features) / torch.clamp(self.temperature, 0.5, 5.0)
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1
        return F.softmax(logits, dim=-1), logits


# ============================================================================
# HETEROGENEOUS MOE NET — {expert1_name} + {expert2_name} + {expert3_name} + {expert4_name}
# ============================================================================
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        n_experts = 4

        # Defaults for expert-specific hyperparameters
{prm_defaults}
        self.experts = nn.ModuleList([
            {expert1_name}Expert(in_shape, out_shape, prm, device),
            {expert2_name}Expert(in_shape, out_shape, prm, device),
            {expert3_name}Expert(in_shape, out_shape, prm, device),
            {expert4_name}Expert(in_shape, out_shape, prm, device),
        ])
        self.gate = HeterogeneousGate(in_shape[1], n_experts=n_experts)

        self.load_balance_weight = 0.01
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights, _ = self.gate(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        return torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)

    def _load_balance_loss(self):
        if not hasattr(self, '_last_gw'):
            return torch.tensor(0.0, device=self.device)
        gw = self._last_gw
        usage = gw.sum(dim=0)
        target = gw.sum() / len(self.experts)
        return F.mse_loss(usage, target.expand_as(usage))

    def _mixup_data(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=self.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        expert_params, gate_params = [], list(self.gate.parameters())
        for e in self.experts:
            expert_params.extend(list(e.parameters()))
        self.optimizer = torch.optim.AdamW([
            {'params': expert_params, 'lr': prm.get('lr', 0.001), 'weight_decay': 5e-4},
            {'params': gate_params, 'lr': prm.get('lr', 0.001) * 2, 'weight_decay': 2.5e-4},
        ])
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5
        )
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self._current_epoch = 0

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            if self.mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
            self._last_gw = self.gate(inputs)[0].detach()
            loss = loss + self.load_balance_weight * self._load_balance_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        self._current_epoch += 1
        if self._current_epoch <= 5:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
'''


# ══════════════════════════════════════════════════════════════════════════════
# Assemble a complete 4-expert MoE file
# ══════════════════════════════════════════════════════════════════════════════

def assemble_moe4(names: list[str], codes: list[str]) -> str:
    """
    Build a self-contained MoE Python file combining four experts.

    Parameters
    ----------
    names : list[str]   — 4 expert display names  (e.g. ``['AirNet', 'AlexNet', 'BagNet', 'DenseNet']``)
    codes : list[str]   — 4 corresponding source codes from the API

    Returns
    -------
    str — full source code ready to be written as ``new_nn.py``.
    """
    assert len(names) == 4 and len(codes) == 4

    transformed = [transform_expert(n, c) for n, c in zip(names, codes)]
    import_lists = [t[0] for t in transformed]
    bodies = [t[1] for t in transformed]

    merged_imports = _merge_imports_multi(import_lists)
    # Ensure core MoE imports are present
    for req in ['import torch',
                'import torch.nn as nn',
                'import torch.nn.functional as F',
                'import numpy as np']:
        if req not in merged_imports:
            merged_imports = req + '\n' + merged_imports

    combo_str = ' + '.join(names)
    header = (
        f'# Auto-generated 4-Expert Heterogeneous MoE: {combo_str}\n'
        f'# Four-expert mixture with learned gating, mixup, and LR scheduling\n'
    )

    # Build supported_hyperparameters dynamically from actual prm usage
    all_prm_keys: set[str] = set()
    for body in bodies:
        all_prm_keys |= _extract_prm_keys(body)
    all_prm_keys |= _extract_prm_keys(_MOE4_WRAPPER)
    all_prm_keys.add('lr')  # always needed by wrapper train_setup
    hp_set_str = ', '.join(f"'{k}'" for k in sorted(all_prm_keys))
    sup_hp = (
        "\n\ndef supported_hyperparameters():\n"
        f"    return {{{hp_set_str}}}\n"
    )

    # Generate prm.setdefault() lines for non-standard expert hyperparameters
    _STANDARD_KEYS = {'lr', 'momentum', 'dropout', 'batch', 'transform', 'epoch'}
    extra_keys = all_prm_keys - _STANDARD_KEYS
    if extra_keys:
        defaults_lines = []
        for k in sorted(extra_keys):
            defaults_lines.append(f"        prm.setdefault('{k}', {_prm_default(k)!r})")
        prm_defaults_block = '\n'.join(defaults_lines)
    else:
        prm_defaults_block = '        pass  # no extra defaults needed'

    wrapper = _MOE4_WRAPPER.replace('{prm_defaults}', prm_defaults_block)
    for i, name in enumerate(names, start=1):
        wrapper = wrapper.replace(f'{{expert{i}_name}}', name)

    parts = [header, merged_imports, sup_hp]
    for idx, (name, body) in enumerate(zip(names, bodies), start=1):
        parts += [
            f'\n\n# {"=" * 76}',
            f'# EXPERT {idx}: {name}',
            f'# {"=" * 76}',
            body,
        ]
    parts.append(wrapper)

    return '\n'.join(parts) + '\n'


# ══════════════════════════════════════════════════════════════════════════════
# Expert-pool query
# ══════════════════════════════════════════════════════════════════════════════

def _family_of(nn_name: str) -> str:
    """Extract the base-model family from a UUID-variant name.

    ``'AlexNet-355f849c-18de-4d93-b571-b7dd5db94569'``  →  ``'AlexNet'``
    """
    return re.sub(r'-[0-9a-f]{8}.*', '', nn_name)


def _get_uuid_variant_models() -> dict[str, dict[str, str]]:
    """Fetch UUID-variant models for CIFAR-10 img-classification, grouped by family.

    Returns ``{family: {full_name: source_code, …}, …}``
    excluding families in ``_EXCLUDE_FAMILIES``.
    """
    df = api.data(only_best_accuracy=True, task=TASK, dataset=DATASET, metric=METRIC)
    all_names = sorted(df['nn'].unique())

    # Filter to UUID-variant names only  (e.g.  AirNet-626c3eb9-…)
    uuid_pattern = re.compile(r'^[A-Z].*-[0-9a-f]{8}')
    uuid_names = [n for n in all_names if uuid_pattern.match(n)]

    families: dict[str, dict[str, str]] = {}
    for nn_name in uuid_names:
        fam = _family_of(nn_name)
        if fam in _EXCLUDE_FAMILIES:
            continue
        row = df[df['nn'] == nn_name].iloc[0]
        code = row['nn_code']
        if isinstance(code, str) and code.strip():
            families.setdefault(fam, {})[nn_name] = code
    return families


# ══════════════════════════════════════════════════════════════════════════════
# Forward-pass validation (catches shape / dimension errors before writing)
# ══════════════════════════════════════════════════════════════════════════════

# CIFAR-10 dimensions used for the validation probe
_PROBE_BATCH = 2
_PROBE_IN_SHAPE = (_PROBE_BATCH, 3, 32, 32)   # (N, C, H, W)
_PROBE_OUT_SHAPE = (10,)                       # 10 classes


def _forward_pass_ok(moe_code: str) -> tuple[bool, str]:
    """Instantiate the assembled model and run a forward pass on CPU.

    Returns ``(True, '')`` on success or ``(False, reason)`` on failure.
    """
    ns: dict = {}
    try:
        exec(compile(moe_code, '<moe4_probe>', 'exec'), ns)
    except Exception as e:
        return False, f'exec: {e}'

    NetClass = ns.get('Net')
    if NetClass is None:
        return False, 'no Net class'

    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.2}
    try:
        model = NetClass(_PROBE_IN_SHAPE, _PROBE_OUT_SHAPE, prm, torch.device('cpu'))
    except Exception as e:
        return False, f'init: {e}'

    try:
        x = torch.randn(_PROBE_IN_SHAPE)
        with torch.no_grad():
            out = model(x)
    except Exception as e:
        return False, f'forward: {e}'

    expected = (_PROBE_BATCH, _PROBE_OUT_SHAPE[0])
    if tuple(out.shape) != expected:
        return False, f'output shape {tuple(out.shape)} != expected {expected}'

    return True, ''


# ══════════════════════════════════════════════════════════════════════════════
# Main generator
# ══════════════════════════════════════════════════════════════════════════════

def alter(max_variants: int = 500, out_dir: str | Path | None = None):
    """
    Generate 4-expert heterogeneous MoE models from cross-family UUID-variant
    expert quartets.

    Iterates through all ``C(n_families, 4)`` family combinations.  Within
    each family combination, forms the Cartesian product of variant models
    and writes each assembled model to ``out_dir / B{n} / new_nn.py``.

    Parameters
    ----------
    max_variants : int
        Maximum number of MoE models to generate (default 500).
    out_dir : str | Path | None
        Output directory.  Defaults to ``out/nngpt/hetero_moe4/synth_nn``.

    Returns
    -------
    int — number of models successfully generated.
    """
    out_dir = Path(out_dir) if out_dir else _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    families = _get_uuid_variant_models()
    fam_names = sorted(families.keys())
    n_fam = len(fam_names)
    total_variants = sum(len(v) for v in families.values())

    print(f'Found {total_variants} UUID variants across {n_fam} families')
    for fn in fam_names:
        print(f'  {fn}: {len(families[fn])} variants')

    from math import comb
    print(f'Cross-family 4-combos (families): C({n_fam},4) = {comb(n_fam, 4)}')

    # Load existing DB checksums so we never generate duplicates
    api.data.cache_clear()
    existing_ids = set(api.data()['nn_id'].unique().tolist())
    print(f'Existing NNs in DB: {len(existing_ids)}')

    counter = 0
    skipped_collision = 0
    skipped_compile = 0
    skipped_forward = 0
    skipped_duplicate = 0

    for fam_combo in itertools.combinations(fam_names, 4):
        if counter >= max_variants:
            break

        # Cartesian product of variants within the 4 chosen families
        variant_lists = [list(families[f].items()) for f in fam_combo]
        for variant_quartet in itertools.product(*variant_lists):
            if counter >= max_variants:
                break

            names = [_family_of(v[0]) for v in variant_quartet]
            full_names = [v[0] for v in variant_quartet]
            codes = [v[1] for v in variant_quartet]

            # Class-collision check across all 4 experts
            if _has_any_class_collision(codes):
                collision_str = ' + '.join(full_names)
                print(f'  SKIP (class collision): {collision_str}')
                skipped_collision += 1
                continue

            moe_code = assemble_moe4(names, codes)

            # Syntax check
            combo_label = ' + '.join(full_names)
            try:
                compile(moe_code, f'MoE4_{"_".join(names)}.py', 'exec')
            except SyntaxError as e:
                print(f'  SKIP (syntax error): {combo_label}: {e}')
                skipped_compile += 1
                continue

            # Forward-pass validation — catches shape/dimension errors
            ok, reason = _forward_pass_ok(moe_code)
            if not ok:
                print(f'  SKIP (forward pass): {combo_label}: {reason}')
                skipped_forward += 1
                continue

            # DB dedup — skip models whose code already exists in the database
            checksum = uuid4(moe_code)
            if checksum in existing_ids:
                print(f'  SKIP (duplicate): {combo_label} (checksum: {checksum})')
                skipped_duplicate += 1
                continue
            existing_ids.add(checksum)  # prevent intra-batch duplicates too

            model_dir = out_dir / f'B{counter}'
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / new_nn_file).write_text(moe_code)

            print(f'B{counter}: {combo_label}')
            counter += 1

    print(f'\nGenerated: {counter}  |  Skipped (collision): {skipped_collision}'
          f'  |  Skipped (syntax): {skipped_compile}'
          f'  |  Skipped (forward): {skipped_forward}'
          f'  |  Skipped (duplicate): {skipped_duplicate}')
    print(f'Output: {out_dir.resolve()}')
    return counter

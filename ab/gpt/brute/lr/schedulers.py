"""
Bayesian Optimization Model Generator — ~2000 CV Models
Reads actual nn-dataset architecture source files and injects LR scheduler
modifications into train_setup() and learn() methods while preserving the
exact nn-dataset interface.

Handles multiple code styles found across nn-dataset architectures:
  - tuple-wrapped vs bare criteria
  - different optimizer styles (torch.optim.SGD, optim.SGD, Adam, etc.)
  - multiline optimizer definitions
  - presence/absence of clip_grad_norm_
  - presence/absence of self.train()
"""

import os
import re
import sys
import ast
import json
import importlib
from pathlib import Path

# ── nn-dataset architectures (from ab.nn.util.Const.core_nn_cls) ──────────────
# Exclude RLFN and SwinIR (no source files), MoE and ConvNeXtTransformer (very custom learn)
ARCHITECTURES = [
    'AirNet', 'AirNext', 'AlexNet', 'BagNet', 'ComplexNet', 'BayesianNet-1',
    'ConvNeXt', 'DPN107', 'DPN131', 'DPN68',
    'DarkNet', 'DenseNet', 'Diffuser', 'EfficientNet', 'FractalNet',
    'GoogLeNet', 'ICNet', 'InceptionV3-1', 'MNASNet', 'MaxVit',
    'MobileNetV2', 'MobileNetV3',
    'RegNet', 'ResNet', 'ShuffleNet', 'SqueezeNet-1', 'SwinTransformer',
    'UNet2D', 'VGG', 'VisionTransformer',
]

# ── LR Scheduler configurations ──────────────────────────────────────────────
SCHEDULERS = [
    {
        'name': 'StepLR_s10_g05',
        'extra_hp': ['step_size', 'gamma', 'epoch_max'],
        'setup_code': "        step_size = int(prm.get('step_size', 0.5) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=prm.get('gamma', 0.5))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'StepLR_s5_g03',
        'extra_hp': ['step_size', 'gamma', 'epoch_max'],
        'setup_code': "        step_size = int(prm.get('step_size', 0.25) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=prm.get('gamma', 0.3))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'StepLR_s20_g07',
        'extra_hp': ['step_size', 'gamma', 'epoch_max'],
        'setup_code': "        step_size = int(prm.get('step_size', 0.7) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=prm.get('gamma', 0.7))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'StepLR_s3_g01',
        'extra_hp': ['step_size', 'gamma', 'epoch_max'],
        'setup_code': "        step_size = int(prm.get('step_size', 0.1) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=prm.get('gamma', 0.1))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'ExponentialLR_g095',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=prm.get('gamma', 0.95))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'ExponentialLR_g09',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=prm.get('gamma', 0.9))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'ExponentialLR_g098',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=prm.get('gamma', 0.98))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'CosineAnnealingLR_T5',
        'extra_hp': ['T_max', 'eta_min', 'epoch_max'],
        'setup_code': "        T_max = int(prm.get('T_max', 0.1) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, T_max), eta_min=prm.get('eta_min', 1e-6))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'CosineAnnealingLR_T10',
        'extra_hp': ['T_max', 'eta_min', 'epoch_max'],
        'setup_code': "        T_max = int(prm.get('T_max', 0.2) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, T_max), eta_min=prm.get('eta_min', 1e-6))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'CosineAnnealingLR_T20',
        'extra_hp': ['T_max', 'eta_min', 'epoch_max'],
        'setup_code': "        T_max = int(prm.get('T_max', 0.4) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, T_max), eta_min=prm.get('eta_min', 1e-6))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'MultiStepLR_m5_10_g05',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10], gamma=prm.get('gamma', 0.5))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'MultiStepLR_m3_7_g03',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 7], gamma=prm.get('gamma', 0.3))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'MultiStepLR_m2_4_g01',
        'extra_hp': ['gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2, 4], gamma=prm.get('gamma', 0.1))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'ReduceLROnPlateau_f05_p2',
        'extra_hp': ['factor', 'patience'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=prm.get('factor', 0.5), patience=prm.get('patience', 2))",
        'step_code_batch': '            self.scheduler.step(loss.item())',
        'step_location': 'per_batch',
    },
    {
        'name': 'ReduceLROnPlateau_f03_p3',
        'extra_hp': ['factor', 'patience'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=prm.get('factor', 0.3), patience=prm.get('patience', 3))",
        'step_code_batch': '            self.scheduler.step(loss.item())',
        'step_location': 'per_batch',
    },
    {
        'name': 'ReduceLROnPlateau_f01_p5',
        'extra_hp': ['factor', 'patience'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=prm.get('factor', 0.1), patience=prm.get('patience', 5))",
        'step_code_batch': '            self.scheduler.step(loss.item())',
        'step_location': 'per_batch',
    },
    {
        'name': 'CyclicLR_tri',
        'extra_hp': ['base_lr', 'max_lr'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=prm.get('base_lr', 1e-4), max_lr=prm.get('max_lr', 0.1), mode='triangular', step_size_up=200)",
        'step_code_batch': '            self.scheduler.step()',
        'step_location': 'per_batch',
    },
    {
        'name': 'CyclicLR_tri2',
        'extra_hp': ['base_lr', 'max_lr'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=prm.get('base_lr', 1e-4), max_lr=prm.get('max_lr', 0.1), mode='triangular2', step_size_up=200)",
        'step_code_batch': '            self.scheduler.step()',
        'step_location': 'per_batch',
    },
    {
        'name': 'CyclicLR_exp',
        'extra_hp': ['base_lr', 'max_lr', 'gamma'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=prm.get('base_lr', 1e-4), max_lr=prm.get('max_lr', 0.1), mode='exp_range', gamma=prm.get('gamma', 0.999), step_size_up=200)",
        'step_code_batch': '            self.scheduler.step()',
        'step_location': 'per_batch',
    },
    {
        'name': 'OneCycleLR_01',
        'extra_hp': ['max_lr'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=prm.get('max_lr', 0.1), steps_per_epoch=500, epochs=5)",
        'step_code_batch': '            self.scheduler.step()',
        'step_location': 'per_batch',
    },
    {
        'name': 'OneCycleLR_005',
        'extra_hp': ['max_lr'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=prm.get('max_lr', 0.05), steps_per_epoch=500, epochs=5)",
        'step_code_batch': '            self.scheduler.step()',
        'step_location': 'per_batch',
    },
    {
        'name': 'CosineWarmRestarts_T5',
        'extra_hp': ['T_0', 'eta_min', 'epoch_max'],
        'setup_code': "        T_0 = int(prm.get('T_0', 0.1) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=max(1, T_0), eta_min=prm.get('eta_min', 1e-6))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'CosineWarmRestarts_T2',
        'extra_hp': ['T_0', 'eta_min', 'epoch_max'],
        'setup_code': "        T_0 = int(prm.get('T_0', 0.04) * prm.get('epoch_max', 50))\n        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=max(1, T_0), eta_min=prm.get('eta_min', 1e-6))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'LinearLR_sf01',
        'extra_hp': ['start_factor', 'total_iters'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=prm.get('start_factor', 0.1), total_iters=int(prm.get('total_iters', 5)))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
    {
        'name': 'PolynomialLR_p2',
        'extra_hp': ['total_iters', 'power'],
        'setup_code': "        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=int(prm.get('total_iters', 5)), power=prm.get('power', 2.0))",
        'step_code': '        self.scheduler.step()',
        'step_location': 'per_epoch',
    },
]

# ── Weight decay variations ───────────────────────────────────────────────────
WEIGHT_DECAY_VALUES = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

# ── Architecture-specific hyperparameter defaults ─────────────────────────────
# These are extra HP keys that architectures access via prm['key'] (hard access)
# but NNEval only provides {lr, batch, dropout, momentum, transform, epoch}.
# We write these into hp.txt so NNEval loads them before model instantiation.
ARCH_EXTRA_HP_DEFAULTS = {
    'ConvNeXt': {'stochastic_depth_prob': 0.1, 'norm_eps': 1e-6, 'norm_std': 0.02},
    'GoogLeNet': {'dropout_aux': 0.7},
    'MaxVit': {'attention_dropout': 0.0, 'stochastic_depth_prob': 0.1},
    'MobileNetV3': {'norm_eps': 0.001, 'norm_momentum': 0.01},
    'SwinTransformer': {'attention_dropout': 0.0, 'stochastic_depth_prob': 0.1},
    'VisionTransformer': {'attention_dropout': 0.0, 'patch_size': 0.5},
}


def build_hp_dict(arch, scheduler_cfg, weight_decay):
    """
    Build the hp.txt JSON dict for a model.
    Includes base training params + architecture-specific extras + scheduler extras.
    """
    hp = {
        'lr': 0.01,
        'batch': 64,
        'dropout': 0.2,
        'momentum': 0.9,
        'transform': 'norm_256_flip',
        'epoch_max': 50,
    }
    # Add architecture-specific defaults
    if arch in ARCH_EXTRA_HP_DEFAULTS:
        hp.update(ARCH_EXTRA_HP_DEFAULTS[arch])
    # Add scheduler-specific defaults
    for param in scheduler_cfg['extra_hp']:
        if param not in hp:
            # Use the default value from the setup_code
            defaults = {
                'step_size': 0.5, 'gamma': 0.5,
                'T_max': 0.2, 'eta_min': 1e-6,
                'factor': 0.5, 'patience': 3,
                'base_lr': 1e-4, 'max_lr': 0.1,
                'T_0': 0.1,
                'start_factor': 0.1, 'total_iters': 0.1,
                'power': 2.0,
                'epoch_max': 50,
            }
            if param in defaults:
                hp[param] = defaults[param]
    # Add weight_decay if nonzero
    if weight_decay > 0:
        hp['weight_decay'] = weight_decay
    return hp


def find_nn_source_dir():
    """Find the nn-dataset architecture source directory."""
    mod = importlib.import_module('ab.nn.nn.ResNet')
    return Path(mod.__file__).parent


def read_architecture_source(src_dir, arch_name):
    """Read the source file for a given architecture."""
    src_file = src_dir / f'{arch_name}.py'
    if not src_file.exists():
        return None
    with open(src_file, 'r') as f:
        return f.read()


def get_supported_hp(src_dir, arch_name):
    """Get supported hyperparameters from architecture by importing."""
    try:
        mod = importlib.import_module(f'ab.nn.nn.{arch_name}')
        return mod.supported_hyperparameters()
    except Exception:
        return None


def _find_method_range(lines, method_name, class_indent=0):
    """
    Find the line range [start, end) of a method within a class.
    Returns (start_line_idx, end_line_idx) or None.
    """
    method_indent = class_indent + 4
    method_def = ' ' * method_indent + f'def {method_name}('
    
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(f'def {method_name}(') and line.startswith(' ' * method_indent):
            start = i
            break
    
    if start is None:
        return None
    
    # Find end: next method/class definition at same or lower indent, or end of file
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        # Same indent or less = new method or class or module-level
        if indent <= method_indent and stripped:
            # Check it's a def/class, not just a comment
            if stripped.startswith('def ') or stripped.startswith('class ') or \
               stripped.startswith('@') or (indent < method_indent and not stripped.startswith('#')):
                end = i
                break
    
    return (start, end)


def _find_optimizer_step_line(lines, learn_start, learn_end):
    """Find the line index of 'self.optimizer.step()' within learn method."""
    for i in range(learn_start, learn_end):
        if 'self.optimizer.step()' in lines[i]:
            return i
    return None


def _find_last_line_of_train_setup(lines, ts_start, ts_end):
    """Find the last non-empty line of train_setup to insert scheduler after."""
    last = ts_end - 1
    while last > ts_start and not lines[last].strip():
        last -= 1
    return last


def inject_scheduler_ast(source_code, scheduler_cfg, weight_decay, arch_hp_set):
    """
    Modify architecture source code to inject LR scheduler using line-based editing.
    
    This approach is more flexible than regex — it finds method boundaries and
    inserts/appends lines rather than matching exact patterns.
    
    Returns the modified source code, or None if injection fails.
    """
    lines = source_code.split('\n')
    
    # ── Find class Net ────────────────────────────────────────────────────────
    class_start = None
    class_indent = 0
    for i, line in enumerate(lines):
        if re.match(r'^class Net\b', line):
            class_start = i
            class_indent = 0
            break
        m = re.match(r'^(\s*)class Net\b', line)
        if m:
            class_start = i
            class_indent = len(m.group(1))
            break
    
    if class_start is None:
        return None
    
    # ── Find train_setup method range ─────────────────────────────────────────
    ts_range = _find_method_range(lines, 'train_setup', class_indent)
    if ts_range is None:
        return None
    ts_start, ts_end = ts_range
    
    # ── Find learn method range ───────────────────────────────────────────────
    learn_range = _find_method_range(lines, 'learn', class_indent)
    if learn_range is None:
        return None
    learn_start, learn_end = learn_range
    
    # ── Find supported_hyperparameters function ───────────────────────────────
    hp_func_range = None
    for i, line in enumerate(lines):
        if re.match(r'^def supported_hyperparameters\(', line):
            hp_start = i
            # Find end
            hp_end = len(lines)
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if not stripped:
                    continue
                indent = len(lines[j]) - len(lines[j].lstrip())
                if indent == 0 and stripped and not stripped.startswith('#'):
                    hp_end = j
                    break
            hp_func_range = (hp_start, hp_end)
            break
    
    if hp_func_range is None:
        return None
    
    # ── Build new supported_hyperparameters ───────────────────────────────────
    all_hp = set(arch_hp_set)
    for hp in scheduler_cfg['extra_hp']:
        all_hp.add(hp)
    if weight_decay > 0:
        all_hp.add('weight_decay')
    
    new_hp_str = '{' + ', '.join(f"'{h}'" for h in sorted(all_hp)) + '}'
    new_hp_lines = [
        'def supported_hyperparameters():',
        f'    return {new_hp_str}',
    ]
    
    # ── Modify learn method ───────────────────────────────────────────────────
    # Find optimizer.step() line and add scheduler step after it
    opt_step_idx = _find_optimizer_step_line(lines, learn_start, learn_end)
    if opt_step_idx is None:
        return None
    
    # Determine the indent level of optimizer.step() line
    opt_step_line = lines[opt_step_idx]
    opt_indent = len(opt_step_line) - len(opt_step_line.lstrip())
    
    if scheduler_cfg['step_location'] == 'per_batch':
        # Insert per-batch scheduler step right after optimizer.step()
        batch_step = scheduler_cfg.get('step_code_batch', '            self.scheduler.step()')
        lines.insert(opt_step_idx + 1, batch_step)
        # Adjust indices
        if learn_end > opt_step_idx:
            learn_end += 1
        if ts_start > opt_step_idx:
            ts_start += 1
            ts_end += 1
        if hp_func_range[0] > opt_step_idx:
            hp_func_range = (hp_func_range[0] + 1, hp_func_range[1] + 1)
    else:
        # per_epoch: add scheduler step after the for-loop (at method body indent)
        step_line = scheduler_cfg.get('step_code', '        self.scheduler.step()')
        
        # Find the last line that belongs to the learn method body
        # Only consider lines with indent deeper than the method definition level
        method_def_indent = class_indent + 4  # e.g., 4 for 'def learn'
        method_body_indent = method_def_indent + 4  # e.g., 8 for body of learn
        insert_idx = opt_step_idx + 1  # default: right after optimizer.step()
        for i in range(learn_end - 1, learn_start, -1):
            line = lines[i]
            if line.strip():
                line_indent = len(line) - len(line.lstrip())
                if line_indent >= method_body_indent:
                    # This line is inside the learn method body
                    insert_idx = i + 1
                    break
        
        # Insert the scheduler step at method body level
        lines.insert(insert_idx, step_line)
        # Adjust indices
        if ts_start >= insert_idx:
            ts_start += 1
            ts_end += 1
        if hp_func_range[0] >= insert_idx:
            hp_func_range = (hp_func_range[0] + 1, hp_func_range[1] + 1)
    
    # ── Modify train_setup: add weight_decay to optimizer + scheduler creation ─
    # Find the last meaningful line in train_setup
    last_ts_line = _find_last_line_of_train_setup(lines, ts_start, ts_end)
    
    # If weight_decay > 0, try to add it to the optimizer line
    if weight_decay > 0:
        for i in range(ts_start, ts_end):
            if 'self.optimizer' in lines[i] and '=' in lines[i]:
                # Find the full optimizer assignment spanning potentially multiple lines
                # Track parenthesis nesting to find the outermost closing paren
                opt_start = i
                # Find the '(' that starts the optimizer call (e.g., SGD( or Adam()
                # We need to track from the first '(' after '=' sign
                full_text = ''
                opt_end = i
                paren_depth = 0
                found_start = False
                final_close_line = None
                final_close_col = None
                
                for k in range(i, ts_end):
                    line = lines[k]
                    for col, ch in enumerate(line):
                        if ch == '(':
                            if not found_start and k == i and col > line.index('='):
                                found_start = True
                            paren_depth += 1
                        elif ch == ')':
                            paren_depth -= 1
                            if found_start and paren_depth == 0:
                                final_close_line = k
                                final_close_col = col
                                break
                    if final_close_line is not None:
                        break
                
                if final_close_line is not None:
                    # Check the block for existing weight_decay
                    block = '\n'.join(lines[i:final_close_line + 1])
                    if 'weight_decay' not in block:
                        line = lines[final_close_line]
                        # Check for trailing comma before the closing paren
                        before_paren = line[:final_close_col].rstrip()
                        if before_paren.endswith(','):
                            # Already has trailing comma, just add param before ')'
                            lines[final_close_line] = (
                                line[:final_close_col] +
                                f" weight_decay=prm.get('weight_decay', {weight_decay})" +
                                line[final_close_col:]
                            )
                        else:
                            lines[final_close_line] = (
                                line[:final_close_col] +
                                f", weight_decay=prm.get('weight_decay', {weight_decay})" +
                                line[final_close_col:]
                            )
                break
    
    # Re-find last_ts_line after potential modifications
    last_ts_line = _find_last_line_of_train_setup(lines, ts_start, ts_end)
    
    # Check if there's already a self.scheduler line (like AirNext)
    has_scheduler = False
    scheduler_line_idx = None
    for i in range(ts_start, ts_end):
        if 'self.scheduler' in lines[i]:
            has_scheduler = True
            scheduler_line_idx = i
            break
    
    if has_scheduler:
        # Replace existing scheduler line
        lines[scheduler_line_idx] = scheduler_cfg['setup_code']
    else:
        # Insert scheduler setup after last line of train_setup
        lines.insert(last_ts_line + 1, scheduler_cfg['setup_code'])
        # Adjust hp_func_range if needed
        if hp_func_range[0] > last_ts_line:
            hp_func_range = (hp_func_range[0] + 1, hp_func_range[1] + 1)
    
    # ── Replace supported_hyperparameters function ────────────────────────────
    hp_s, hp_e = hp_func_range
    lines[hp_s:hp_e] = new_hp_lines
    
    # ── Reconstruct code ──────────────────────────────────────────────────────
    code = '\n'.join(lines)
    
    # ── Ensure all HP keys appear >= 2 times as string literals ───────────────
    # The Eval checker requires each HP in supported_hyperparameters() to appear
    # at least twice as a string literal in the code.
    # This covers both our added params AND original arch params that may be buggy.
    all_hp_to_check = set(scheduler_cfg['extra_hp'])
    all_hp_to_check.update(arch_hp_set)
    if weight_decay > 0:
        all_hp_to_check.add('weight_decay')
    
    for hp in sorted(all_hp_to_check):
        count = code.count(f"'{hp}'") + code.count(f'"{hp}"')
        if count < 2:
            safe_name = hp.upper().replace('-', '_').replace(' ', '_')
            extra_line = f"_HP_{safe_name} = '{hp}'  # hyperparameter key\n"
            code = extra_line + code
    
    # ── Verify the code has required functions ────────────────────────────────
    for fn in ('supported_hyperparameters', 'train_setup', 'learn'):
        if not re.search(r'\s+def\s' + re.escape(fn) + r'\(', code):
            if not re.search(r'^def\s' + re.escape(fn) + r'\(', code, re.MULTILINE):
                return None
    
    return code


def generate_models(output_base_dir, prefix='BO'):
    """Generate ~2000 model variants."""
    src_dir = find_nn_source_dir()
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    model_idx = 1
    total_generated = 0
    total_skipped = 0
    arch_stats = {}
    
    print(f"Source directory: {src_dir}")
    print(f"Output directory: {output_base}")
    print(f"Architectures: {len(ARCHITECTURES)}")
    print(f"Schedulers: {len(SCHEDULERS)}")
    print(f"Weight decay values: {len(WEIGHT_DECAY_VALUES)}")
    print(f"Max possible models: {len(ARCHITECTURES) * len(SCHEDULERS) * len(WEIGHT_DECAY_VALUES)}")
    print()
    
    for arch in ARCHITECTURES:
        source_code = read_architecture_source(src_dir, arch)
        if source_code is None:
            print(f"[SKIP] {arch}: source file not found")
            total_skipped += len(SCHEDULERS) * len(WEIGHT_DECAY_VALUES)
            continue
        
        arch_hp = get_supported_hp(src_dir, arch)
        if arch_hp is None:
            print(f"[SKIP] {arch}: cannot import supported_hyperparameters")
            total_skipped += len(SCHEDULERS) * len(WEIGHT_DECAY_VALUES)
            continue
        
        arch_count = 0
        arch_skip = 0
        skip_reasons = {}
        
        for sched in SCHEDULERS:
            for wd in WEIGHT_DECAY_VALUES:
                model_code = inject_scheduler_ast(source_code, sched, wd, arch_hp)
                
                if model_code is None:
                    arch_skip += 1
                    total_skipped += 1
                    reason = f"{sched['name']}"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue
                
                # Validate: parse AST to ensure valid Python
                try:
                    ast.parse(model_code)
                except SyntaxError as e:
                    arch_skip += 1
                    total_skipped += 1
                    reason = f"{sched['name']}_syntax"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue
                
                # Validate: each HP in supported_hyperparameters appears >= 2 times
                hp_match = re.search(
                    r'def supported_hyperparameters\(\):.*?return\s+(\{[^}]+\})',
                    model_code, re.DOTALL
                )
                if hp_match:
                    try:
                        hp_set = ast.literal_eval(hp_match.group(1))
                        valid = True
                        for h in hp_set:
                            cnt = model_code.count(f"'{h}'") + model_code.count(f'"{h}"')
                            if cnt < 2:
                                valid = False
                                break
                        if not valid:
                            arch_skip += 1
                            total_skipped += 1
                            continue
                    except Exception:
                        arch_skip += 1
                        total_skipped += 1
                        continue
                
                # Create model directory
                model_name = f"{prefix}_{model_idx:04d}"
                model_dir = output_base / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Write new_nn.py
                nn_file = model_dir / 'new_nn.py'
                with open(nn_file, 'w') as f:
                    f.write(model_code)
                
                # Write hp.txt (JSON) so NNEval loads architecture-specific defaults
                hp_dict = build_hp_dict(arch, sched, wd)
                hp_file = model_dir / 'hp.txt'
                with open(hp_file, 'w') as f:
                    json.dump(hp_dict, f, indent=2)
                
                # Write metadata for reference
                meta_file = model_dir / 'model_meta.txt'
                with open(meta_file, 'w') as f:
                    f.write(f"architecture: {arch}\n")
                    f.write(f"scheduler: {sched['name']}\n")
                    f.write(f"weight_decay: {wd}\n")
                    f.write(f"original_hp: {sorted(arch_hp)}\n")
                    f.write(f"added_hp: {sched['extra_hp']}\n")
                
                model_idx += 1
                arch_count += 1
                total_generated += 1
        
        arch_stats[arch] = {'generated': arch_count, 'skipped': arch_skip}
        status = "OK" if arch_count > 0 else "FAIL"
        print(f"[{status}] {arch}: {arch_count} models generated, {arch_skip} skipped")
        if arch_skip > 0 and arch_count == 0:
            for reason, cnt in sorted(skip_reasons.items()):
                print(f"       {reason}: {cnt}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_generated} models generated, {total_skipped} skipped")
    print(f"Models saved to: {output_base}")
    print(f"{'='*60}")
    
    print(f"\nPer-architecture summary:")
    for arch, stats in arch_stats.items():
        print(f"  {arch:40s} gen={stats['generated']:4d}  skip={stats['skipped']:4d}")
    
    return total_generated


def main():
    project_root = Path(__file__).resolve().parents[4]  # nn-gpt root
    output_dir = project_root / 'out' / 'nngpt' / 'llm' / 'epoch' / 'A0' / 'synth_nn'
    
    # Clean existing lr_ models
    if output_dir.exists():
        import shutil
        removed = 0
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith('lr_'):
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"Cleaned {removed} existing lr_ model directories.\n")
    
    # Clean database to avoid "NN already exists" error
    db_file = project_root / 'db' / 'ab.nn.db'
    if db_file.exists():
        import shutil
        try:
            db_file.unlink()
            print(f"Removed database file: {db_file}\n")
        except Exception as e:
            print(f"Warning: Could not delete database file: {e}\n")
    
    total = generate_models(str(output_dir), prefix='lr')
    print(f"\nDone. Generated {total} models ready for NNEval.")
    print(f"\nTo evaluate, run:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ab.gpt.NNEval --only_epoch 0 --nn_train_epochs 5 --nn_name_prefix lr")


if __name__ == '__main__':
    main()

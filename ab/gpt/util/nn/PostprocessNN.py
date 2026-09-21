"""
Post-processes generated new_nn.py files before evaluation.

Called by Tune._evaluate_epoch() after generation, before the NN evaluator runs,
so models that almost follow the LEMUR interface are repaired instead of lost.
(Previously lived in patches/postprocess_nn.py and was monkey-patched in; now
part of the source eval path.)

Fixes applied in order:
  1. Add missing "import torch.nn.functional as F" when F. is used
  2. Fix 3-variable in_shape unpack (c,h,w = in_shape) -> (_, c, h, w = in_shape)
  3. Rename the main nn.Module subclass to "Net" if it has any other name
  4. Rename a non-standard training-loop method to "learn"
  5. Strip keys from supported_hyperparameters() that are never accessed via prm[key]
"""
import ast
import re
from pathlib import Path


# ── individual fix functions ──────────────────────────────────────────────────

def _fix_f_import(code: str):
    if 'F.' not in code or 'import torch.nn.functional as F' in code:
        return code, None
    if 'import torch.nn as nn' in code:
        code = code.replace(
            'import torch.nn as nn',
            'import torch.nn as nn\nimport torch.nn.functional as F',
            1,
        )
    else:
        code = re.sub(r'(import torch\b)', r'\1\nimport torch.nn.functional as F', code, count=1)
    return code, 'added import torch.nn.functional as F'


def _fix_in_shape_unpack(code: str):
    # Matches: optional leading whitespace, optional parens, exactly 3 names, = in_shape
    pattern = r'(\s+)\(?\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)?\s*=\s*in_shape'
    m = re.search(pattern, code)
    if m is None:
        return code, None
    indent = m.group(1)
    c, h, w = m.group(2), m.group(3), m.group(4)
    old = m.group(0)
    new = f'{indent}_, {c}, {h}, {w} = in_shape'
    return code.replace(old, new, 1), f'fixed 3-var in_shape unpack -> (_, {c}, {h}, {w})'


def _find_main_class(tree) -> "str | None":
    """Return the name of the class with __init__(self, in_shape, out_shape, prm, device)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != 'Net':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    arg_names = [a.arg for a in item.args.args]
                    if 'in_shape' in arg_names and 'prm' in arg_names:
                        return node.name
    return None


def _fix_class_name(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    old = _find_main_class(tree)
    if old is None:
        return code, None
    code = re.sub(r'\b' + re.escape(old) + r'\b', 'Net', code)
    return code, f'renamed class {old} -> Net'


def _fix_learn_method(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    net_cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Net'), None)
    if net_cls is None:
        return code, None
    methods = {item.name for item in net_cls.body if isinstance(item, ast.FunctionDef)}
    if 'learn' in methods:
        return code, None
    skip = {'__init__', 'forward', 'train_setup'}
    for item in net_cls.body:
        if not isinstance(item, ast.FunctionDef) or item.name in skip:
            continue
        arg_names = [a.arg for a in item.args.args]
        if len(arg_names) < 2:
            continue
        data_arg = arg_names[1]
        for sub in ast.walk(item):
            if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Name) and sub.iter.id == data_arg:
                old = item.name
                code = re.sub(r'\bdef\s+' + re.escape(old) + r'\s*\(', 'def learn(', code, count=1)
                return code, f'renamed method {old} -> learn'
    return code, None


def _fix_hyperparams(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    declared = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'supported_hyperparameters':
            if node.body and isinstance(node.body[0], ast.Return):
                try:
                    val = ast.literal_eval(node.body[0].value)
                    if isinstance(val, (set, frozenset)):
                        declared = set(val)
                except Exception:
                    pass
            break

    if not declared:
        return code, None

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'prm':
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    used.add(node.slice.value)
        elif isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'prm'):
                if node.args and isinstance(node.args[0], ast.Constant):
                    used.add(node.args[0].value)

    unused = declared - used
    if not unused:
        return code, None

    new_keys = declared - unused
    keys_str = ', '.join(f"'{k}'" for k in sorted(new_keys))
    new_return = '{' + keys_str + '}'

    new_code = re.sub(
        r'(def\s+supported_hyperparameters\s*\(\s*\)\s*:[^\n]*\n\s*return\s*)\{[^}]*\}',
        r'\g<1>' + new_return,
        code,
    )
    if new_code == code:
        return code, None
    return new_code, f'removed unused hyperparams: {unused}'


# ── public interface ──────────────────────────────────────────────────────────

_FIXES = (
    _fix_f_import,
    _fix_in_shape_unpack,
    _fix_class_name,
    _fix_learn_method,
    _fix_hyperparams,
)


def postprocess_file(nn_path) -> list:
    """Apply all fixes to one new_nn.py. Returns descriptions of changes made."""
    path = Path(nn_path)
    try:
        code = path.read_text()
    except OSError:
        return []

    original = code
    changes = []
    for fix_fn in _FIXES:
        try:
            code, msg = fix_fn(code)
            if msg:
                changes.append(msg)
        except Exception as exc:
            changes.append(f'{fix_fn.__name__} error: {exc}')

    if code != original:
        path.write_text(code)
    return changes


def postprocess_directory(models_dir) -> None:
    """Postprocess every B*/new_nn.py under models_dir."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return
    for bdir in sorted(models_dir.glob('B*')):
        nn_path = bdir / 'new_nn.py'
        if not nn_path.exists():
            continue
        changes = postprocess_file(nn_path)
        label = '; '.join(changes) if changes else 'no fixes needed'
        print(f'[POSTPROCESS] {bdir.name}: {label}', flush=True)

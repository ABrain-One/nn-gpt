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
  6. Inject a LEMUR-correct train_setup(self, prm) stub when Net has none (NNGPT_STUB_FIXERS)
  7. Inject a LEMUR-correct learn(self, train_data) stub when Net has forward but no learn (NNGPT_STUB_FIXERS)
"""
import ast
import os
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


# ── method-injection helpers ──────────────────────────────────────────────────

# Stubs are written at column 0; _inject_method() re-indents them to match the
# class. Both follow the LEMUR interface exactly: train_setup(self, prm) sets
# self.optimizer + self.criteria (a tuple); learn(self, train_data) iterates
# `for inputs, labels in train_data`. They rely on __init__ having set
# self.device (mandated by the interface and done by all well-formed models).
_TRAIN_SETUP_STUB = (
    "def train_setup(self, prm):\n"
    "    self.to(self.device)\n"
    "    self.criteria = (nn.CrossEntropyLoss().to(self.device),)\n"
    "    self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))"
)

_LEARN_STUB = (
    "def learn(self, train_data):\n"
    "    self.train()\n"
    "    for inputs, labels in train_data:\n"
    "        inputs, labels = inputs.to(self.device), labels.to(self.device)\n"
    "        self.optimizer.zero_grad()\n"
    "        outputs = self(inputs)\n"
    "        loss = self.criteria[0](outputs, labels)\n"
    "        loss.backward()\n"
    "        self.optimizer.step()"
)


def _net_class_node(tree):
    return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Net'), None)


def _method_indent(net_cls) -> int:
    for item in net_cls.body:
        if isinstance(item, ast.FunctionDef):
            return item.col_offset
    return 4


def _inject_method(code: str, stub_col0: str, indent: int) -> str:
    """Insert stub_col0 (re-indented by `indent`) right before Net's last method."""
    tree = ast.parse(code)
    net_cls = _net_class_node(tree)
    if net_cls is None:
        return code
    methods = [item for item in net_cls.body if isinstance(item, ast.FunctionDef)]
    if methods:
        last = methods[-1]
        insert_lineno = min([d.lineno for d in last.decorator_list] + [last.lineno])
    elif net_cls.body:
        insert_lineno = net_cls.body[0].lineno
    else:
        return code
    lines = code.splitlines(keepends=True)
    insert_idx = insert_lineno - 1
    pad = ' ' * indent
    stub_lines = [(pad + ln if ln.strip() else '') for ln in stub_col0.split('\n')]
    block = '\n'.join(stub_lines) + '\n\n'
    new_lines = lines[:insert_idx] + [block] + lines[insert_idx:]
    return ''.join(new_lines)


def _fix_missing_train_setup(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    net_cls = _net_class_node(tree)
    if net_cls is None:
        return code, None
    methods = {item.name for item in net_cls.body if isinstance(item, ast.FunctionDef)}
    if 'train_setup' in methods:
        return code, None
    new_code = _inject_method(code, _TRAIN_SETUP_STUB, _method_indent(net_cls))
    if new_code == code:
        return code, None
    return new_code, 'injected train_setup stub'


def _fix_missing_learn(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    net_cls = _net_class_node(tree)
    if net_cls is None:
        return code, None
    methods = {item.name for item in net_cls.body if isinstance(item, ast.FunctionDef)}
    if 'learn' in methods or 'forward' not in methods:
        return code, None
    new_code = _inject_method(code, _LEARN_STUB, _method_indent(net_cls))
    if new_code == code:
        return code, None
    return new_code, 'injected learn stub'


# ── public interface ──────────────────────────────────────────────────────────

_FIXES = (
    _fix_f_import,
    _fix_in_shape_unpack,
    _fix_class_name,
    _fix_learn_method,
    _fix_hyperparams,
)

# Fixes 6 and 7 (stub injection); opt-in with NNGPT_STUB_FIXERS.
_STUB_FIXES = (
    _fix_missing_train_setup,
    _fix_missing_learn,
)


def _active_fixes():
    if os.environ.get("NNGPT_STUB_FIXERS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _FIXES + _STUB_FIXES
    return _FIXES


def postprocess_file(nn_path) -> list:
    """Apply all fixes to one new_nn.py. Returns descriptions of changes made."""
    path = Path(nn_path)
    try:
        code = path.read_text()
    except OSError:
        return []

    original = code
    changes = []
    for fix_fn in _active_fixes():
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

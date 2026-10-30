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
import json
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


def _entry_signature(node) -> bool:
    """Does this class take the trainer's __init__(self, in_shape, out_shape, prm, device)?"""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            arg_names = [a.arg for a in item.args.args]
            return 'in_shape' in arg_names and 'prm' in arg_names
    return False


def _instantiated_by_another_class(tree, name: str) -> bool:
    """Is `name` constructed inside some *other* class? Then it is a component."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name == name:
            continue
        for node in ast.walk(cls):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == name):
                return True
    return False


def _find_main_class(tree) -> "str | None":
    """The class to rename to Net, or None to leave the file alone.

    Two ways this used to go wrong on a mixture-of-experts file, which is most
    of what the generator writes. It took the first class in walk order whose
    __init__ matched, and the experts match it too -- they are built with the
    same (in_shape, out_shape, prm, device) signature as the wrapper that holds
    them. And it renamed even when the file already defined Net.

    Both at once produced silent infinite recursion: an expert became a second
    `class Net`, shadowing the real wrapper, whose own `Net(in_shape, ...)` call
    for that expert then resolved to the wrapper itself. Measured on one cycle:
    three candidates rejected with RecursionError at construct, every one of
    them a file this step had "repaired".

    So: never rename when a Net already exists, and among the candidates prefer
    one no other class instantiates -- the wrapper, not its experts.
    """
    if any(isinstance(n, ast.ClassDef) and n.name == 'Net' for n in ast.walk(tree)):
        return None
    candidates = [n.name for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and _entry_signature(n)]
    if not candidates:
        return None
    standalone = [n for n in candidates
                  if not _instantiated_by_another_class(tree, n)]
    # Last, not first: the wrapper is written after the parts it composes.
    return (standalone or candidates)[-1]


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


def _prm_usage(tree):
    """``(used, resolvable)`` -- the prm keys this file really takes, if knowable.

    ``used`` is every key reached by a literal ``prm['k']`` or ``prm.get('k')``.
    ``setdefault`` is deliberately not a use: it writes a default into prm, and
    a key only ever setdefault-ed is never taken from it.

    ``resolvable`` is False when the file reaches prm in a way no static read
    can follow -- splatting it, iterating it, a computed key, handing it to a
    helper. A generated file may do any of that, and then the read set is
    unknown rather than empty, so the caller must leave the declaration alone
    instead of rewriting it to something smaller than the truth.
    """
    used = set()
    accounted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'prm':
                accounted.add(id(node.value))
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    used.add(node.slice.value)
                else:
                    return used, False          # prm[expr]: key not knowable
        elif isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == 'prm'):
                accounted.add(id(func.value))
                if func.attr in ('get', 'setdefault'):
                    if not (node.args and isinstance(node.args[0], ast.Constant)):
                        return used, False      # prm.get(expr)
                    if func.attr == 'get':
                        used.add(node.args[0].value)
                else:
                    return used, False          # keys(), items(), pop(), ...
    # Any other mention of the name -- **prm, `for k in prm`, f(prm) -- means
    # the file may read keys this pass cannot see.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == 'prm' and id(node) not in accounted:
            if isinstance(getattr(node, 'ctx', None), ast.Store):
                continue                        # the parameter itself
            return used, False
    return used, True


def _fix_hyperparams(code: str):
    """Make supported_hyperparameters() name exactly the prm keys the file takes.

    The contract runs both ways: the trainer builds prm from these names, so a
    name that is read but not declared is a KeyError, and a name declared but
    never taken from prm is a claim the file does not keep. Both are repaired
    here rather than rejected downstream -- the file is otherwise sound, and a
    rewritten set costs nothing while a discarded candidate costs a generation.

    Nothing is rewritten when the read set cannot be resolved; see _prm_usage.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    declared = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == 'supported_hyperparameters':
            if node.body and isinstance(node.body[0], ast.Return):
                try:
                    val = ast.literal_eval(node.body[0].value)
                    if isinstance(val, (set, frozenset, list, tuple)):
                        declared = set(str(v) for v in val)
                except Exception:
                    declared = None
            break
    if declared is None:
        return code, None

    used, resolvable = _prm_usage(tree)
    if not resolvable:
        return code, None
    if used == declared:
        return code, None

    keys_str = ', '.join(f"'{k}'" for k in sorted(used))
    new_return = '{' + keys_str + '}' if used else 'set()'
    new_code = re.sub(
        r'(def\s+supported_hyperparameters\s*\(\s*\)\s*:[^\n]*\n\s*return\s*)(\{[^}]*\}|set\(\))',
        lambda m: m.group(1) + new_return,
        code,
        count=1,
    )
    if new_code == code:
        return code, None
    added, dropped = sorted(used - declared), sorted(declared - used)
    bits = []
    if added:
        bits.append('declared ' + ', '.join(added))
    if dropped:
        bits.append('dropped unused ' + ', '.join(dropped))
    return new_code, 'hyperparams: ' + '; '.join(bits)


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


def _declared_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == 'supported_hyperparameters':
            if node.body and isinstance(node.body[0], ast.Return):
                try:
                    val = ast.literal_eval(node.body[0].value)
                    if isinstance(val, (set, frozenset, list, tuple)):
                        return set(str(v) for v in val)
                except Exception:
                    return None
            return None
    return None


def _own_defaults(tree):
    """The value the file itself offers for a prm name: prm.get(k, v), setdefault(k, v)."""
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('get', 'setdefault')
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'prm'
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            try:
                out.setdefault(str(node.args[0].value), ast.literal_eval(node.args[1]))
            except Exception:
                pass
    return out


def sync_hp_with_declaration(bdir: Path, code: str):
    """Give hp.txt a value for every name the file declares.

    The trainer builds prm from the record plus this file, so a declared name
    with nothing behind it is a prm key that will not exist when the file reads
    it -- and that is what nn-dataset's own probe_prm refuses. Declaring a name
    is therefore only half a repair; the value has to arrive too.

    Where the file already states its own default -- prm.get(name, 0.5) -- that
    is the value it was written for, so it is what goes in. A name read with no
    default anywhere cannot be filled from here and is reported instead: only
    the model knows what it meant, and the prompt asks it to propose one.
    """
    hp_path = bdir / 'hp.txt'
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    declared = _declared_names(tree)
    if not declared:
        return None
    try:
        hp = json.loads(hp_path.read_text(encoding='utf-8')) if hp_path.is_file() else {}
        if not isinstance(hp, dict):
            hp = {}
    except Exception:
        hp = {}
    missing = sorted(n for n in declared if n not in hp)
    if not missing:
        return None
    defaults = _own_defaults(tree)
    filled, unfilled = {}, []
    for name in missing:
        if name in defaults:
            filled[name] = defaults[name]
        else:
            unfilled.append(name)
    notes = []
    if filled:
        hp.update(filled)
        try:
            hp_path.write_text(json.dumps(hp), encoding='utf-8')
            notes.append('hp filled ' + ', '.join(f'{k}={filled[k]!r}' for k in sorted(filled)))
        except Exception as exc:
            notes.append(f'hp not written: {exc}')
    if unfilled:
        notes.append('declared with no value anywhere: ' + ', '.join(unfilled))
    return '; '.join(notes) if notes else None


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
        try:
            note = sync_hp_with_declaration(bdir, nn_path.read_text(encoding='utf-8',
                                                                    errors='replace'))
        except Exception as exc:
            note = f'hp sync error: {exc}'
        if note:
            changes.append(note)
        label = '; '.join(changes) if changes else 'no fixes needed'
        print(f'[POSTPROCESS] {bdir.name}: {label}', flush=True)

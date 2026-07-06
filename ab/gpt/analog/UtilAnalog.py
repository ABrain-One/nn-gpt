import ast
import functools
import importlib
import inspect
import json
import os
import os.path
import re
import shutil
from pathlib import Path
import torch

from ab.gpt.util.Const import conf_dir, new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir
from ab.gpt.util.EditUtil import normalize_edit_text, parse_edit_text

from ..util.Code import *


DATASET_META_PATH = conf_dir / 'dataset_meta.json'
_EXPLICIT_ECHO_SIZES = {28, 32, 64, 128, 224, 256, 299, 512}


@functools.lru_cache(maxsize=1)
def load_dataset_meta():
    with open(DATASET_META_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def get_dataset_meta(dataset: str):
    if not dataset:
        return None
    return load_dataset_meta().get(str(dataset).lower())


def _default_transform_name(img_size: int):
    if img_size in _EXPLICIT_ECHO_SIZES:
        return f'echo_{img_size}'
    return 'echo'


def build_default_transform_code(img_size: int):
    if img_size and img_size > 0:
        return (
            "import torchvision.transforms as transforms\n"
            "def transform(_):\n"
            f"    return transforms.Compose([transforms.Resize(({img_size}, {img_size})), transforms.ToTensor()])"
        )
    return (
        "import torchvision.transforms as transforms\n"
        "def transform(_):\n"
        "    return transforms.Compose([transforms.ToTensor()])"
    )


def _build_dataset_smoke_profiles():
    profiles = {}
    for dataset, meta in load_dataset_meta().items():
        img_size = int(meta.get('img_size', 0) or 0)
        num_channels = int(meta.get('num_channels', 0) or 0)
        num_classes = int(meta.get('num_classes', 0) or 0)
        if img_size <= 0 or num_channels <= 0 or num_classes <= 0:
            continue
        profiles[str(dataset).lower()] = {
            'in_shape': (1, num_channels, img_size, img_size),
            'out_shape': (num_classes,),
            'default_transform': _default_transform_name(img_size),
            'transform_code': build_default_transform_code(img_size),
            'small_rgb32': bool(num_channels == 3 and img_size == 32),
            'img_size': img_size,
            'num_channels': num_channels,
            'num_classes': num_classes,
            'description': meta.get('description', ''),
        }
    return profiles


DATASET_SMOKE_PROFILES = _build_dataset_smoke_profiles()


def nn_accepted(nn_dir):
    accepted = True
    return accepted


def verify_nn_code(nn_dir, nn_file):
    verified = True
    error_message = ''
    if not verified:
        with open(nn_dir / f"error_code_verification.txt", "w+") as error_file:
            error_file.write(f"Code verification failed: {error_message}")
    return verified


def exists(f):
    return f and os.path.exists(f)


def get_dataset_smoke_profile(dataset: str):
    if not dataset:
        return None
    return DATASET_SMOKE_PROFILES.get(str(dataset).lower())


def get_dataset_prompt_defaults(dataset: str):
    profile = get_dataset_smoke_profile(dataset)
    if not profile:
        return None
    return {
        'dataset': str(dataset).lower(),
        'display_name': str(dataset).upper() if str(dataset).lower() == 'svhn' else str(dataset),
        'img_size': profile['img_size'],
        'num_channels': profile['num_channels'],
        'num_classes': profile['num_classes'],
        'default_transform': profile['default_transform'],
        'transform_code': profile['transform_code'],
        'hp_json': json.dumps(
            {"batch": 64, "transform": profile['default_transform'], "lr": 0.01, "momentum": 0.9},
            separators=(',', ':'),
        ),
        'description': profile.get('description', ''),
    }


def is_small_rgb32_dataset(dataset: str):
    profile = get_dataset_smoke_profile(dataset)
    return bool(profile and profile.get('small_rgb32'))


def create_symlink(src, dst):
    """
    Create a symbolic link from src to dst.
    If dst already exists (as file or link), do nothing.
    """
    dst = Path(dst)
    src = Path(src)
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError as e:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def create_file(directory, filename, content):
    """
    Create a file with given content in the specified directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def read_py_file_as_string(file_path):
    """
    read_py_file_as_string。

    param:
        file_path (str): path of the file to read.

    Return:
        str: Content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"error when reading file: {e}")
        return None


def extract_str(s: str, start: str, end: str):
    try:
        start_idx = s.find(start)
        if start_idx >= 0:
            start_idx += len(start)
            end_idx = s.find(end, start_idx)
            if end_idx >= 0:
                return s[start_idx:end_idx].strip()
    except:
        pass
    return None


def extract_open_tag_content(s: str, tag: str, stop_tags=()):
    if not s or f'<{tag}>' not in s:
        return None
    content = s.rsplit(f'<{tag}>', 1)[-1]
    stop_positions = []
    for stop_tag in stop_tags:
        pos = content.find(stop_tag)
        if pos >= 0:
            stop_positions.append(pos)
    if stop_positions:
        content = content[:min(stop_positions)]
    return content.strip()


def strip_generation_prefix(text: str):
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*(?:<\|im_start\|>\s*)?assistant\s*>?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*assistant\s*:?\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def normalize_hyperparam_text(text: str):
    if text is None:
        return None
    cleaned = strip_generation_prefix(text)
    if '<hp>' in cleaned and '</hp>' in cleaned:
        nested = extract_str(cleaned, '<hp>', '</hp>')
        if nested:
            cleaned = nested
    cleaned = cleaned.replace('```json', '').replace('```python', '').replace('```', '').strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned.strip()


def parse_hyperparam_text(text: str):
    cleaned = normalize_hyperparam_text(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            parsed = ast.literal_eval(cleaned)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_by_pattern(name, res, options) -> str:
    res = improve_code(next(filter(None, map(lambda l: extract_str(res, *l), options)), None))
    if res:
        print(f'[EXTRACT] ✓ Found {name}: {len(res)} chars')
    else:
        print(f'[EXTRACT] ✗ No {name} found')
    return res


def extract_all_str(s: str, start: str, end: str):
    matches = []
    try:
        search_idx = 0
        while True:
            start_idx = s.find(start, search_idx)
            if start_idx < 0:
                break
            start_idx += len(start)
            end_idx = s.find(end, start_idx)
            if end_idx < 0:
                break
            matches.append(s[start_idx:end_idx].strip())
            search_idx = end_idx + len(end)
    except:
        pass
    return matches


def is_code_like_nn_block(code: str) -> bool:
    if not code:
        return False
    patterns = (
        'class Net',
        'def forward',
        'import torch',
        'torch.nn',
        'nn.Module',
        'def train_setup',
        'def learn',
    )
    hits = sum(1 for pattern in patterns if pattern in code)
    return hits >= 2


def _extract_broken_nn_block(txt: str):
    if not txt:
        return None

    candidates = []
    if '</nn>' in txt:
        candidates.append(txt.split('</nn>', 1)[0])
    if '<nn>' in txt:
        candidates.append(txt.rsplit('<nn>', 1)[-1])

    for candidate in candidates:
        cleaned = strip_generation_prefix(candidate)
        cleaned = re.sub(r'(?is)^.*?</hp>', '', cleaned)
        cleaned = re.sub(r'(?is)^.*?</tr>', '', cleaned)
        cleaned = re.sub(r'(?is)^\s*<hp>\s*', '', cleaned, count=1)
        cleaned = re.sub(r'(?is)^\s*<tr>\s*', '', cleaned, count=1)
        cleaned = cleaned.strip()
        for anchor in (r'(?m)^import\s+torch', r'(?m)^from\s+torch', r'(?m)^def\s+supported_hyperparameters', r'(?m)^class\s+\w+'):
            match = re.search(anchor, cleaned)
            if match:
                cleaned = cleaned[match.start():].strip()
                break
        if is_code_like_nn_block(cleaned):
            print(f'[EXTRACT] ✓ Recovered malformed NN code: {len(cleaned)} chars')
            return improve_code(cleaned)
    return None


def _extract_open_nn_block(txt: str):
    if not txt or '<nn>' not in txt:
        return None
    cleaned = strip_generation_prefix(txt.rsplit('<nn>', 1)[-1]).strip()
    if is_code_like_nn_block(cleaned):
        print(f'[EXTRACT] ✓ Recovered open <nn> block: {len(cleaned)} chars')
        return improve_code(cleaned)
    return None


def extract_code(txt):
    nn_blocks = list(map(improve_code, extract_all_str(txt, '<nn>', '</nn>')))
    if nn_blocks:
        for code in reversed(nn_blocks):
            if is_code_like_nn_block(code):
                print(f'[EXTRACT] ✓ Found NN code: {len(code)} chars (last code-like <nn> block from {len(nn_blocks)} matches)')
                return code
        code = nn_blocks[0]
        print(f'[EXTRACT] ✓ Found NN code: {len(code)} chars (fallback first <nn> block from {len(nn_blocks)} matches)')
        return code
    fenced = extract_by_pattern('NN code', txt, (('```python', '```'), ('```', '```')))
    if fenced:
        return fenced
    open_nn = _extract_open_nn_block(txt)
    if open_nn:
        return open_nn
    return _extract_broken_nn_block(txt)


def extract_hyperparam(txt):
    cleaned = txt.replace('< hp >', '<hp>').replace('<.hp>', '<hp>').replace('</ hp >', '</hp>')

    candidates = []
    candidates.extend(extract_all_str(cleaned, '<hp>', '</hp>'))
    candidates.extend(extract_all_str(cleaned, '```json', '```'))
    candidates.extend(extract_all_str(cleaned, '```python', '```'))

    open_hp = extract_open_tag_content(cleaned, 'hp', stop_tags=('</tr>', '<tr>', '<delta>', '</delta>', '<nn>', '</nn>'))
    if open_hp:
        candidates.append(open_hp)

    for candidate in candidates:
        normalized = normalize_hyperparam_text(candidate)
        if parse_hyperparam_text(normalized) is not None:
            print(f'[EXTRACT] ✓ Found hyper-parameters: {len(normalized)} chars')
            return normalized

    print(f'[EXTRACT] ✗ No hyper-parameters found')
    return None


def extract_transform(txt):
    cleaned = txt.replace('< tr >', '<tr>').replace('<.tr>', '<tr>').replace('</ tr >', '</tr>')
    candidates = extract_all_str(cleaned, '<tr>', '</tr>')
    open_tr = extract_open_tag_content(cleaned, 'tr', stop_tags=('<delta>', '</delta>', '<nn>', '</nn>'))
    if open_tr:
        candidates.append(open_tr)

    for candidate in candidates:
        block = improve_code(strip_generation_prefix(candidate))
        if block and ('def transform' in block or 'transforms.' in block):
            print(f'[EXTRACT] ✓ Found transform code: {len(block)} chars')
            return block

    print(f'[EXTRACT] ✗ No transform code found')
    return None


def extract_edit(txt):
    cleaned = txt.replace('< edit >', '<edit>').replace('<.edit>', '<edit>').replace('</ edit >', '</edit>')

    candidates = []
    candidates.extend(extract_all_str(cleaned, '<edit>', '</edit>'))
    candidates.extend(extract_all_str(cleaned, '```json', '```'))
    candidates.extend(extract_all_str(cleaned, '```python', '```'))

    open_edit = extract_open_tag_content(cleaned, 'edit', stop_tags=('<delta>', '</delta>', '<nn>', '</nn>'))
    if open_edit:
        candidates.append(open_edit)

    for candidate in candidates:
        parsed = parse_edit_text(candidate)
        if parsed is not None:
            normalized = normalize_edit_text(candidate)
            print(f'[EXTRACT] ✓ Found structured edit: {len(normalized)} chars')
            return normalized

    print(f'[EXTRACT] ✗ No structured edit found')
    return None


def extract_all_to_train(txt):
    return extract_code(txt), extract_hyperparam(txt), extract_transform(txt)


def _ensure_supported_hyperparameters(code: str):
    def extract_prm_key(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):
            return node.s
        return None

    try:
        tree = ast.parse(code)
    except Exception:
        tree = None

    used_keys = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == 'prm':
                    prm_key = extract_prm_key(node.slice)
                    if prm_key:
                        used_keys.add(prm_key)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'prm':
                        if node.args:
                            prm_key = extract_prm_key(node.args[0])
                            if prm_key:
                                used_keys.add(prm_key)

    preferred_order = ['lr', 'momentum', 'dropout', 'batch', 'transform', 'epoch']
    ordered_keys = [key for key in preferred_order if key in used_keys]
    ordered_keys.extend(sorted(used_keys - set(preferred_order)))
    if not ordered_keys:
        ordered_keys = ['lr', 'momentum']

    replacement = "def supported_hyperparameters():\n    return {" + ', '.join(f"'{key}'" for key in ordered_keys) + "}"

    if tree is None:
        if re.search(r"def\s+supported_hyperparameters\s*\(", code):
            return re.sub(
                r"def\s+supported_hyperparameters\s*\([\s\S]*?\):[\s\S]*?(?=^(?:def|class)\s|\Z)",
                replacement + "\n\n",
                code,
                flags=re.MULTILINE,
            )
        return code.rstrip() + "\n\n" + replacement + "\n"

    supported_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'supported_hyperparameters'
    ]
    lines = code.splitlines()

    if supported_nodes:
        replacement_lines = replacement.splitlines()
        keep_last = supported_nodes[-1]
        for node in reversed(supported_nodes):
            start = node.lineno - 1
            end = node.end_lineno
            if node is keep_last:
                lines[start:end] = replacement_lines
            else:
                del lines[start:end]
        return '\n'.join(lines).rstrip() + '\n'

    return code.rstrip() + "\n\n" + replacement + "\n"


def _ensure_torch_imports(code: str):
    fixed = code
    has_torch_import = bool(re.search(r'(?m)^\s*import\s+torch(?:\s+as\s+\w+)?\s*$', fixed))
    if not has_torch_import:
        fixed = 'import torch\n' + fixed
    has_nn_import = bool(re.search(r'(?m)^\s*import\s+torch\.nn\s+as\s+nn\s*$', fixed))
    has_from_nn_import = bool(re.search(r'(?m)^\s*from\s+torch\s+import\s+nn\b', fixed))
    if not has_nn_import and not has_from_nn_import:
        fixed = 'import torch.nn as nn\n' + fixed
    return fixed


def _fix_make_layer_shadowing(code: str):
    lines = code.splitlines()
    in_make_layer = False
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def _make_layer('):
            in_make_layer = True
        elif in_make_layer and stripped.startswith('def '):
            in_make_layer = False
        elif in_make_layer and stripped.startswith('class '):
            in_make_layer = False

        if in_make_layer:
            if stripped == 'layers = []':
                line = line.replace('layers = []', 'blocks = []')
            elif 'layers.append(' in line:
                line = line.replace('layers.append(', 'blocks.append(')
            elif 'return nn.Sequential(*layers)' in line:
                line = line.replace('return nn.Sequential(*layers)', 'return nn.Sequential(*blocks)')
        fixed_lines.append(line)
    return '\n'.join(fixed_lines)


def _ensure_net_init_signature(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    init_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Net':
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == '__init__':
                    init_node = sub
                    break
            break

    if init_node is None:
        return code

    arg_names = [arg.arg for arg in init_node.args.args]
    lines = code.splitlines()
    indent = ' ' * init_node.col_offset
    body_indent = indent + '    '
    init_segment = '\n'.join(lines[init_node.lineno - 1:init_node.end_lineno])
    init_body_segment = '\n'.join(lines[init_node.lineno:init_node.end_lineno])

    if arg_names[:5] != ['self', 'in_shape', 'out_shape', 'prm', 'device']:
        lines[init_node.lineno - 1] = f"{indent}def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"

    insert_after = init_node.lineno
    if init_node.body:
        first_stmt = init_node.body[0]
        first_line = lines[first_stmt.lineno - 1].strip()
        if first_line.startswith('super('):
            insert_after = getattr(first_stmt, 'end_lineno', first_stmt.lineno)

    bootstrap = []
    if not re.search(r'\bself\.device\s*=', init_body_segment):
        bootstrap.append(f'{body_indent}self.device = device')
    if re.search(r'\bc_in\b', init_body_segment) and not re.search(r'\bc_in\s*=', init_body_segment):
        bootstrap.append(f'{body_indent}c_in = in_shape[1]')
    if re.search(r'\bn_cls\b', init_body_segment) and not re.search(r'\bn_cls\s*=', init_body_segment):
        bootstrap.append(f'{body_indent}n_cls = out_shape[0]')
    if re.search(r'\bnum_classes\b', init_body_segment) and not re.search(r'\bnum_classes\s*=', init_body_segment):
        bootstrap.append(f'{body_indent}num_classes = out_shape[0]')

    if bootstrap:
        lines[insert_after:insert_after] = bootstrap
    return '\n'.join(lines)


def _ensure_classification_net_methods(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    net_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Net':
            net_node = node
            break

    if net_node is None:
        return code

    train_setup_nodes = [n for n in net_node.body if isinstance(n, ast.FunctionDef) and n.name == 'train_setup']
    learn_nodes = [n for n in net_node.body if isinstance(n, ast.FunctionDef) and n.name == 'learn']
    train_setup_node = train_setup_nodes[-1] if train_setup_nodes else None
    learn_node = learn_nodes[-1] if learn_nodes else None

    train_setup_args = [arg.arg for arg in train_setup_node.args.args] if train_setup_node else []
    learn_args = [arg.arg for arg in learn_node.args.args] if learn_node else []
    has_lr_use = bool(re.search(r"prm\s*(?:\[\s*['\"]lr['\"]\s*\]|\.get\(\s*['\"]lr['\"]\s*[,)]?)", code))
    has_momentum_use = bool(re.search(r"prm\s*(?:\[\s*['\"]momentum['\"]\s*\]|\.get\(\s*['\"]momentum['\"]\s*[,)]?)", code))
    has_duplicate_train_setup = len(train_setup_nodes) != 1
    has_duplicate_learn = len(learn_nodes) > 1
    has_legacy_tuple_criteria = 'self.criteria = (' in code or 'self.criteria[' in code

    needs_train_setup = (
        train_setup_node is None
        or has_duplicate_train_setup
        or train_setup_args[:2] != ['self', 'prm']
        or not has_lr_use
        or not has_momentum_use
        or has_legacy_tuple_criteria
    )
    needs_learn = (
        learn_node is None
        or learn_args[:2] != ['self', 'train_data']
        or has_duplicate_learn
    )
    if needs_train_setup:
        needs_learn = True

    if not needs_train_setup and not needs_learn:
        return code

    lines = code.splitlines()
    indent = ' ' * (net_node.col_offset + 4)
    additions = []
    if needs_train_setup:
        additions.extend([
            f'{indent}def train_setup(self, prm):',
            f'{indent}    self.to(self.device)',
            f'{indent}    self.criteria = nn.CrossEntropyLoss().to(self.device)',
            f"{indent}    self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])",
            '',
        ])
    if needs_learn:
        additions.extend([
            f'{indent}def learn(self, train_data):',
            f'{indent}    self.train()',
            f'{indent}    for inputs, labels in train_data:',
            f'{indent}        inputs, labels = inputs.to(self.device), labels.to(self.device)',
            f'{indent}        self.optimizer.zero_grad()',
            f'{indent}        outputs = self(inputs)',
            f'{indent}        loss = self.criteria(outputs, labels)',
            f'{indent}        loss.backward()',
            f'{indent}        nn.utils.clip_grad_norm_(self.parameters(), 3)',
            f'{indent}        self.optimizer.step()',
            '',
        ])

    if additions:
        lines[net_node.end_lineno:net_node.end_lineno] = additions
    return '\n'.join(lines)


def _reject_used_direct_module_without_forward(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    invalid_helpers = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name == 'Net':
            continue

        base_names = []
        for base in node.bases:
            try:
                base_names.append(ast.unparse(base))
            except Exception:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)

        is_direct_module = any(name in ('nn.Module', 'torch.nn.Module', 'Module') for name in base_names)
        if not is_direct_module:
            continue

        method_names = {
            sub.name for sub in node.body if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if 'forward' in method_names:
            continue

        usage_count = len(re.findall(rf'\b{re.escape(node.name)}\s*\(', code))
        if usage_count > 1:
            invalid_helpers.append(node.name)

    if invalid_helpers:
        print(f"[WARNING] Rejecting generated code: helper nn.Module class(es) missing forward(): {', '.join(invalid_helpers)}")
        return None
    return code


def normalize_generated_nn_code(code: str):
    if not code:
        return code
    fixed = code.strip()

    try:
        ast.parse(fixed)
    except Exception:
        cut_points = [
            fixed.find('\n    @staticmethod\n    def train_setup('),
            fixed.find('\n    def train_setup('),
            fixed.find('\n    @staticmethod\n    def learn('),
            fixed.find('\n    def learn('),
        ]
        cut_points = [point for point in cut_points if point >= 0]
        if cut_points:
            fixed = fixed[:min(cut_points)].rstrip()

    fixed = re.sub(r'(?m)^\s*(?:model|net)\s*=\s*Net\(.*\)\s*$', '', fixed)
    fixed = re.sub(r'in_shape\[\s*0\s*\]', 'in_shape[1]', fixed)
    fixed = re.sub(r'out_shape\[\s*1\s*\]', 'out_shape[0]', fixed)
    fixed = fixed.replace('self.to(device)', 'self.to(self.device)')
    fixed = fixed.replace('nnAdaptiveAvgPool2d', 'nn.AdaptiveAvgPool2d')
    fixed = _ensure_torch_imports(fixed)
    fixed = _ensure_supported_hyperparameters(fixed)
    fixed = _fix_make_layer_shadowing(fixed)
    fixed = _ensure_net_init_signature(fixed)
    fixed = _ensure_classification_net_methods(fixed)
    fixed = _reject_used_direct_module_without_forward(fixed)
    if not fixed:
        return None
    fixed = re.sub(r'\n{3,}', '\n\n', fixed).strip() + '\n'
    return fixed


def validate_generated_nn_smoke(code: str, in_shape=(1, 3, 32, 32), out_shape=(10,)):
    if not code:
        return False, 'No code to validate.'

    namespace = {}
    try:
        exec(compile(code, '<generated_nn>', 'exec'), namespace, namespace)
    except Exception as e:
        return False, f'Code import failed: {e}'

    net_cls = namespace.get('Net')
    if net_cls is None:
        return False, 'Net class is missing.'

    device = torch.device('cpu')
    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.2}
    try:
        model = net_cls(in_shape, out_shape, prm, device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            batch = 2
            x = torch.randn(batch, in_shape[1], in_shape[2], in_shape[3], device=device)
            y = model(x)
            if isinstance(y, (tuple, list)) and y:
                y = y[0]
            if not isinstance(y, torch.Tensor):
                return False, f'Forward returned non-tensor output: {type(y)!r}'
            if y.ndim < 2:
                return False, f'Forward returned rank-{y.ndim} tensor.'
            if y.shape[0] != batch:
                return False, f'Forward changed batch dimension unexpectedly: {tuple(y.shape)}'
            if y.shape[-1] <= 0:
                return False, f'Forward returned invalid class dimension: {tuple(y.shape)}'
    except Exception as e:
        return False, str(e)

    return True, None


def is_cifar_spatial_collapse_error(error: str):
    if not error:
        return False
    text = str(error).lower()
    return (
        'output size is too small' in text
        or ('calculated output size' in text and '0x0' in text)
    )


def repair_cifar_spatial_collapse(code: str):
    if not code:
        return None

    repaired = code
    changed = False

    # CIFAR-10 inputs are 32x32, so ImageNet-style stride-4 stems are too aggressive.
    repaired, count = re.subn(
        r'(nn\.Conv2d\([^)\n]*?\bstride\s*=\s*)4(\b)',
        r'\g<1>1\2',
        repaired,
    )
    changed = changed or count > 0

    # Relax repeated 3x3/stride-2 pooling, which is the main source of 0x0 collapses.
    # ceil_mode keeps a 1x1 feature map from becoming 0x0 at the last CIFAR pooling stage.
    repaired, count = re.subn(
        r'nn\.MaxPool2d\(\s*kernel_size\s*=\s*3\s*,\s*stride\s*=\s*2(?:\s*,\s*padding\s*=\s*0)?\s*\)',
        'nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)',
        repaired,
    )
    changed = changed or count > 0
    repaired, count = re.subn(
        r'nn\.MaxPool2d\(\s*3\s*,\s*2\s*\)',
        'nn.MaxPool2d(2, 2, ceil_mode=True)',
        repaired,
    )
    changed = changed or count > 0

    if not changed:
        return None

    try:
        ast.parse(repaired)
    except Exception:
        return None

    return repaired


def is_cifar_unsafe_seed(code: str, model_name: str = None):
    text = code or ''
    name = str(model_name or '')

    if 'alexnet' in name.lower():
        return True

    has_stride4_stem = bool(re.search(r'nn\.Conv2d\([^)\n]*?\bstride\s*=\s*4\b', text))
    pool_3x2_count = len(re.findall(
        r'nn\.MaxPool2d\(\s*kernel_size\s*=\s*3\s*,\s*stride\s*=\s*2(?:\s*,\s*padding\s*=\s*0)?\s*\)',
        text,
    )) + len(re.findall(r'nn\.MaxPool2d\(\s*3\s*,\s*2\s*\)', text))

    return has_stride4_stem and pool_3x2_count >= 2


def extract_delta(txt):
    """
    Extract delta (unified diff) from text.
    Looks for:
    1. <delta>...</delta> XML tags
    2. Full unified diff blocks (---, +++, @@) - picks the most complete one
    3. Line-by-line diff extraction across multiple blocks
    4. Last resort - any diff-like content

    Args:
        txt: Text containing delta

    Returns:
        Delta string or None if not found
    """
    if not txt:
        return None

    # Strategy 1: Try XML tags first (with common typo fixes)
    cleaned = txt.replace('< delta >', '<delta>').replace('<.delta>', '<delta>')
    cleaned = cleaned.replace('</ delta >', '</delta>').replace('< /delta>', '</delta>')
    delta = extract_str(cleaned, '<delta>', '</delta>')
    if delta and ('---' in delta or '@@' in delta or '+' in delta):
        return delta.strip()

    # Strategy 1b: recover open <delta> block when the model never emits </delta>
    open_delta = extract_open_tag_content(cleaned, 'delta')
    if open_delta:
        diff_start = re.search(r'(?m)^(---\s+\S+|@@\s)', open_delta)
        if diff_start:
            open_delta = open_delta[diff_start.start():]
        if '---' in open_delta and ('+++' in open_delta or '@@' in open_delta):
            return open_delta.strip()

    # Strategy 2: Find ALL raw unified diff blocks and pick the best one
    diff_pattern = re.compile(
        r'(---\s*\S+.*?\n\+\+\+\s*\S+.*?\n(?:@@[^\n]+@@\n(?:[+\- ].*?\n)*)+)',
        re.MULTILINE | re.DOTALL
    )
    all_matches = diff_pattern.findall(txt)
    if all_matches:
        best_diff = max(all_matches, key=lambda d: (d.count('@@'), len(d)))
        return best_diff.strip()

    # Strategy 3: Line-by-line extraction - find ALL diff blocks, pick best
    lines = txt.splitlines()
    all_diff_blocks = []
    current_block = []
    in_diff = False
    found_header = False

    for i, line in enumerate(lines):
        if line.startswith('---') and not line.startswith('----'):
            if current_block and found_header and len(current_block) >= 3:
                all_diff_blocks.append('\n'.join(current_block))
            in_diff = True
            found_header = True
            current_block = [line]
        elif in_diff and line.startswith('+++'):
            current_block.append(line)
        elif in_diff and line.startswith('@@'):
            current_block.append(line)
        elif in_diff:
            if line.startswith('-') or line.startswith('+') or line.startswith(' '):
                current_block.append(line)
            elif line.strip() == '':
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.startswith(('-', '+', ' ', '@@')):
                        current_block.append(line)
                    else:
                        if current_block and found_header and len(current_block) >= 3:
                            all_diff_blocks.append('\n'.join(current_block))
                        in_diff = False
                        found_header = False
                        current_block = []
            elif not line.startswith(('diff', 'index', 'new', 'old', 'Binary')):
                if current_block and found_header and len(current_block) >= 3:
                    all_diff_blocks.append('\n'.join(current_block))
                in_diff = False
                found_header = False
                current_block = []

    if current_block and found_header and len(current_block) >= 3:
        all_diff_blocks.append('\n'.join(current_block))

    if all_diff_blocks:
        return max(all_diff_blocks, key=lambda d: (d.count('@@'), len(d)))

    # Strategy 4: Last resort - any diff-like content
    if '---' in txt and '+++' in txt:
        lines = txt.splitlines()
        start_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('---') and 'baseline' in l.lower()), -1)
        if start_idx < 0:
            start_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('---')), -1)
        if start_idx >= 0:
            result_lines = []
            for line in lines[start_idx:]:
                if line.startswith(('---', '+++', '@@', '-', '+', ' ')) or line.strip() == '':
                    result_lines.append(line)
                elif result_lines and not line.startswith(('---', '+++', '@@', '-', '+', ' ')):
                    if len(result_lines) > 3:
                        break
            if len(result_lines) >= 3:
                return '\n'.join(result_lines)

    return None


def copy_to_lemur(gen_nn_dir, name, task, dataset, metric):
    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f'{name}.py')
    dr_nm = new_lemur_stat_dir / f"{task}_{dataset}_{metric}_{name}"
    Path(dr_nm).mkdir(parents=True, exist_ok=True)
    for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+.json', f)]:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)

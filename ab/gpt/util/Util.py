import importlib
import inspect
import math
import os
import os.path
import re
import shutil
from pathlib import Path

from ab.gpt.util.Const import new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir

from ab.gpt.util.nn.Code import *


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
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"error when reading file: {e}")
        return None


def extract_str(s: str, start: str, end: str):
    try:
        if end in s:
            s = s[:s.rindex(end)]
            if start in s:
                spl = s.split(start)
                if len(spl) >= 1:
                    s = spl[-1]
                    spl = s.split(end)
                    if len(spl) >= 1:
                        s = spl[0]
                    return s.strip()
    except:
        pass
    return None


def extract_by_pattern(name, res, options) -> str:
    res = improve_code(next(filter(None, map(lambda l: extract_str(res, *l), options)), None))
    if res:
        print(f'[EXTRACT] ✓ Found {name}: {len(res)} chars')
    else:
        print(f'[EXTRACT] ✗ No {name} found')
    return res


# ── Tolerant NN-code extraction (was: strict <nn> tags + fence only) ──────────
# Real generations frequently drop the opening <nn>, wrap the model in a bare
# markdown fence, or emit an untagged `class Net`. The strict path lost all of
# those. The tolerant path below tries, in strict priority order, stopping at the
# first that yields valid code, and applies a rejection guard so junk ('...'
# echoes, sub-50-char fragments, anything lacking class/def) can never get in.
_FIX_E_S = '\x00<<EXTRACT_REGION_START>>\x00'
_FIX_E_E = '\x00<<EXTRACT_REGION_END>>\x00'
_ANCHOR_B = re.compile(r'(?m)^(import|from|class |def |supported_hyperparameters)')
_ANCHOR_D = re.compile(r'(?m)^(import|class |def )')


def _is_real_code(cand) -> bool:
    """True only for plausible NN code; False for the '...' echo, sub-50-char
    fragments, and anything missing class/def."""
    if not cand:
        return False
    nonws = re.sub(r'\s', '', cand)
    if not nonws:
        return False
    if set(nonws) <= {'.'}:          # pure '...' (or '.'-only) placeholder
        return False
    if len(nonws) < 50:              # too short to be a real model
        return False
    if 'class' not in cand or 'def' not in cand:
        return False
    return True


def _guarded(cand):
    """Apply the rejection guard to an already-extracted candidate."""
    if cand is None:
        return None
    if _is_real_code(cand):
        return cand
    n_nonws = len(re.sub(r'\s', '', cand))
    print(f'[EXTRACT] ✗ rejected non-code candidate ({n_nonws} non-ws chars)')
    return None


def _extract_region(region):
    """Run a raw candidate region through extract_by_pattern (sentinel-wrapped) so
    improve_code runs and the canonical '✓ Found ... N chars' line prints."""
    if not region or not region.strip():
        print('[EXTRACT] ✗ No NN code found')
        return None
    return extract_by_pattern('NN code', f'{_FIX_E_S}{region}{_FIX_E_E}', ((_FIX_E_S, _FIX_E_E),))


def _extract_code_strict(txt):
    """Original strict behavior: <nn> tags then markdown fences."""
    return extract_by_pattern('NN code', txt, (('<nn>', '</nn>'), ('```python', '```'), ('```', '```')))


def extract_code(txt):
    """Tolerant, junk-rejecting NN-code extraction. Returns the extracted code
    string, or None when nothing valid is found."""
    if not isinstance(txt, str):
        return _extract_code_strict(txt)

    has_open = '<nn>' in txt
    has_close = '</nn>' in txt

    # (a) Both tags -> original behavior; guard only ever fires on the degenerate
    #     '<nn>...</nn>' junk echo.
    if has_open and has_close:
        return _guarded(_extract_code_strict(txt))

    # (b) Closing tag only, opening tag missing.
    if has_close:
        m = _ANCHOR_B.search(txt[:txt.rindex('</nn>')])
        region = txt[m.start():txt.rindex('</nn>')] if m else None
        return _guarded(_extract_region(region))

    # Opening tag only (no close): no defined fallback -> mirror the original.
    if has_open:
        return _guarded(_extract_code_strict(txt))

    # ── No tags at all ────────────────────────────────────────────────────────
    # (c) Fenced block.
    cand = _guarded(extract_by_pattern(
        'NN code', txt, (('```python', '```'), ('```', '```'))))
    if cand is not None:
        return cand

    # (d) No tags, no usable fence, but a recognizable Net+forward model.
    if 'class Net' in txt and 'def forward' in txt:
        m = _ANCHOR_D.search(txt)
        region = txt[m.start():] if m else None
        cand = _guarded(_extract_region(region))
        if cand is not None:
            return cand

    print('[EXTRACT] ✗ No NN code found')
    return None


def extract_hyperparam(txt):
    return extract_by_pattern('hyper-parameters', txt.replace('< hp >', '<hp>').replace('<.hp>', '<hp>').replace('</ hp >', '</hp>'),
                              (('<hp>', '</hp>'), ('```json', '```')))


_NN_MODULE_CLASS = re.compile(r'class\s+\w+\s*\([^)]*nn\.Module[^)]*\)')


def is_nn_model_code(s) -> bool:
    """True when a string looks like a full NN model rather than a transform:
    it defines a `class Net` (or any nn.Module subclass) and has a `def forward`.
    Used to detect NN code misrouted into the <tr> transform slot."""
    if not isinstance(s, str) or 'def forward' not in s:
        return False
    return 'class Net' in s or bool(_NN_MODULE_CLASS.search(s))


def extract_transform(txt):
    return extract_by_pattern('transform code', txt.replace('< tr >', '<tr>').replace('<.tr>', '<tr>').replace('</ tr >', '</tr>'),
                              (('<tr>', '</tr>'),))


def extract_all_to_train(txt):
    return extract_code(txt), extract_hyperparam(txt), extract_transform(txt)


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
    for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+\.json', f)]:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)


# ========== FORMULA EVALUATION FUNCTION ==========
def evaluate_delimited_formulas(text: str, para_dict: dict) -> str:
    """
    Find patterns like <<accuracy / duration>> and replace with calculated values.
    Works for ANY formula inside << >> delimiters.
    """
    pattern = r'<<(.*?)>>'

    def replace_match(match):
        formula = match.group(1).strip()
        try:
            expr = formula
            # Replace variable names with their values
            for key in sorted(para_dict.keys(), key=len, reverse=True):
                val = para_dict[key]
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
                if isinstance(val, (int, float)):
                    expr = re.sub(rf'\b{re.escape(key)}\b', str(val), expr)

            # Safe evaluation
            safe_globals = {
                "__builtins__": {},
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
            }
            result = eval(expr, safe_globals)

            # Format result nicely
            if isinstance(result, float):
                if abs(result) < 0.001:
                    return f"{result:.2e}"
                elif result > 100:
                    return f"{result:.1f}"
                else:
                    return f"{result:.4f}"
            return str(result)
        except Exception as e:
            print(f"[FORMULA ERROR] '{formula}' - {e}")
            return f"<<{formula}>>"

    return re.sub(pattern, replace_match, text)
# =================================================


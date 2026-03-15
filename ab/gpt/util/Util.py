import importlib
import inspect
import os
import os.path
import re
import shutil
from pathlib import Path

from ab.gpt.util.Const import new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir

from ..util.Code import *


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
        spec = importlib.util.spec_from_file_location("module_name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source_code = inspect.getsource(module)
        return source_code
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


def extract_by_pattern(name, res, options) -> str:
    res = improve_code(next(filter(None, map(lambda l: extract_str(res, *l), options)), None))
    if res:
        print(f'[EXTRACT] ✓ Found {name}: {len(res)} chars')
    else:
        print(f'[EXTRACT] ✗ No {name} found')
    return res


def extract_code(txt):
    return extract_by_pattern('NN code', txt, (('<nn>', '</nn>'), ('```python', '```'), ('```', '```')))


def extract_hyperparam(txt):
    return extract_by_pattern('hyper-parameters', txt.replace('< hp >', '<hp>').replace('<.hp>', '<hp>').replace('</ hp >', '</hp>'),
                              (('<hp>', '</hp>'), ('```json', '```')))


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

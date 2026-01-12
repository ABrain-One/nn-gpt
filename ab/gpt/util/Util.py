import importlib
import inspect
import os.path
import re
import shutil
import ast
from pathlib import Path

from ab.gpt.util.Const import new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir
from ..util.Code import *


# todo: Verify that the model's accuracy does not decrease by more than 10%, or increase at some epochs
def nn_accepted(nn_dir):
    accepted = True
    return accepted


# todo: Verify if model has implementation of all required methods, and use all mentioned hyperparameters, like 'lr', 'momentum'
# todo: Optimize code with library like 'deadcode' (after: pip install deadcode)
def verify_nn_code(nn_dir, nn_file):
    verified = True
    error_message = ''
    if not verified:
        with open(nn_dir / f"error_code_verification.txt", "w+") as error_file:
            error_file.write(f"Code verification failed: {error_message}")
    return verified


def exists(f):
    return f and os.path.exists(f)


def extract_str(s: str, start: str, end: str):
    try:
        s = s[:s.rindex(end)]
        spl = s.split(start)
        if len(spl) > 1:
            s = spl[-1]
            spl = s.split(end)
            if len(spl) > 1:
                s = spl[0]
            return s.strip()
    except:
        pass
    return None


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


def extract_code(txt):
    return improve_code(next(filter(None, map(lambda l: extract_str(txt, *l),
                                              (('<nn>', '</nn>'), ('```python', '```'), ('```', '```')))), ''))


def extract_hyperparam(txt):
    return improve_code(next(filter(None, map(lambda l: extract_str(txt.replace('< hp >', '<hp>').replace('<.hp>', '<hp>').replace('</ hp >', '</hp>'), *l),
                                              (('<hp>', '</hp>'), ('```json', '```')))), ''))


def extract_transform(txt):
    return improve_code(next(filter(None, map(lambda l: extract_str(txt.replace('< tr >', '<tr>').replace('<.tr>', '<tr>').replace('</ tr >', '</tr>'),
                                                                    *l),
                                              (('<tr>', '</tr>'),))), ''))


def extract_delta(txt):
    """
    Extract delta (unified diff) from text.
    
    Looks for:
    1. <delta>...</delta> XML tags
    2. Unified diff format (lines starting with ---, +++, @@)
    
    Args:
        txt: Text containing delta
        
    Returns:
        Delta string or None if not found
    """
    # Try XML tags first
    delta = extract_str(txt.replace('< delta >', '<delta>').replace('<.delta>', '<delta>').replace('</ delta >', '</delta>'),
                        '<delta>', '</delta>')
    if delta:
        return delta

    # Try to extract unified diff format
    # Look for lines starting with ---, +++, or @@
    lines = txt.splitlines()
    delta_lines = []
    in_diff = False

    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            in_diff = True
            delta_lines.append(line)
        elif in_diff:
            if line.startswith('-') or line.startswith('+') or line.startswith(' '):
                delta_lines.append(line)
            elif line.strip() and not line.startswith('diff'):
                # End of diff block
                break

    if delta_lines:
        return '\n'.join(delta_lines)

    return None


def copy_to_lemur(gen_nn_dir, name, task, dataset, metric):
    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f'{name}.py')
    dr_nm = new_lemur_stat_dir / f"{task}_{dataset}_{metric}_{name}"
    Path(dr_nm).mkdir(parents=True, exist_ok=True)
    for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+\.json', f)]:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)

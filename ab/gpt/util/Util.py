import importlib
import inspect
import math
import os
import os.path
import re
import shutil
from pathlib import Path

from ab.nn.util.Util import uuid4
from ab.gpt.util.Const import (hp_file, new_lemur_nn_dir, new_lemur_stat_dir,
                               new_lemur_transform_dir, new_nn_file,
                               transformer_file)

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


_FENCE_OPEN = re.compile(r'\A\s*```[ \t]*(?:python|py|python3)?[ \t]*\r?\n')
_FENCE_CLOSE = re.compile(r'\r?\n[ \t]*```[ \t]*\s*\Z')


def _strip_code_fence(cand):
    """Remove a markdown fence the model put inside its <nn> block (a leading
    ```python / ``` line and a trailing ``` line); the fence made the saved code
    fail to parse at line 1. Code without a fence is returned unchanged."""
    if cand is None:
        return None
    stripped = _FENCE_CLOSE.sub('', _FENCE_OPEN.sub('', cand, count=1), count=1)
    return stripped.strip() if stripped != cand else cand


def _guarded(cand):
    """Apply the rejection guard to an already-extracted candidate."""
    cand = _strip_code_fence(cand)
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


#: Generated transform names: "tr-<uid>".
#:
#: A hyphen, never an underscore. The dataset's own transforms are named this
#: way -- bf-v1-CenterCrop_Pad and 223 others -- so the loader clearly resolves
#: them by file and not by importing ab.nn.transform.<name>, where a hyphen
#: would not be a legal module name. Not configurable: an underscore separator
#: would produce exactly the name shape this avoids.
TRANSFORM_NAME_SEP = "-"

TRANSFORM_NAME_PREFIX = "tr"


def normalise_transform_code(code) -> str:
    """A transform's text with formatting that cannot change behaviour removed.

    Blank lines and trailing whitespace only. Measured need: one cycle shipped
    six generated transforms of which three were the same pipeline differing by
    a blank line after the import, and two more were another such pair -- six
    files, three transforms, and none of them checked against the ones the
    dataset already holds.
    """
    return "\n".join(line.rstrip() for line in (code or "").strip().splitlines()
                      if line.strip())


def transform_uid(code) -> str:
    """The project's standard uid, over the normalised transform text.

    The same ab.nn.util.Util.uuid4 the dataset uses to name a model, so a
    transform is identified the way everything else in the pipeline is. No
    local fallback: a divergent uid would silently split one transform into
    two, which is the bug this exists to fix.
    """
    return uuid4(normalise_transform_code(code))


def known_transforms() -> dict:
    """``{uid: name}`` for every transform already available by name.

    Both sources that matter: the transforms the installed dataset ships, and
    the ones earlier cycles of this run already generated and submitted. A
    transform found here needs no new file and no new name -- the row simply
    refers to the name that already exists.
    """
    index = {}
    dirs = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("ab.nn")
        roots = []
        if spec is not None:
            if spec.origin:
                roots.append(Path(spec.origin).parent)
            # A namespace package has no origin, only search locations.
            for loc in (spec.submodule_search_locations or ()):
                roots.append(Path(loc))
        for root in roots:
            d = root / "transform"
            if d.is_dir() and d not in dirs:
                dirs.append(d)
    except Exception:
        pass
    dirs.append(Path(new_lemur_transform_dir))
    for d in dirs:
        try:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                try:
                    code = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if "def transform" not in code:
                    continue
                # First writer wins, and the dataset's own directory is first,
                # so a dataset name is never shadowed by a generated one.
                index.setdefault(transform_uid(code), f.stem)
        except Exception:
            continue
    return index


def resolve_transform_name(code):
    """``(name, is_new)`` for a transform. Reuses an existing name when it can.

    This is the check that was missing: a generated transform was named from a
    hash of its raw text, so the same pipeline written with a different blank
    line became a new transform, and one the dataset already had under
    norm_32_flip became a second copy under a generated name.
    """
    uid = transform_uid(code)
    existing = known_transforms().get(uid)
    if existing:
        return existing, False
    return f"{TRANSFORM_NAME_PREFIX}{TRANSFORM_NAME_SEP}{uid}", True


def generated_transform_name(code) -> str:
    """The name a transform should be recorded under. Kept for callers."""
    return resolve_transform_name(code)[0]


def copy_generated_transform(gen_nn_dir, name=None):
    """Put this model's transform module into the submission, next to nn/.

    A statistics row records ``prm["transform"]`` by name. If the generated
    module is not shipped alongside, the submitted row refers to a transform
    that is not in the submission and cannot be reproduced from it. Returns the
    name written, or None when the model used a dataset transform that the
    LEMUR tree already holds.
    """
    tr_path = Path(gen_nn_dir) / transformer_file
    if not tr_path.is_file():
        return None
    try:
        code = tr_path.read_text(encoding='utf-8')
    except Exception:
        return None
    if not code.strip():
        return None
    if not name:
        # Prefer the name the evaluation actually trained under.
        hp_path = Path(gen_nn_dir) / hp_file
        if hp_path.is_file():
            try:
                import json as _json
                prm = _json.loads(hp_path.read_text(encoding='utf-8'))
                if isinstance(prm, dict) and isinstance(prm.get('transform'), str):
                    name = prm['transform']
            except Exception:
                name = None
    resolved, is_new = resolve_transform_name(code)
    if not is_new:
        # Already in the dataset under its own name. Shipping a second copy
        # under a different name is exactly the duplication this avoids.
        return resolved
    if not name or name != resolved:
        name = resolved
    Path(new_lemur_transform_dir).mkdir(parents=True, exist_ok=True)
    target = new_lemur_transform_dir / f'{name}.py'
    if not target.exists():
        target.write_text(code, encoding='utf-8')
    return name


def copy_to_lemur(gen_nn_dir, name, task, dataset, metric):
    """Stage one trained network and its statistics for submission.

    A submission is a network *and* the measurements behind it. This used to
    create the statistics directory first and then copy whatever numbered JSON
    it found, so a model whose statistics never reached gen_nn_dir -- the
    trainer warns and carries on when save_path is unset -- was staged with an
    empty directory beside it. The archive counted the directory, reported the
    record, and the submission claimed a measurement it did not have.

    So the statistics decide. No numbered JSON, nothing staged.
    """
    try:
        stats = sorted(f for f in os.listdir(gen_nn_dir)
                       if re.match(r'[0-9]+\.json$', f))
    except OSError as exc:
        print(f'[submission] {name} skipped: cannot read {gen_nn_dir} ({exc})',
              flush=True)
        return
    if not stats:
        print(f'[submission] {name} skipped: no statistics in {gen_nn_dir}; '
              f'a network is not submitted without the measurement behind it',
              flush=True)
        return
    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f'{name}.py')
    dr_nm = new_lemur_stat_dir / f"{task}_{dataset}_{metric}_{name}"
    Path(dr_nm).mkdir(parents=True, exist_ok=True)
    for f_nm in stats:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)
    try:
        written = copy_generated_transform(gen_nn_dir)
        if written:
            print(f'[submission] transform {written}.py copied for {name}', flush=True)
    except Exception as exc:
        print(f'[submission] could not copy the transform for {name}: {exc}', flush=True)


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


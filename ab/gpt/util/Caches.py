"""Keep regenerable caches inside the project's ``out`` directory.

Several libraries in this stack write compiled artefacts to a path relative to
the current working directory or to the user's home:

* unsloth resolves ``UNSLOTH_COMPILE_LOCATION`` as a BARE RELATIVE PATH, so the
  generated ``UnslothSFTTrainer.py`` lands in whatever directory the run was
  started from -- for this repo that is ``ab/gpt/act/tune/unsloth_compiled_cache``,
  inside the source tree and inside git's view of it;
* torch inductor, triton, torch extensions, numba, matplotlib and the CUDA JIT
  all default to ``/tmp`` or ``~/.cache``.

Pinning them under ``out/cache`` keeps the checkout clean, puts every
regenerable artefact in one place that can be deleted wholesale, and makes the
compiled cache follow the project rather than the shell's cwd.

A cache compiled against one torch build is not valid for another, so a stale
cache after an upgrade shows up as a crash deep inside the generated trainer.
``clear_caches()`` exists for that case.

An environment value set by the caller always wins; blank counts as unset.
"""

import os
import shutil
from pathlib import Path

#: name -> path relative to the cache root
CACHE_VARS = {
    'UNSLOTH_COMPILE_LOCATION': 'unsloth_compiled_cache',
    'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
    'TRITON_CACHE_DIR': 'triton/cache',
    'TRITON_DUMP_DIR': 'triton/dump',
    'TORCH_EXTENSIONS_DIR': 'torch_extensions',
    'CUDA_CACHE_PATH': 'nv_compute_cache',
    'NUMBA_CACHE_DIR': 'numba',
    'MPLCONFIGDIR': 'matplotlib',
}

# HF_HOME is deliberately NOT redirected: it holds the auth token, and moving it
# would detach the run from an existing login. TRITON_HOME is left alone for the
# same reason upstream leaves it -- it also covers hand-written kernel overrides.


def cache_root(root=None):
    """Directory holding every redirected cache."""
    if root:
        return Path(root)
    env = os.environ.get('NNGPT_CACHE_ROOT', '').strip()
    if env:
        return Path(env)
    # Imported here, not at module scope: setup_caches() must run before torch
    # is imported, and this import may pull torch in transitively.
    from ab.nn.util.Const import out_dir
    return Path(out_dir) / 'cache'


def setup_caches(root=None, verbose=True):
    """Point the cache environment variables under ``out/cache``.

    Must run BEFORE torch, unsloth or triton are imported: each reads its
    variable at import time. Returns the cache root. Never raises.
    """
    base = cache_root(root)
    applied, kept = [], []

    for var, relative in CACHE_VARS.items():
        existing = os.environ.get(var, '').strip()
        if existing:
            kept.append(var)
            continue
        path = base / relative
        # torch/_inductor/cpp_builder.py splits some toolchain paths on spaces,
        # so a path containing whitespace is worse than the default.
        if ' ' in str(path):
            kept.append(var)
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f'[CACHE] Could not create {path}: {exc}')
            continue
        os.environ[var] = str(path)
        applied.append(var)

    if verbose:
        print(f'[CACHE] Regenerable caches under {base} ({len(applied)} redirected)'
              + (f'; left as set by the environment: {", ".join(kept)}' if kept else ''))
    return base


def clear_caches(root=None):
    """Delete the redirected caches. Returns the number of directories removed.

    A compiled cache is only valid for the torch build that produced it; after a
    torch or unsloth upgrade the stale copy surfaces as an error inside the
    generated trainer rather than as a version complaint.
    """
    base = cache_root(root)
    if not base.exists():
        return 0
    removed = 0
    for relative in CACHE_VARS.values():
        path = base / relative
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except Exception as exc:
            print(f'[CACHE] Could not remove {path}: {exc}')
    print(f'[CACHE] Cleared {removed} cache directory(ies) under {base}')
    return removed

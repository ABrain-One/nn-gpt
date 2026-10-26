"""Generation-seed selection for the NNGPT generate -> evaluate -> fine-tune loop.

``nn_gen()`` picks the seed networks each epoch starts from. Drawing them afresh
every epoch makes the epochs incomparable, because the LEMUR pool *grows*: every
evaluated candidate is written back, so a later epoch samples from a larger and
increasingly self-generated corpus. The accuracy of the networks "before
modification" then moves with the draw rather than with the model, and neither a
per-epoch improvement nor a trend across epochs can be attributed to fine-tuning.

A fixed ``random_state`` does **not** solve this: ``groupby("nn").sample()`` sees
more groups each epoch, so the same seed yields a different set. The chosen
*names* have to be recorded. That is what this module does.

Pinning is on by default and is a within-run property: the file is written on the
first epoch of a run and reused by the rest, and ``reset_pinned_seeds()`` clears
it when a new run starts, so one run never inherits another's draw.

What stays unchanged
--------------------
* ``NNGPT_SEED_SHUFFLE`` still controls the initial draw, so a run remains
  reproducible end to end; it is simply not re-rolled every epoch.
* With pinning disabled the behaviour is exactly the previous one, including the
  per-cycle ``cycle_from_path`` offset.

Environment
-----------
``NNGPT_PIN_SEEDS``   ``0``/``false``/``no``/``off`` disables pinning.
``NNGPT_SEED_FILE``   Path of the pinned list (default ``<nngpt_dir>/generation_seeds.json``).
``NNGPT_SEED_SHUFFLE``  Run seed for the initial draw, as before.
"""

import json
import os
from pathlib import Path

from ab.gpt.util.Const import nngpt_dir

SEED_FILE_NAME = "generation_seeds.json"

#: Default for ``pin_seeds``; ``ab/gpt/act/tune/Tune.py`` imports this so the
#: CLI default and the library default cannot drift apart.
PIN_SEEDS = True

_FALSE = {"0", "false", "no", "off", "none", ""}


def seed_file_path(seed_file=None):
    """Where the pinned seed list lives.

    Explicit argument wins, then ``NNGPT_SEED_FILE``, then the default inside
    ``nngpt_dir``.
    """
    if seed_file:
        return Path(seed_file)
    env = os.environ.get("NNGPT_SEED_FILE", "").strip()
    return Path(env) if env else Path(nngpt_dir) / SEED_FILE_NAME


def pinning_enabled(enabled=None):
    """Whether seeds should be pinned. ``enabled`` overrides the environment."""
    if enabled is not None:
        return bool(enabled)
    raw = os.environ.get("NNGPT_PIN_SEEDS")
    if raw is None:
        return PIN_SEEDS
    return raw.strip().lower() not in _FALSE


def set_pinning(enabled):
    """Publish the choice for the worker processes ``nn_gen()`` runs in."""
    os.environ["NNGPT_PIN_SEEDS"] = "1" if enabled else "0"
    return bool(enabled)


def read_pinned_seeds(seed_file=None):
    """The recorded seed names, or None when nothing is pinned yet."""
    path = seed_file_path(seed_file)
    if not path.is_file():
        return None
    try:
        names = json.loads(path.read_text()).get("nn")
    except Exception as exc:
        print(f"[seeds] could not read {path}: {exc}", flush=True)
        return None
    return list(names) if names else None


def write_pinned_seeds(names, seed_file=None, epoch=None):
    """Record the seed names for the rest of the run. Never raises."""
    path = seed_file_path(seed_file)
    payload = {"nn": list(names)}
    if epoch is not None:
        payload["pinned_at_epoch"] = epoch
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"[seeds] pinned {len(payload['nn'])} seeds -> {path}", flush=True)
    except Exception as exc:
        print(f"[seeds] could not persist seeds: {exc}", flush=True)
    return path


def reset_pinned_seeds(seed_file=None):
    """Drop the previous run's pinned list so a new run draws its own.

    Returns the path when something was removed, else None. Never raises: a
    file that cannot be deleted is reported and the run continues.

    ``NNGPT_SEED_FILE_EXTERNAL`` suppresses the reset. A caller that builds the
    seed list itself -- a held-out test set written before the run starts --
    would otherwise have it deleted here and silently replaced by a fresh draw
    over the whole corpus, so the models used to measure the run would be the
    same ones the fine-tune was trained on.
    """
    if os.environ.get("NNGPT_SEED_FILE_EXTERNAL", "").strip().lower() not in _FALSE:
        print("[seeds] external seed file: keeping the pinned set", flush=True)
        return None
    path = seed_file_path(seed_file)
    if not path.is_file():
        return None
    try:
        path.unlink()
        print(f"[seeds] cleared the previous run's pinned seeds: {path}", flush=True)
        return path
    except Exception as exc:
        print(f"[seeds] could not remove {path}: {exc}", flush=True)
        return None


def _fresh_draw(data, test_nn, out_path):
    """The original selection, unchanged, including the per-cycle offset."""
    shuffle_seed = os.environ.get("NNGPT_SEED_SHUFFLE", "").strip()
    if shuffle_seed:
        from ab.gpt.util.method.llmatic.seeds import cycle_from_path
        rs = int(shuffle_seed) + cycle_from_path(out_path)
        print(f"[seeds] shuffle random_state={rs}", flush=True)
        return (data.groupby(by="nn").sample(n=1, random_state=rs)
                    .sample(frac=1, random_state=rs)[:test_nn])
    return data.groupby(by="nn").sample(n=1).sample(frac=1)[:test_nn]


def pin_generation_seeds(data, test_nn, out_path, seed_file=None,
                         enabled=None, epoch=None):
    """Return this epoch's seed rows, holding the set fixed across the run.

    Drop-in replacement for the inline
    ``data.groupby(by="nn").sample(n=1).sample(frac=1)[:test_nn]`` in
    ``nn_gen()``. With pinning disabled it is exactly that expression.

    ``data`` is the LEMUR frame, one row per (architecture, trial); ``out_path``
    is the epoch directory, used only to derive the cycle for the shuffle seed.
    """
    if not pinning_enabled(enabled):
        return _fresh_draw(data, test_nn, out_path)

    pinned = read_pinned_seeds(seed_file)
    if pinned:
        subset = data[data["nn"].isin(pinned)]
        if not subset.empty:
            chosen = subset.groupby(by="nn").sample(n=1, random_state=0)
            # Restore the recorded order; groupby sorts by name, which would
            # otherwise reorder the set and change which rows survive [:test_nn].
            order = {name: i for i, name in enumerate(pinned)}
            chosen = chosen.sort_values(by="nn", key=lambda col: col.map(order))
            missing = len(pinned) - len(chosen)
            if missing:
                print(f"[seeds] {missing} of {len(pinned)} pinned seeds are no "
                      f"longer in the corpus; continuing with {len(chosen)}",
                      flush=True)
            print(f"[seeds] reusing {len(chosen)} pinned seeds", flush=True)
            return chosen[:test_nn]
        print("[seeds] none of the pinned seeds are in the corpus; reselecting",
              flush=True)

    chosen = _fresh_draw(data, test_nn, out_path)
    write_pinned_seeds(chosen["nn"].tolist(), seed_file, epoch=epoch)
    return chosen
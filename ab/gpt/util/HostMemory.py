"""Return host memory to the operating system between fine-tuning cycles.

``release_memory()`` collects garbage and empties the CUDA caching allocator,
which is enough for device memory and for anything held only by a Python
reference. Neither covers the three ways a long run accumulates *host* memory:

* A process-global container can hold the allocation. bitsandbytes' paged
  optimizers register every state buffer with a module-level page manager, and
  a fresh trainer -- built once per cycle -- adds a new set. ``gc.collect()``
  cannot free what a live global list references, so the optimizer state of
  every cycle stays resident for the whole run.
* PyTorch's pinned (page-locked) host allocator caches its blocks and never
  returns them to the OS. ``torch.cuda.empty_cache()`` is about device memory
  and does not touch it.
* glibc keeps freed pages in its arena rather than handing them back, so a
  large transient working set -- reading and quantising model shards, or
  building the supervised corpus -- raises the resident set permanently even
  though Python freed every object.

Each cycle of a 25-50 cycle run pays all three, which is why such a run ends
with a few GiB free on a machine that started with 31, and why the evaluation
worker is then the process the kernel kills.

Every step here is optional and guarded: these are implementation details of
other projects, so a step that does not apply to the installed versions is
skipped rather than failing the run.
"""

import ctypes
import gc
import sys


def host_rss_gib():
    """This process's resident set in GiB, or None if it cannot be read."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_available_gib():
    """Memory the host can still hand out, in GiB, or None."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _drop_cuda_caches():
    try:
        import torch
    except Exception:
        return
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return
    for name in ("empty_cache", "ipc_collect"):
        fn = getattr(torch.cuda, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _drop_pinned_host_cache():
    """Release PyTorch's cached pinned host blocks, where the build allows it.

    Pinned memory is allocated for dataloader staging and for offloaded
    activation checkpoints. The caching host allocator keeps those blocks for
    reuse, and because every cycle's corpus has a slightly different shape the
    set of cached block sizes grows rather than being reused. The API for
    emptying it has moved between releases and is private in most, so each
    known spelling is tried and a build that offers none is left alone.
    """
    try:
        import torch
    except Exception:
        return 0
    released = 0
    candidates = [
        ("torch.cuda.host_empty_cache", getattr(getattr(torch, "cuda", None), "host_empty_cache", None)),
        ("torch._C._host_emptyCache", getattr(getattr(torch, "_C", None), "_host_emptyCache", None)),
    ]
    for _name, fn in candidates:
        if callable(fn):
            try:
                fn()
                released += 1
            except Exception:
                pass
    return released


def _prune_paged_optimizer_state():
    """Forget bitsandbytes' record of paged tensors from previous cycles.

    A paged optimizer deliberately allocates its state in host-backed managed
    memory, and ``Optimizer8bit.get_state_buffer`` appends each buffer to a
    module-level page manager. The trainer is rebuilt every cycle, so each
    cycle appends a new set and nothing ever removes the old ones: the list is
    a strong reference, so the buffers survive ``del trainer`` and every
    ``gc.collect()`` that follows.

    Dropping the entries lets that memory go. The list is bookkeeping for
    prefetch hints rather than something the optimizer reads back, and the
    optimizer that owned the previous entries no longer exists -- but if a
    future version relies on it, the worst case is losing a prefetch hint for
    tensors that are already gone.

    Returns the number of entries dropped, or 0 when the library is absent or
    its internals have changed.
    """
    try:
        import bitsandbytes.functional as bnb_functional
    except Exception:
        return 0

    manager_cls = getattr(bnb_functional, "GlobalPageManager", None)
    if manager_cls is None:
        return 0
    get_instance = getattr(manager_cls, "get_instance", None)
    if not callable(get_instance):
        return 0
    try:
        manager = get_instance()
    except Exception:
        return 0

    dropped = 0
    for attr in ("paged_tensors", "tensors"):
        registry = getattr(manager, attr, None)
        if registry is None:
            continue
        try:
            count = len(registry)
            registry.clear()
        except Exception:
            # Not a list, or not clearable: leave that one exactly as it was
            # without discarding what an earlier attribute already reported.
            continue
        dropped += count
    return dropped


def _trim_glibc_arena():
    """Hand the allocator's free pages back to the kernel."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if not callable(malloc_trim):
            return False
        malloc_trim(0)
        return True
    except Exception:
        return False


def release_host_memory(label="", verbose=True):
    """Give back as much host memory as this process can, and say how much.

    Called at the end of every cycle. The numbers are logged rather than kept
    because the useful signal is the trend: a resident set that returns to
    roughly the same value each cycle means the run can continue indefinitely,
    while one that climbs says memory is still being retained somewhere and
    names the cycle where it started.
    """
    before = host_rss_gib()
    gc.collect()
    _drop_cuda_caches()
    pinned = _drop_pinned_host_cache()
    paged = _prune_paged_optimizer_state()
    gc.collect()
    trimmed = _trim_glibc_arena()
    after = host_rss_gib()

    if verbose:
        parts = []
        if before is not None and after is not None:
            parts.append(f"resident {before:.2f} -> {after:.2f} GiB")
        elif after is not None:
            parts.append(f"resident {after:.2f} GiB")
        available = host_available_gib()
        if available is not None:
            parts.append(f"host available {available:.2f} GiB")
        if paged:
            parts.append(f"dropped {paged} paged optimizer buffers")
        if pinned:
            parts.append("emptied pinned host cache")
        if trimmed:
            parts.append("arena trimmed")
        suffix = f" after {label}" if label else ""
        print(f"[HostMemory] Released{suffix}: " + ", ".join(parts), flush=True)

    return {
        "rss_before_gib": before,
        "rss_after_gib": after,
        "paged_buffers_dropped": paged,
        "pinned_cache_emptied": pinned,
        "arena_trimmed": trimmed,
        "host_available_gib": host_available_gib(),
    }

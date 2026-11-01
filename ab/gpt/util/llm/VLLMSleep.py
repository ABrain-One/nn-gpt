import ctypes
import gc
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import torch

from ab.gpt.util.Util import extract_all_to_train

MIN_NEW_TOKENS = 256
SKIPPED_OUTPUT = (None, None, None, "")
GIB = 1024 ** 3

# A park that frees less than this did not really happen. bitsandbytes
# Params4bit can accept .to("cpu") while keeping its quantized storage on the
# device, so the move is verified against the allocator instead of trusted.
MIN_PARK_FREED = 1 * GIB


# Sampling fields that model.generate() takes from the model's generation_config.json whenever the
# caller does not pass them, mapped to their SamplingParams names. The HF path passes temperature,
# top_k, top_p and the token budget explicitly but inherits these, so the vLLM path must apply them
# too or the two backends sample from different distributions (Qwen2.5-Coder: repetition_penalty 1.1).
_INHERITED_SAMPLING = {"repetition_penalty": "repetition_penalty", "min_p": "min_p"}
# Fields that also change HF sampling but have no SamplingParams equivalent: reported, not applied.
_HF_ONLY_SAMPLING = ("no_repeat_ngram_size", "typical_p", "epsilon_cutoff", "eta_cutoff",
                     "encoder_repetition_penalty")
_NEUTRAL_SAMPLING = {"repetition_penalty": 1.0, "min_p": 0.0, "no_repeat_ngram_size": 0,
                     "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                     "encoder_repetition_penalty": 1.0}


def read_generation_config(*model_paths):
    """The generation_config of the first model path that has one, as (dict, source); ({}, None) if none."""
    for path in model_paths:
        if not path:
            continue
        f = Path(path) / "generation_config.json"
        if f.is_file():
            try:
                return json.loads(f.read_text()), str(f)
            except (OSError, ValueError):
                pass
        try:
            from transformers import GenerationConfig
            return GenerationConfig.from_pretrained(path).to_dict(), str(path)
        except Exception:
            pass
    return {}, None


def inherited_sampling_params(generation_config, overrides=None):
    """SamplingParams kwargs that the HF path inherits from generation_config.

    Returns (kwargs, ignored): kwargs holds every inherited field with a non-neutral value, with explicit
    overrides (values that are not None) taking precedence; ignored holds non-neutral HF-only fields.
    """
    kwargs = {}
    for hf_name, vllm_name in _INHERITED_SAMPLING.items():
        value = generation_config.get(hf_name)
        if value is not None and value != _NEUTRAL_SAMPLING[hf_name]:
            kwargs[vllm_name] = value
    for name, value in (overrides or {}).items():
        if value is not None:
            kwargs[name] = value
    ignored = {k: generation_config[k] for k in _HF_ONLY_SAMPLING
               if generation_config.get(k) not in (None, _NEUTRAL_SAMPLING[k])}
    return kwargs, ignored


def _env_flag(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(float(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


_ALLOC_VARS = ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')


def _strip_expandable_segments():
    """Drop expandable_segments from the allocator config; returns the old values.

    Sleep mode allocates through vLLM's CuMemAllocator, which uses the CUDA
    virtual-memory API. PyTorch's expandable_segments uses the same API and the
    two cannot coexist -- vLLM asserts rather than corrupting memory
    (pytorch/pytorch#147851). Unsloth sets expandable_segments:True to limit
    fragmentation during training, which is how it reaches the engine.

    The engine runs in a child process that inherits os.environ, so clearing the
    flag here reaches it. The parent's allocator was configured at its first CUDA
    allocation and is unaffected, so the trainer keeps the behaviour it started
    with; the value is restored once the engine is built.
    """
    saved = {}
    for var in _ALLOC_VARS:
        value = os.environ.get(var)
        if value is None or 'expandable_segments' not in value:
            continue
        saved[var] = value
        kept = [part for part in value.split(',')
                if 'expandable_segments' not in part]
        if kept:
            os.environ[var] = ','.join(kept)
        else:
            os.environ.pop(var, None)
        print(f"[vLLM] {var}: dropped expandable_segments for the engine "
              f"(incompatible with sleep mode)", flush=True)
    return saved


def _restore_alloc_conf(saved):
    for var, value in saved.items():
        os.environ[var] = value


#: Destroy the engine when an epoch's generation is done, instead of parking it
#: with sleep(). A parked engine keeps the CUDA context, the compiled graphs and
#: the pool reserved through gpu_memory_utilization, which is most of the card --
#: memory the evaluator needs to train the networks that were just generated.
#: The cost is one engine build per epoch; set NNGPT_VLLM_TEARDOWN=0 to trade
#: that memory back for the faster wake-up.
TEARDOWN_BETWEEN_EPOCHS = _env_flag("NNGPT_VLLM_TEARDOWN", True)


def sleep_mode_enabled():
    return _env_flag("NNGPT_VLLM_SLEEP", True)


def _free_gpu():
    if not torch.cuda.is_available():
        return 0, 0
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return free, total


# ---------------------------------------------------------------------------
# Parking the trainer
#
# The engine runs in its own OS process, so it cannot borrow the trainer's
# weights: CUDA memory is not shared across processes and vLLM builds the model
# from the checkpoint with its own layer implementations. Generating therefore
# needs the card to itself, and the trainer has to leave it first.
#
# Two things made that fail silently before:
#
#   1. The model's device was read from next(model.parameters()).device.
#      Unsloth offloads embed_tokens to host RAM ("Offloading embeddings to RAM
#      to save 2.37 GB") and that tensor can come first in iteration order, so
#      the first parameter reads as cpu while ~15 GiB of quantized layers sit on
#      the card. park_trainer concluded the trainer was already parked and
#      returned without moving anything or logging anything.
#
#   2. Even when it did run, Module.to("cpu") is not enough for a bitsandbytes
#      4-bit model. The packed uint8 .data may move while the QuantState
#      (absmax / code / offset, and a nested state2 under double quantization)
#      stays on the device, so little is freed -- and a weight whose quant_state
#      is elsewhere cannot be used.
#
# Both are addressed by measuring bytes instead of trusting a device attribute,
# and by moving the tensors explicitly.
# ---------------------------------------------------------------------------

#: QuantState fields that are tensors. shape/dtype/blocksize/quant_type are not.
_QUANT_STATE_TENSOR_ATTRS = ("absmax", "code", "offset")

#: Set NNGPT_VLLM_FORCE_PARK=0 to go back to a plain Module.to("cpu"), which on
#: a 4-bit model frees little and leaves the engine sharing the card.
FORCE_PARK = _env_flag("NNGPT_VLLM_FORCE_PARK", True)

#: Checksum a sample of every moved tensor before parking and after restoring.
#: Cheap (64 elements per tensor) and it is the only thing that would catch a
#: quant_state left behind on the device, which corrupts weights silently
#: instead of raising. OFF by default: it costs a sampling pass over every
#: tensor twice per cycle, and the failure it guards is specific to quantized
#: parking. Set NNGPT_VLLM_PARK_VERIFY=1 for the first run on a new model, or
#: after a bitsandbytes / unsloth upgrade.
PARK_VERIFY = _env_flag("NNGPT_VLLM_PARK_VERIFY", False)

#: Park the trainer's weights to disk instead of host RAM. The trainer is idle
#: for the whole generation phase, and holding its 7+ GiB there is what leaves
#: the host without room for anything else.
PARK_TO_DISK = _env_flag("NNGPT_VLLM_PARK_TO_DISK", True)


def _host_available_gib():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return 0.0


def _host_dirty_gib():
    total = 0
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith(("Dirty:", "Writeback:")):
                    total += int(line.split()[1])
    except Exception:
        return 0.0
    return total / 2**20


def _park_dir():
    override = os.environ.get("NNGPT_VLLM_PARK_DIR", "").strip()
    if override:
        d = Path(override)
    else:
        try:
            from ab.gpt.util.Const import out_dir
            d = Path(out_dir) / "tmp"
        except Exception:
            d = Path(tempfile.gettempdir())
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Spill:
    """Weights written to disk as raw bytes, in the order they were taken."""

    def __init__(self, path):
        self.path = Path(path)
        self.count = 0
        self.plan = []

    @staticmethod
    def _byte_view(tensor):
        flat = tensor.contiguous().reshape(-1)
        if flat.dtype == torch.bfloat16:
            flat = flat.view(torch.int16)
        return memoryview(flat.numpy()).cast("B")

    def write(self, handle, tensor):
        contiguous = tensor.contiguous()
        payload = self._byte_view(contiguous)
        handle.write(payload)
        self.plan.append((contiguous.dtype, tuple(contiguous.shape),
                          payload.nbytes))
        self.count += 1

    def read(self, handle, index):
        dtype, shape, nbytes = self.plan[index]
        view = torch.int16 if dtype == torch.bfloat16 else dtype
        flat = torch.empty(nbytes // torch.empty(0, dtype=view).element_size(),
                           dtype=view)
        got = handle.readinto(memoryview(flat.numpy()).cast("B"))
        if got != nbytes:
            raise RuntimeError(f"spilled weight {index} is short: "
                               f"{got} of {nbytes} bytes")
        if dtype == torch.bfloat16:
            flat = flat.view(torch.bfloat16)
        return flat.reshape(shape)

    @property
    def bytes(self):
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def discard(self):
        try:
            self.path.unlink()
        except OSError:
            pass


def _tensor_bytes(t):
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


def _iter_quant_states(tensor):
    """Yield the QuantState of a bitsandbytes parameter and any nested states.

    Plain tensors have no quant_state and yield nothing, so this is safe to call
    on every parameter and buffer.
    """
    seen = set()
    stack = [getattr(tensor, "quant_state", None)]
    while stack:
        state = stack.pop()
        if state is None or id(state) in seen:
            continue
        seen.add(id(state))
        yield state
        stack.append(getattr(state, "state2", None))


def _iter_model_tensors(model):
    """Yield (store, key, tensor) for every parameter and buffer of the model.

    Iterating the _parameters / _buffers dicts rather than named_parameters()
    gives the container, which is what makes an in-place swap possible.
    """
    if model is None:
        return
    try:
        modules = list(model.modules())
    except Exception:
        return
    for module in modules:
        for store_name in ("_parameters", "_buffers"):
            store = getattr(module, store_name, None)
            if not store:
                continue
            for key, tensor in list(store.items()):
                if tensor is None or not torch.is_tensor(tensor):
                    continue
                yield store, key, tensor


def _model_gpu_bytes(model):
    """GPU bytes the model really holds: parameters, buffers and quant states.

    This replaces "is the first parameter on cuda?" as the test for whether the
    trainer is on the card. Shared tensors are counted once.
    """
    total = 0
    seen = set()
    for _store, _key, tensor in _iter_model_tensors(model):
        if tensor.device.type == "cuda" and id(tensor) not in seen:
            seen.add(id(tensor))
            total += _tensor_bytes(tensor)
        for state in _iter_quant_states(tensor):
            for attr in _QUANT_STATE_TENSOR_ATTRS:
                q = getattr(state, attr, None)
                if (torch.is_tensor(q) and q.device.type == "cuda"
                        and id(q) not in seen):
                    seen.add(id(q))
                    total += _tensor_bytes(q)
    return total


def _move_quant_state(tensor, device):
    """Move a bitsandbytes quant_state to `device`, nested states included.

    bitsandbytes' own QuantState.to() is preferred when present, because it
    knows the layout of the version actually installed; the attribute walk is
    the fallback for versions that do not define it.
    """
    for state in _iter_quant_states(tensor):
        to_fn = getattr(state, "to", None)
        if callable(to_fn):
            try:
                to_fn(device)
                continue
            except Exception:
                pass
        for attr in _QUANT_STATE_TENSOR_ATTRS:
            q = getattr(state, attr, None)
            if torch.is_tensor(q) and q.device != device:
                try:
                    setattr(state, attr, q.to(device))
                except Exception as exc:
                    print(f"[vLLM] quant_state.{attr} could not be moved to "
                          f"{device}: {type(exc).__name__}: {exc}", flush=True)


def _fingerprint(model):
    """Cheap checksum over a sample of every GPU-or-host tensor of the model.

    64 elements per tensor, plus the full absmax/code/offset sums, which is
    where a dropped quant_state would show up.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    try:
        for _store, key, tensor in _iter_model_tensors(model):
            h.update(key.encode())
            flat = tensor.detach().reshape(-1)
            sample = flat[:64].to("cpu", copy=True)
            h.update(sample.float().numpy().tobytes())
            h.update(str(tuple(tensor.shape)).encode())
            for state in _iter_quant_states(tensor):
                for attr in _QUANT_STATE_TENSOR_ATTRS:
                    q = getattr(state, attr, None)
                    if torch.is_tensor(q):
                        h.update(attr.encode())
                        h.update(q.detach().reshape(-1)[:64]
                                 .to("cpu", copy=True).float().numpy().tobytes())
        return h.hexdigest()
    except Exception as exc:
        print(f"[vLLM] park fingerprint unavailable "
              f"({type(exc).__name__}: {exc}); the park/restore round trip "
              f"will not be verified.", flush=True)
        return None


class ParkedTrainer:
    """Where every tensor of the trainer was, so the move can be undone exactly.

    Carries gpu_bytes so the engine can refuse to build when the card cannot
    hold the weights, instead of discovering it 25 seconds into loading.
    """

    def __init__(self, entries, gpu_bytes, fingerprint=None, spill=None):
        self.entries = entries
        self.gpu_bytes = gpu_bytes
        self.fingerprint = fingerprint
        self.spill = spill

    def __bool__(self):
        return bool(self.entries)

    @property
    def type(self):
        # park_trainer used to return a torch.device and release() checked
        # .type; keep that working for any caller still doing so.
        return "cuda"


def _force_park(model, fingerprint=None):
    """Move every GPU tensor of the model to the host, in place.

    Assignment is to .data, and to the quant_state attributes, rather than
    through Module.to() or Parameter.to(). Two reasons:

      * Parameter objects keep their identity, so optimizer state, the LoRA
        wiring and anything else holding a reference stays valid.
      * Nothing re-enters bitsandbytes' quantization path. The packed uint8
        .data is already quantized; a .to("cuda") that decided to quantize it
        again would corrupt the weights rather than fail.
    """
    entries = []
    host = torch.device("cpu")
    spill = None
    handle = None
    if PARK_TO_DISK:
        try:
            path = _park_dir() / f"trainer_park_{os.getpid()}_{uuid.uuid4().hex[:8]}.pt"
            handle = open(path, "wb")
            spill = _Spill(path)
        except Exception as exc:
            print(f"[vLLM] weights cannot be spilled to disk ({exc}); "
                  f"parking to host memory instead.", flush=True)
            spill = handle = None

    try:
        for store, key, tensor in _iter_model_tensors(model):
            if tensor.device.type != "cuda":
                continue
            origin = tensor.device
            try:
                staged = tensor.data.to(host)
                if handle is not None:
                    spill.write(handle, staged)
                    tensor.data = torch.empty(0, dtype=staged.dtype, device=host)
                    del staged
                else:
                    tensor.data = staged
            except Exception as exc:
                print(f"[vLLM] {key} could not be parked "
                      f"({type(exc).__name__}: {exc}); rolling back.", flush=True)
                if handle is not None:
                    handle.close()
                    handle = None
                _force_unpark(ParkedTrainer(entries, 0, spill=spill))
                if spill is not None:
                    spill.discard()
                return None
            _move_quant_state(tensor, host)
            entries.append((tensor, origin))
    finally:
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if spill is not None:
        gc.collect()
        print(f"[vLLM] {spill.count} weight tensor(s) spilled to disk "
              f"({spill.bytes / GIB:.2f} GiB); host now has "
              f"{_host_available_gib():.1f} GiB available, "
              f"{_host_dirty_gib():.2f} GiB awaiting writeback.", flush=True)
    return entries, spill


def _force_unpark(parked):
    """Put every parked tensor back on the device it came from."""
    spill = getattr(parked, "spill", None)
    handle = None
    if spill is not None and spill.count:
        try:
            handle = open(spill.path, "rb")
        except Exception as exc:
            print(f"[vLLM] the spilled weights cannot be read back "
                  f"({type(exc).__name__}: {exc}).", flush=True)
            raise
    try:
        for index, (tensor, origin) in enumerate(parked.entries):
            try:
                if handle is not None:
                    staged = spill.read(handle, index)
                    tensor.data = staged.to(origin)
                    del staged
                else:
                    tensor.data = tensor.data.to(origin)
            except Exception as exc:
                print(f"[vLLM] a parked tensor could not be restored to "
                      f"{origin} ({type(exc).__name__}: {exc}).", flush=True)
                raise
            _move_quant_state(tensor, origin)
    finally:
        if handle is not None:
            handle.close()
        if spill is not None:
            spill.discard()


def _host_rss_gib():
    """This process's resident set, in GiB, or None when it cannot be read."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _trim_host_arena() -> None:
    """Return memory freed inside this process to the operating system.

    Building the engine is a large host-side transaction even though the
    weights end up on the GPU: the safetensors shards are read into host
    memory and, with the bitsandbytes loader, quantised there before the
    copy. Python frees all of it when the build finishes, but glibc keeps
    the pages in its arena rather than handing them back, so the process's
    resident set ratchets up by that working set on every build and never
    comes down. Over a long run of cycles that is indistinguishable from a
    leak, and it is what pushed a 31 GiB machine to 3 GiB free by the
    seventh cycle -- after which the evaluation worker had nowhere to live.

    malloc_trim releases the arena's free pages. ab.gpt.util.Reward already
    does this for the same reason; the engine teardown simply never did.
    """
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if callable(malloc_trim):
            malloc_trim(0)
    except Exception:
        pass


def park_trainer(model):
    """Move the trainer off the GPU so the engine can have the card.

    Returns a ParkedTrainer on success and None when the card was not freed, so
    the caller neither restores a model that never moved nor builds an engine
    that cannot fit. Every exit logs, because a silent return here is what made
    this fail invisibly: the engine was then sized against 8 GiB of free memory
    for a 16 GiB model.
    """
    if model is None:
        print("[vLLM] prepare() was given no trainer model, so nothing was "
              "parked. If the trainer is resident the engine cannot fit; pass "
              "the model to prepare() or set NNGPT_VLLM_SLEEP=0.", flush=True)
        return None

    before, total = _free_gpu()
    resident = _model_gpu_bytes(model)
    if resident < MIN_PARK_FREED:
        print(f"[vLLM] Trainer holds only {resident / GIB:.2f} GiB on the card; "
              f"treating it as already parked ({before / GIB:.1f} GiB of "
              f"{total / GIB:.1f} GiB free).", flush=True)
        return None

    print(f"[vLLM] Parking trainer: {resident / GIB:.2f} GiB resident on the "
          f"card, {before / GIB:.1f} GiB of {total / GIB:.1f} GiB free.",
          flush=True)

    fingerprint = _fingerprint(model) if PARK_VERIFY else None

    spill = None
    if FORCE_PARK:
        parked = _force_park(model, fingerprint)
        if parked is None:
            return None
        entries, spill = parked
    else:
        # Legacy path. Module.to("cpu") leaves bitsandbytes quant_states on the
        # device and frees little; kept only as an escape hatch.
        try:
            model.to("cpu")
        except Exception as exc:
            print(f"[vLLM] Trainer could not be parked on the host ({exc}); "
                  f"the engine will share the card with it.", flush=True)
            return None
        entries = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after, _ = _free_gpu()
    freed = after - before
    still_resident = _model_gpu_bytes(model)

    if still_resident >= MIN_PARK_FREED:
        # The tensors did not leave. Either the move was refused or something
        # holds them on the device.
        print(f"[vLLM] Parking did not move the trainer off the card: "
              f"{still_resident / GIB:.2f} GiB is still resident "
              f"({after / GIB:.1f} GiB of {total / GIB:.1f} GiB free). The "
              f"engine needs about {resident / GIB:.1f} GiB of its own, so it "
              f"is not built; generation uses the HuggingFace path.", flush=True)
        if entries:
            _force_unpark(ParkedTrainer(entries, resident, fingerprint, spill))
        return None

    if freed < MIN_PARK_FREED:
        # The model's tensors did leave, but the driver still reports the memory
        # as taken, so this process's caching allocator is holding it. The engine
        # is a separate process and allocates from the driver, so it cannot use
        # pooled memory no matter how much this process has freed internally.
        # expandable_segments:True (which unsloth sets during training) is the
        # usual reason empty_cache() cannot hand the pages back.
        print(f"[vLLM] The trainer's tensors left the card "
              f"({still_resident / GIB:.2f} GiB resident) but the driver still "
              f"reports only {after / GIB:.1f} GiB of {total / GIB:.1f} GiB "
              f"free, so this process's allocator is holding about "
              f"{resident / GIB:.1f} GiB in its pool. The engine is a separate "
              f"process and cannot allocate from that pool. Try running the "
              f"trainer without expandable_segments "
              f"(PYTORCH_ALLOC_CONF without it), which lets empty_cache() "
              f"return the pages. Restoring the trainer and using the "
              f"HuggingFace path.", flush=True)
        if entries:
            _force_unpark(ParkedTrainer(entries, resident, fingerprint, spill))
        return None

    print(f"[vLLM] Trainer parked on the host; freed {freed / GIB:.1f} GiB "
          f"({after / GIB:.1f} GiB of {total / GIB:.1f} GiB free, "
          f"{still_resident / GIB:.2f} GiB left on the card).", flush=True)
    return ParkedTrainer(entries, resident, fingerprint, spill)


def unpark_trainer(model, parked):
    """Restore a parked trainer. Accepts a ParkedTrainer or a torch.device."""
    if model is None or parked is None:
        return
    if isinstance(parked, torch.device):
        # Legacy signature: park_trainer used to return the previous device.
        if parked.type == "cpu":
            return
        try:
            model.to(parked)
            gc.collect()
            torch.cuda.empty_cache()
            # The host copies of the weights have just been dropped; hand their
            # pages back rather than leaving them in this process's arena.
            _trim_host_arena()
            print(f"[vLLM] Trainer restored to {parked}.", flush=True)
        except Exception as exc:
            print(f"[vLLM] Could not restore the trainer to {parked}: {exc}",
                  flush=True)
            raise
        return

    if not parked.entries:
        try:
            model.to("cuda")
        except Exception as exc:
            print(f"[vLLM] Could not restore the trainer: {exc}", flush=True)
            raise
    else:
        _force_unpark(parked)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Restoring the weights to the card drops this process's host copies of
    # them -- about 7 GiB for a 9B model. Freeing them inside the process is
    # not enough: without the trim those pages stay in glibc's arena and the
    # resident set keeps whichever cycle's copy was largest, for every cycle.
    _trim_host_arena()
    free, total = _free_gpu()
    resident = _model_gpu_bytes(model)
    host_rss = _host_rss_gib()
    print(f"[vLLM] Trainer restored to the card ({resident / GIB:.2f} GiB "
          f"resident, {free / GIB:.1f} GiB of {total / GIB:.1f} GiB free"
          + (f", host {host_rss:.2f} GiB)." if host_rss is not None else ")."),
          flush=True)

    if parked.fingerprint is not None:
        now = _fingerprint(model)
        if now is None:
            pass
        elif now != parked.fingerprint:
            raise RuntimeError(
                "[vLLM] The trainer's weights changed across the park/restore "
                "round trip: a quantization state was probably left behind on "
                "the device. Training from here would use corrupted weights, "
                "so the run stops. Set NNGPT_VLLM_FORCE_PARK=0 to disable "
                "parking (the engine then cannot fit and generation falls back "
                "to the HuggingFace path).")
        else:
            print("[vLLM] Park/restore verified: weights unchanged.", flush=True)


def _unwrap_tokenizer(tokenizer_or_processor):
    """Return the real tokenizer behind a multimodal processor.

    Qwen3.x VL checkpoints load as Qwen3VLProcessor. Its __call__ takes images
    first and it treats its input as a batch, so it cannot be used as a
    tokenizer. Plain tokenizers have no .tokenizer attribute and pass through.
    """
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "convert_tokens_to_ids"):
        return inner
    return tokenizer_or_processor


def _flatten_token_ids(obj):
    """Coerce whatever a tokenizer or processor returned into a flat [int].

    A multimodal processor treats one conversation as a batch of one, so it
    hands back [[id, id, ...]]. vLLM then runs min(prompt_ids) over a list of
    lists and dies with "'<' not supported between instances of 'list' and
    'int'" -- and before that, len(ids) reads as 1, so the output budget is
    computed from a one-token prompt. Tensors, BatchEncoding and plain dicts
    all turn up here too, depending on the transformers version.
    """
    if obj is None:
        return []
    if hasattr(obj, "keys") and "input_ids" in obj:
        return _flatten_token_ids(obj["input_ids"])
    if hasattr(obj, "tolist") and not isinstance(obj, (list, tuple)):
        return _flatten_token_ids(obj.tolist())
    if isinstance(obj, int):
        return [obj]
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if all(isinstance(x, int) and not isinstance(x, bool) for x in obj):
            return list(obj)
        # Nested: one conversation rendered as a batch of one. Take the row.
        return _flatten_token_ids(obj[0])
    raise TypeError(f"cannot read token ids from {type(obj).__name__}")


def assistant_prefill():
    """Text the assistant turn is forced to begin with (NNGPT_ASSISTANT_PREFILL).

    A reasoning model expands to fill whatever budget it is given: at 1024
    tokens every reply was cut off mid-deliberation, and at 3072 all thirty
    still were. Telling it to be brief does not work, because the instruction
    is itself something to reason about. Starting the assistant turn mid-token
    removes the opportunity: there is no position at which prose could go.

    The prefix is prepended back onto the generated text, so everything
    downstream sees a complete reply and nothing else needs to know.
    """
    return os.environ.get("NNGPT_ASSISTANT_PREFILL", "")


def _render_messages(tok, messages, suppress_thinking=False):
    """Render a messages list to a prompt string.

    Two steps, text first and then encoding, rather than tokenize=True: a
    processor's tokenize=True return shape is batched and version-dependent,
    and rendering to text keeps this prompt identical to the one the
    HuggingFace path builds, so output from the two paths stays comparable.
    """
    apply = getattr(tok, "apply_chat_template", None)
    if callable(apply):
        # Qwen3.x is a hybrid thinking model with thinking ON by default. The
        # HuggingFace path asks for non-thinking mode via the template flag, so
        # this path must too: otherwise the same run suppresses reasoning
        # traces on one path and not the other, and the <think> block breaks
        # code extraction from the vLLM output.
        attempts = []
        if suppress_thinking:
            attempts.append({"enable_thinking": False})
            attempts.append({"chat_template_kwargs": {"enable_thinking": False}})
        attempts.append({})
        rendered = None
        for extra in attempts:
            try:
                rendered = apply(messages, tokenize=False,
                                 add_generation_prompt=True, **extra)
                break
            except TypeError:
                continue
            except Exception:
                rendered = None
                break
        try:
            if rendered is None:
                raise ValueError("apply_chat_template produced nothing")
            if not isinstance(rendered, str):
                # Batched processors return a list of strings.
                rendered = rendered[0] if rendered else ""
            if rendered:
                return rendered + assistant_prefill()
        except Exception as exc:
            print(f"[vLLM] apply_chat_template failed "
                  f"({type(exc).__name__}: {exc}); using the plain "
                  f"User/Assistant format.", flush=True)
    body = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                      for m in messages)
    return f"{body}\nAssistant:" + assistant_prefill()


def _encode_prompt(tok, messages, suppress_thinking=False):
    """Flat token ids for one conversation, processor or tokenizer alike."""
    rendered = _render_messages(tok, messages, suppress_thinking)
    try:
        encoded = tok(text=rendered, add_special_tokens=False)
    except TypeError:
        encoded = tok(rendered, add_special_tokens=False)
    ids = _flatten_token_ids(encoded)
    if not ids:
        raise ValueError("the prompt encoded to zero tokens")
    return ids


def _engine_window(max_model_len):
    """The sequence window the engine should use, which may be below the trainer's.

    KV cache is linear in max_model_len, and on a card where the weights eat
    most of the budget that single number decides whether anything can batch.
    The trainer's context_length is the wrong value to inherit: it has to cover
    the longest prompt plus a generous answer, while the engine only has to
    cover the longest prompt plus the answer actually produced. Measured output
    in this pipeline is ~700-1400 tokens, so reserving 6000 per sequence buys
    nothing and costs concurrency.

    NNGPT_VLLM_MAX_MODEL_LEN caps it; 0 or unset keeps the value passed in. The
    cap is never allowed to raise the window above what the caller asked for.
    """
    want = int(max_model_len)
    cap = _env_int("NNGPT_VLLM_MAX_MODEL_LEN", 0)
    if cap > 0 and cap < want:
        print(f"[vLLM] engine window capped to {cap} tokens "
              f"(trainer context is {want}); KV per sequence scales with this.",
              flush=True)
        return cap
    return want


def _speculative_config():
    """n-gram speculative decoding config, or None.

    Speculation multiplies single-stream throughput, which is the regime this
    card is stuck in: with the weights taking most of the budget there is KV
    for only a few sequences, so continuous batching cannot help and per-token
    speed is all that is left.

    The n-gram method is used rather than a draft model or MTP because it needs
    no extra weights: it proposes continuations by matching n-grams already in
    the prompt and the text generated so far. MTP would be stronger but does
    not fit in what is left of a 24 GiB card. Generated code is unusually
    repetitive (import blocks, nn.Conv2d(, self., indentation), which is the
    case n-gram speculation is good at; where it misses, the proposals are
    simply rejected and little is lost.

    OFF by default, because the trade depends on how much batching the card can
    do. Speculation wins when few sequences are in flight and the GPU sits idle
    waiting on a serial decode -- the 27B case. At high concurrency the GPU is
    already saturated and every rejected draft token is wasted compute, so on a
    model that actually fills max_num_seqs it can be a net loss. Set
    NNGPT_VLLM_SPEC_TOKENS=4 where it helps.
    """
    n = _env_int("NNGPT_VLLM_SPEC_TOKENS", 0)
    if n <= 0:
        return None
    return {
        "method": "ngram",
        "num_speculative_tokens": n,
        "prompt_lookup_max": _env_int("NNGPT_VLLM_SPEC_LOOKUP_MAX", 4),
        "prompt_lookup_min": _env_int("NNGPT_VLLM_SPEC_LOOKUP_MIN", 1),
    }


def _drop_unsupported_engine_kwargs(kwargs):
    """Remove engine kwargs this vLLM build does not accept.

    vLLM renames and retires EngineArgs fields between releases (swap_space was
    removed, and kv_cache_dtype / enforce_eager have moved before), which turns
    an upgrade into "EngineArgs.__init__() got an unexpected keyword argument".
    Introspect the dataclass and drop what it no longer knows, naming each one,
    rather than failing the whole engine build over an optional tuning flag.
    """
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return kwargs
    accepted = set()
    try:
        import dataclasses
        accepted = {f.name for f in dataclasses.fields(EngineArgs)}
    except Exception:
        pass
    if not accepted:
        try:
            import inspect
            accepted = set(inspect.signature(EngineArgs.__init__).parameters) - {"self"}
        except Exception:
            return kwargs
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"[vLLM] engine kwargs not supported by this vLLM build, "
              f"dropped: {', '.join(dropped)}", flush=True)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return kwargs


class VLLMSleepEngine:

    def __init__(self, base_model_name, tokenizer, max_model_len,
                 temperature=0.8, top_k=70, top_p=0.9,
                 gpu_memory_utilization=None, quantization=None,
                 max_lora_rank=None, enforce_eager=None, dtype="bfloat16",
                 served_model=None, max_num_seqs=None, trainer_reserve_gib=None,
                 kv_cache_dtype=None, required_gib=None, repetition_penalty=None, min_p=None):
        from vllm import LLM

        self.tokenizer = tokenizer
        # The engine is handed whatever the loader produced, which for a VL
        # checkpoint is a processor. Keep both: the processor for anything that
        # wants it, the unwrapped tokenizer for all text work.
        self._tok = _unwrap_tokenizer(tokenizer)
        if self._tok is not tokenizer:
            print(f"[vLLM] {type(tokenizer).__name__} detected; using its "
                  f"inner {type(self._tok).__name__} to encode prompts.",
                  flush=True)
        self.system_prompt = None
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_model_len = _engine_window(max_model_len)
        self.suppress_thinking = False
        self.budget_tokens = None

        self._adapter_path = None
        self._adapter_version = 0
        self._asleep = False

        served_model = served_model or os.getenv("NNGPT_VLLM_MODEL", "").strip()
        if served_model and served_model != base_model_name:
            print(f"[vLLM] Inference weights: {served_model} "
                  f"(trainer keeps {base_model_name})", flush=True)
        else:
            served_model = base_model_name

        # Opt-in (NNGPT_VLLM_INHERIT_GEN_CONFIG=1): apply what model.generate() inherits from the model's
        # generation_config.json, so that vLLM samples like the HF path. An explicit argument or
        # NNGPT_VLLM_REPETITION_PENALTY sets a value directly. With neither, SamplingParams are
        # built from temperature, top_k, top_p and the token budget only, as before.
        env_rp = os.getenv("NNGPT_VLLM_REPETITION_PENALTY", "").strip()
        if repetition_penalty is None and env_rp:
            repetition_penalty = float(env_rp)
        explicit = {"repetition_penalty": repetition_penalty, "min_p": min_p}
        self.sampling_extra = {}
        if _env_flag("NNGPT_VLLM_INHERIT_GEN_CONFIG", False):
            gen_cfg, gen_src = read_generation_config(served_model, base_model_name)
            self.sampling_extra, ignored = inherited_sampling_params(gen_cfg, explicit)
            print(f"[vLLM] Sampling: temperature={temperature} top_k={top_k} top_p={top_p} "
                  f"{' '.join(f'{k}={v}' for k, v in self.sampling_extra.items()) or '(no inherited fields)'} "
                  f"(generation_config: {gen_src or 'not found'})", flush=True)
            if ignored:
                print(f"[vLLM] WARNING: generation_config sets {ignored}, which SamplingParams cannot express; "
                      f"vLLM and HF generation differ in these.", flush=True)
        else:
            self.sampling_extra = {k: v for k, v in explicit.items() if v is not None}
            if self.sampling_extra:
                print(f"[vLLM] Sampling: temperature={temperature} top_k={top_k} top_p={top_p} "
                      f"{' '.join(f'{k}={v}' for k, v in self.sampling_extra.items())} (explicit)", flush=True)

        free, total = _free_gpu()
        reserve = _env_float("NNGPT_VLLM_TRAINER_RESERVE_GIB",
                             trainer_reserve_gib if trainer_reserve_gib is not None else 0.0)
        budget = min(free, total - reserve * GIB) - _env_float("NNGPT_VLLM_MARGIN_GIB", 1.0) * GIB
        if gpu_memory_utilization is None:
            gpu_memory_utilization = _env_float("NNGPT_VLLM_GPU_MEMORY_UTILIZATION", 0.0)
        if not gpu_memory_utilization:
            gpu_memory_utilization = max(0.10, min(0.95, budget / total)) if total else 0.90

        # Refuse before loading rather than after. The engine needs roughly the
        # same bytes the trainer held, and gpu_memory_utilization is derived
        # from *free* memory, so when the trainer is still resident this budget
        # is quietly smaller than the weights. Loading then dies about halfway
        # through with a bare CUDA OOM, 25 seconds later, having said nothing
        # about why. The headroom factor covers the loader's own working set and
        # the CUDA context; it is a heuristic, so NNGPT_VLLM_REQUIRED_GIB
        # overrides the estimate and 0 disables the check.
        required = _env_float("NNGPT_VLLM_REQUIRED_GIB",
                              required_gib if required_gib is not None else 0.0)
        if required > 0:
            headroom = _env_float("NNGPT_VLLM_REQUIRED_HEADROOM", 1.15)
            need = required * headroom * GIB
            have = gpu_memory_utilization * total
            if have < need:
                raise RuntimeError(
                    f"engine needs about {need / GIB:.1f} GiB (weights "
                    f"{required:.1f} GiB x {headroom:.2f} headroom) but its "
                    f"budget is {have / GIB:.1f} GiB "
                    f"(gpu_memory_utilization={gpu_memory_utilization:.3f} of "
                    f"{total / GIB:.1f} GiB, {free / GIB:.1f} GiB free). The "
                    f"trainer is most likely still on the card: parking it is "
                    f"what frees the room. Not loading weights that cannot fit.")

        default_quant = "" if served_model != base_model_name else "bitsandbytes"
        quantization = quantization or os.getenv(
            "NNGPT_VLLM_QUANTIZATION", default_quant).strip() or None
        if quantization and quantization.lower() in {"none", "off", "auto"}:
            quantization = None

        max_lora_rank = _env_int("NNGPT_VLLM_MAX_LORA_RANK",
                                 max_lora_rank or 32)
        enforce_eager = (enforce_eager if enforce_eager is not None
                         else _env_flag("NNGPT_VLLM_ENFORCE_EAGER", True))
        max_num_seqs = _env_int("NNGPT_VLLM_MAX_NUM_SEQS", max_num_seqs or 16)
        # The profiling pass vLLM runs straight after loading weights does a
        # forward of max_num_batched_tokens, and whatever that peaks at is
        # memory the KV cache cannot have. vLLM 0.28 raised the default from
        # 8192 to 16384, which is far too much here for two compounding
        # reasons: a 16k-token prefill on a 27B model is large by itself, and
        # bitsandbytes has no fused 4-bit gemm for prefill, so it dequantizes
        # each weight to bf16 first (a 17408x5120 matrix is ~178 MB
        # materialized, per layer, transiently). With ~6 GiB left after the
        # weights that overflows, which is why the OOM lands after
        # "Loading safetensors checkpoint shards: 100%" rather than during it.
        # Chunked prefill (on by default in V1) splits long prompts to fit, so
        # a smaller value costs throughput, not capability.
        # 0 / unset => do not pass it at all, so vLLM keeps its own default.
        # A small value is right only when the weights leave almost no KV: it
        # caps the profiling forward and the prefill chunk, which is what the
        # 27B needed. On a model with room to spare (a 14B keeps ~14 GiB of KV)
        # forcing 2048 makes prefill 8x more chunked and is a straight
        # throughput loss, so it must stay opt-in.
        max_num_batched_tokens = _env_int("NNGPT_VLLM_MAX_NUM_BATCHED_TOKENS", 0)
        # "fp8" halves the KV cache, roughly doubling how many sequences fit,
        # at the cost of a small quantization error in attention. Left at "auto"
        # by default because the generated code is the measured output.
        kv_cache_dtype = kv_cache_dtype or os.getenv(
            "NNGPT_VLLM_KV_CACHE_DTYPE", "auto").strip() or "auto"

        kwargs = dict(
            model=served_model,
            enable_sleep_mode=True,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            max_model_len=self.max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            swap_space=0,
            kv_cache_dtype=kv_cache_dtype,
        )
        if max_num_batched_tokens > 0:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        spec = _speculative_config()
        if spec:
            kwargs["speculative_config"] = spec

        if quantization:
            kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"

        print(f"[vLLM] Building engine: model={served_model} "
              f"max_model_len={self.max_model_len} max_num_seqs={max_num_seqs} "
              f"max_num_batched_tokens={max_num_batched_tokens or 'default'} "
              f"spec={spec['num_speculative_tokens'] if spec else 'off'} "
              f"quantization={quantization or 'from checkpoint'} "
              f"kv_cache_dtype={kv_cache_dtype} "
              f"gpu_memory_utilization={gpu_memory_utilization:.3f} "
              f"(free={free / GIB:.1f} GiB, total={total / GIB:.1f} GiB)", flush=True)
        kwargs = _drop_unsupported_engine_kwargs(kwargs)
        _saved_alloc = _strip_expandable_segments()
        try:
            self.llm = LLM(**kwargs)
        finally:
            _restore_alloc_conf(_saved_alloc)
        print("[vLLM] Engine ready: sleep mode on, LoRA hot-swap on.", flush=True)

    def sleep(self, level=1):
        if self._asleep:
            return
        try:
            self.llm.sleep(level=level)
            self._asleep = True
            gc.collect()
            torch.cuda.empty_cache()
            free, total = _free_gpu()
            print(f"[vLLM] Asleep (level={level}); {free / GIB:.1f} GiB of "
                  f"{total / GIB:.1f} GiB free for the trainer.", flush=True)
        except Exception as exc:
            print(f"[vLLM] sleep() failed, engine stays resident: {exc}", flush=True)

    def wake_up(self, adapter_path=None):
        if self._asleep:
            gc.collect()
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self._asleep = False
            free, total = _free_gpu()
            print(f"[vLLM] Awake; {free / GIB:.1f} GiB of {total / GIB:.1f} GiB "
                  f"still free.", flush=True)
        self.set_adapter(adapter_path)

    def set_adapter(self, adapter_path):
        if adapter_path is None:
            return
        path = Path(adapter_path)
        if not (path / "adapter_config.json").is_file():
            print(f"[vLLM] No adapter_config.json under {path}; generating with "
                  f"the base weights.", flush=True)
            return
        self._adapter_path = path
        self._adapter_version += 1
        print(f"[vLLM] Adapter v{self._adapter_version} <- {path}", flush=True)

    def shutdown(self):
        """Destroy the engine outright and give the whole card back.

        sleep() keeps the process able to generate again in milliseconds, but
        the engine still owns its CUDA context, its compiled graphs and the
        pool it reserved through gpu_memory_utilization -- most of the card.
        Evaluating the generated networks needs that memory, so between epochs
        the engine is destroyed rather than parked, and rebuilt when the next
        epoch generates. Never raises: a failure to tear down must not take
        the run with it.
        """
        before, total = _free_gpu()
        llm, self.llm = getattr(self, "llm", None), None
        engine = getattr(llm, "llm_engine", None)
        for owner in (getattr(engine, "engine_core", None), engine, llm):
            if owner is None:
                continue
            for name in ("shutdown", "close"):
                fn = getattr(owner, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        # owner and fn outlive the loop and the last owner IS llm, so deleting
        # only llm and engine leaves the engine strongly referenced from this
        # frame: the collect and empty_cache below would then run against a
        # live engine, and the "released" figure printed at the end would
        # describe a teardown that had not happened yet.
        owner = None
        fn = None
        del owner, fn
        del llm, engine
        # The next build initialises the model-parallel group again; leaving
        # the old one registered makes that build fail outright, so a torn-down
        # engine that was not unregistered here can never be replaced.
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment, destroy_model_parallel)
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        self._asleep = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        after, _ = _free_gpu()
        rss_before_trim = _host_rss_gib()
        _trim_host_arena()
        rss_after_trim = _host_rss_gib()
        print(f"[vLLM] Engine destroyed; {after / GIB:.1f} GiB of "
              f"{total / GIB:.1f} GiB free "
              f"(+{(after - before) / GIB:.1f} GiB released).", flush=True)
        if rss_before_trim is not None and rss_after_trim is not None:
            # Printed every teardown so a host-side ratchet is visible in the
            # log instead of having to be inferred from the machine running out.
            print(f"[vLLM] Host memory after teardown: "
                  f"{rss_after_trim:.2f} GiB resident "
                  f"(trim returned {max(0.0, rss_before_trim - rss_after_trim):.2f} GiB).",
                  flush=True)

    def _lora_request(self):
        if self._adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest(f"nngpt-v{self._adapter_version}",
                           self._adapter_version, str(self._adapter_path))

    def _build_messages(self, user_content):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _token_ids(self, prompt_text):
        return _encode_prompt(self._tok, self._build_messages(prompt_text),
                              self.suppress_thinking)

    def token_length(self, prompt_text):
        return len(self._token_ids(prompt_text))

    def output_budget(self, in_len, max_new_tokens):
        want = max_new_tokens or self.max_model_len
        room = self.max_model_len - in_len
        if room < MIN_NEW_TOKENS:
            return None
        return min(want, room)

    def chat_batch(self, prompts, max_len=None, max_new_tokens=None,
                   engineer_prompt=False):
        if not prompts:
            return []
        if self._asleep:
            self.wake_up()

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        requests, budgets, kept, results = [], [], [], {}
        for pos, prompt_text in enumerate(prompts):
            ids = self._token_ids(prompt_text)
            budget = self.output_budget(len(ids), max_new_tokens)
            if budget is None:
                print(f"[vLLM] SKIPPED prompt {pos}: input_tokens={len(ids)} "
                      f"max_model_len={self.max_model_len} leaves "
                      f"{self.max_model_len - len(ids)} for output.", flush=True)
                results[pos] = SKIPPED_OUTPUT
                continue
            requests.append(TokensPrompt(prompt_token_ids=ids))
            budgets.append(budget)
            kept.append(pos)

        if not kept:
            return [results[pos] for pos in range(len(prompts))]

        sampling = [SamplingParams(temperature=self.temperature,
                                   top_k=self.top_k, top_p=self.top_p,
                                   max_tokens=budget, **self.sampling_extra)
                    for budget in budgets]
        print(f"[vLLM] generate: {len(kept)} prompts in one continuous batch, "
              f"max_new_tokens={max(budgets)}", flush=True)
        outputs = self.llm.generate(requests, sampling,
                                    lora_request=self._lora_request())

        prefill = assistant_prefill()
        for pos, out in zip(kept, outputs):
            text = out.outputs[0].text if out.outputs else ""
            if prefill and text:
                text = prefill + text
            results[pos] = (*extract_all_to_train(text), text)
        return [results[pos] for pos in range(len(prompts))]

    def chat(self, prompt, max_len=None, max_new_tokens=None,
             engineer_prompt=False):
        return self.chat_batch([prompt], max_len=max_len,
                               max_new_tokens=max_new_tokens,
                               engineer_prompt=engineer_prompt)[0]


def build_engine(base_model_name, tokenizer, max_model_len, temperature=0.8,
                 top_k=70, top_p=0.9, trainer_reserve_gib=None,
                 required_gib=None):
    try:
        return VLLMSleepEngine(
            base_model_name, tokenizer, max_model_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
            trainer_reserve_gib=trainer_reserve_gib,
            required_gib=required_gib,
        )
    except Exception as exc:
        print(f"[vLLM] ENGINE BUILD FAILED: {exc}", flush=True)
        print("[vLLM] Generation falls back to the HuggingFace path, which is "
              "several times slower. Install a vllm build matching this torch, "
              "or set NNGPT_VLLM_SLEEP=0 to silence this.", flush=True)
        return None


class VLLMGeneration:

    def __init__(self, base_model_name, tokenizer, max_model_len,
                 temperature=0.8, top_k=70, top_p=0.9):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self._tok = _unwrap_tokenizer(tokenizer)
        # Measure prompts against the window the engine will really use, so the
        # fitting pass agrees with the engine even between epochs when the
        # engine has been torn down and cannot be asked.
        self.max_model_len = _engine_window(max_model_len)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.system_prompt = None
        self.suppress_thinking = False
        self.budget_tokens = None

        self.engine = None
        self.fallback = None
        self._build_failed = False
        self._parked_device = None

    @property
    def llm(self):
        if self.engine is None:
            raise AttributeError("vLLM engine is not built")
        return self.engine.llm

    @property
    def active(self):
        return self.engine is not None

    def _target(self):
        return self.engine if self.engine is not None else self.fallback

    def prepare(self, adapter_path, trainer_model):
        if self._build_failed:
            return
        self._parked_device = park_trainer(trainer_model)

        # A trainer that is still resident leaves far too little room, and the
        # engine would spend ~25s loading weights only to OOM. park_trainer has
        # already said why, so give up on vLLM for the rest of the run rather
        # than repeating that cost once per epoch. NNGPT_VLLM_BUILD_UNPARKED=1
        # forces the attempt anyway.
        if (self.engine is None and self._parked_device is None
                and trainer_model is not None
                and not _env_flag("NNGPT_VLLM_BUILD_UNPARKED", False)):
            resident = _model_gpu_bytes(trainer_model)
            _free, total = _free_gpu()
            # Refuse only when two copies genuinely cannot coexist. A small
            # model can share the card -- an 8 GiB trainer plus an 8 GiB engine
            # and its cache fits in 24 GiB -- and refusing there would disable
            # vLLM for a configuration that works. Only a model whose two
            # copies exceed the card is hopeless, and that is the 27B.
            if (resident >= MIN_PARK_FREED and total
                    and resident * 2 + MIN_PARK_FREED > total):
                print(f"[vLLM] The trainer is still on the card "
                      f"({resident / GIB:.1f} GiB) and the engine needs about "
                      f"as much again, which does not fit in "
                      f"{total / GIB:.1f} GiB. Generation uses the "
                      f"HuggingFace path for this run.", flush=True)
                self._build_failed = True
                return
            if resident >= MIN_PARK_FREED:
                print(f"[vLLM] The trainer did not park ({resident / GIB:.1f} "
                      f"GiB resident) but two copies should fit in "
                      f"{total / GIB:.1f} GiB; building the engine anyway.",
                      flush=True)

        required_gib = (self._parked_device.gpu_bytes / GIB
                        if self._parked_device is not None
                        and hasattr(self._parked_device, "gpu_bytes") else None)
        if self.engine is None:
            self.engine = build_engine(
                self.base_model_name, self.tokenizer, self.max_model_len,
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                required_gib=required_gib,
            )
            if self.engine is None:
                self._build_failed = True
                unpark_trainer(trainer_model, self._parked_device)
                self._parked_device = None
                return
            self.engine.set_adapter(adapter_path)
            return
        self.engine.wake_up(adapter_path)

    def release(self, trainer_model):
        """Called once generation for the epoch is finished.

        The evaluator trains every generated network on the same GPU, so the
        engine is destroyed here and rebuilt by the next prepare(). Set
        NNGPT_VLLM_TEARDOWN=0 to park it with sleep() instead, which wakes
        faster but leaves the reserved pool in place for the whole evaluation.
        """
        if self.engine is not None:
            if TEARDOWN_BETWEEN_EPOCHS:
                self.engine.shutdown()
                self.engine = None
            else:
                self.engine.sleep(level=1)
        if self._parked_device is not None:
            unpark_trainer(trainer_model, self._parked_device)
            self._parked_device = None

    def ensure_asleep(self):
        if self.engine is not None:
            self.engine.sleep(level=1)

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def token_length(self, prompt_text):
        """Prompt length in tokens, with or without a live engine.

        Between epochs the engine is torn down, but the prompt-fitting pass
        still has to measure prompts. Measuring needs the tokenizer only, so
        it is done here rather than delegated to an engine that may be gone --
        returning None instead would make every prompt look unmeasurable and
        the epoch would generate nothing.
        """
        if self.engine is not None:
            return self.engine.token_length(prompt_text)
        target = self.fallback
        if target is not None:
            rendered = target._templated_from_messages(
                target._build_messages(prompt_text))
            return len(_flatten_token_ids(
                self._tok(text=rendered, add_special_tokens=False)))
        return len(_encode_prompt(self._tok, self._build_messages(prompt_text),
                                  self.suppress_thinking))

    def output_budget(self, in_len, max_new_tokens):
        if self.engine is not None:
            return self.engine.output_budget(in_len, max_new_tokens)
        want = max_new_tokens or self.max_model_len
        room = self.max_model_len - in_len
        return None if room < MIN_NEW_TOKENS else min(want, room)

    def _build_messages(self, user_content):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, prompt, max_len=None, max_new_tokens=None, engineer_prompt=False):
        target = self._target()
        if target is None:
            return SKIPPED_OUTPUT
        target.system_prompt = self.system_prompt
        target.suppress_thinking = self.suppress_thinking
        return target.chat(prompt, max_len=max_len, max_new_tokens=max_new_tokens,
                           engineer_prompt=engineer_prompt)

    def chat_batch(self, prompts, max_len=None, max_new_tokens=None,
                   engineer_prompt=False):
        target = self._target()
        if target is None:
            return [SKIPPED_OUTPUT for _ in prompts]
        target.system_prompt = self.system_prompt
        target.suppress_thinking = self.suppress_thinking
        return target.chat_batch(prompts, max_len=max_len,
                                 max_new_tokens=max_new_tokens,
                                 engineer_prompt=engineer_prompt)
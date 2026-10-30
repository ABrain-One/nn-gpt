"""Does this generated model actually build and run one batch?

``static_validity`` answers "does the file have the right shape" by AST alone:
class Net, supported_hyperparameters, train_setup, learn, forward. Every one of
the 218 evaluation failures of one measured run passed that check. They died in
the constructor:

    if len(in_shape) == 4: _, _, C, H, W = in_shape
    ValueError: not enough values to unpack (expected 5, got 4)

The whole class of failure -- a wrong unpack, a channel count that does not
match, an undefined name in a branch the AST never evaluates, a torchvision
model name that does not exist, an in-place op the autograd graph still needs
-- is detectable by doing the thing: import the file, construct Net the way
``ab/nn/util/Train.py`` constructs it, push a random batch through it and take
one optimiser step. On CPU that is a second or two, against the ten minutes of
GPU time the same file would otherwise waste before failing in exactly the same
place.

Fail-open by design. A model is rejected only on a reason that is the model's
own: if the harness cannot run, times out, or fails on something
environmental (a backbone that has to be downloaded and cannot be), the
verdict is "unknown" and the model is kept. Wrongly discarding a good
generation costs a sample; wrongly keeping a bad one costs the training time
the pipeline was going to spend anyway.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

CHILD = Path(__file__).with_name("_smoke_child.py")

#: Filled in only for a standard key the record itself lacks. A declared name
#: that is not here and not in the record stays absent, which is what the
#: trainer does -- see the note in _smoke_child.py.
PRM_DEFAULTS = {"lr": 0.01, "momentum": 0.9, "dropout": 0.5, "batch": 8,
                "pretrained": 0.0}

#: Used only when nn-dataset cannot be asked what the trainer will really pass.
FALLBACK_IN_SHAPE = (3, 32, 32)
FALLBACK_OUT_SHAPE = (10,)

#: Samples per smoke batch. Two is enough to catch a batch-dependent reshape.
SMOKE_BATCH = max(2, int(os.environ.get("NNGPT_SMOKE_BATCH", "2") or 2))

#: (task, dataset, transform, transform_dir) -> ((C, H, W), out_shape) or None.
_SHAPE_CACHE: Dict[tuple, Optional[tuple]] = {}


#: Seconds the shape probe may take, including a first-time dataset download.
SHAPE_TIMEOUT = float(os.environ.get("NNGPT_SHAPE_TIMEOUT", "900") or 900)

#: Asks nn-dataset for the shapes and prints them as JSON, then exits. Reads
#: one sample rather than calling get_in_shape, which builds a DataLoader and
#: moves the tensor to torch_device() -- worker processes and a CUDA allocation
#: in a process that has neither to spare.
_SHAPE_PROBE = """
import json, sys
task, dataset, transform, transform_dir = sys.argv[1:5]
try:
    from ab.nn.util.Loader import load_dataset
    out_shape, _m, train_set, _t = load_dataset(
        task, dataset, transform, transform_dir or None)
    sample = train_set[0]
    items = sample if isinstance(sample, (tuple, list)) else [sample]
    shape = None
    for item in items:
        s = getattr(item, 'shape', None)
        if s is not None and len(s) >= 2:
            shape = [int(x) for x in s]
            break
    if shape is None:
        raise RuntimeError('no tensor in the first sample')
    print(json.dumps({'ok': True, 'in_shape': shape,
                      'out_shape': [int(x) for x in out_shape]}))
except Exception as exc:
    print(json.dumps({'ok': False,
                      'error': f'{type(exc).__name__}: {exc}'}))
"""


def _first_sample_shape(train_set):
    """The transformed shape of one sample, read on the CPU.

    Deliberately not ab.nn's get_in_shape, which is the obvious call and the
    wrong one here. It goes through first_tensor, which builds a DataLoader --
    spawning worker processes -- and then does ``.to(torch_device())``, which
    allocates on the GPU. This runs inside the generation process while the
    vLLM engine holds most of the card, and inside the evaluation process where
    CUDA is already initialised; the first gives an out-of-memory failure and
    the second a fork of an initialised CUDA context.

    Indexing the dataset once does the same job: the transform is applied, the
    tensor stays on the CPU, no loader and no workers exist, and the shape is
    deterministic whatever the transform's randomness does to the values.
    """
    try:
        sample = train_set[0]
    except Exception:
        return None
    items = sample if isinstance(sample, (tuple, list)) else [sample]
    for item in items:
        shape = getattr(item, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        return tuple(int(x) for x in shape)
    return None


def trainer_shapes(task, dataset, transform, transform_dir=None):
    """What the trainer will really hand the model, asked rather than assumed.

    The only fixed part of a generated file is the nn-dataset contract: a Net
    class, the method signatures, PyTorch. Everything else is the model's own
    business -- the input resolution it is built for included, which follows
    from the record's task, dataset and transform and from nothing in this
    repository.

    So the shapes come from nn-dataset's own loader, by the same two calls
    Train.py makes. Assuming them instead cost a whole cycle: the hardcoded
    (2, 3, 32, 32) tested CIFAR-10 at its native size while the records' own
    norm_256_flip resizes to 256 and centre-crops, so a model built for the
    256 input it would actually be trained on failed here on a shape the
    trainer never passes. out_shape was hardcoded to ten classes in the same
    spirit, which would have mis-tested every non-CIFAR task.

    Cached per configuration: the dataset is loaded once per cycle, not once
    per candidate. Returns None when nn-dataset cannot be reached, and the
    caller falls back to the constants above.
    """
    key = (str(task), str(dataset), str(transform), str(transform_dir or ""))
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]
    result = None
    try:
        # In a subprocess, because load_dataset holds CIFAR-sized arrays and
        # this runs in the generation process, where the parked trainer already
        # has 8 GiB and vLLM another 10. Loading a dataset there once per
        # configuration is what pushed a 31 GiB host to its floor and got the
        # run killed; a child returns the memory the moment it exits.
        available = _host_available_gib()
        if available and available < HOST_FLOOR_GIB:
            raise RuntimeError(f"host has only {available:.1f} GiB available")
        probe_env = dict(os.environ)
        probe_env["CUDA_VISIBLE_DEVICES"] = ""
        probe_env.setdefault("OMP_NUM_THREADS", "2")
        probe_env.setdefault("MKL_NUM_THREADS", "2")
        out = subprocess.run(
            [sys.executable, "-c", _SHAPE_PROBE, str(task), str(dataset),
             str(transform), str(transform_dir or "")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=SHAPE_TIMEOUT, text=True, env=probe_env)
        payload = json.loads((out.stdout or "").strip().splitlines()[-1])
        if payload.get("ok"):
            result = (tuple(payload["in_shape"]), tuple(payload["out_shape"]))
            print(f"[smoke] trainer shape for {dataset}/{transform}: "
                  f"{result[0]} -> {result[1]}", flush=True)
        else:
            raise RuntimeError(payload.get("error", "no shape"))
    except Exception as exc:
        print(f"[smoke] could not ask nn-dataset for the shape of "
              f"{task}/{dataset}/{transform} ({type(exc).__name__}: {exc}); "
              f"testing at {FALLBACK_IN_SHAPE} -> {FALLBACK_OUT_SHAPE}",
              flush=True)
    _SHAPE_CACHE[key] = result
    return result

#: Seconds one candidate may take. The batch is whatever the record's own
#: transform produces, so this has to cover a CPU forward and one step at that
#: resolution -- at 256 square that is some sixty times the work it was at 32.
#: A timeout is reported as "unknown", so the candidate is kept rather than
#: rejected on a guess.
TIMEOUT = float(os.environ.get("NNGPT_SMOKE_TIMEOUT", "900") or 900)

#: Reasons that are about the environment, not the model. Seen in a reason, the
#: verdict is downgraded from "broken" to "unknown".
ENVIRONMENTAL = (
    "urlopen", "urlerror", "httperror", "connection", "timed out", "timeout",
    "download", "certificate", "ssl", "no space left", "temporarily unavailable",
    "name resolution", "proxy", "out of memory", "oom", "disk",
    "host memory", "memoryerror",
)


def _is_environmental(reason: str) -> bool:
    low = (reason or "").lower()
    return any(tok in low for tok in ENVIRONMENTAL)


PRM_READ = re.compile(r"prm\s*\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]")
PRM_GET = re.compile(r"prm\s*\.\s*get\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")
PRM_SETDEFAULT = re.compile(
    r"prm\s*\.\s*setdefault\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")

#: Supplied by the trainer for every model out of the record's own prm, so a
#: file may both declare and read them. Neither half of the contract applies.
FRAMEWORK_PRM = frozenset({"epoch", "epoch_max", "transform", "num_workers"})


def check_prm_contract(code: str) -> Tuple[Optional[bool], str]:
    """Would this file raise KeyError on the prm the trainer hands it?

    One half of the contract is enforceable and one is not. The trainer builds
    prm from the declared names, so ``prm['k']`` for an undeclared k raises
    KeyError after the worker has started and the dataset has loaded: worth
    rejecting. Declaring a name without reading it costs nothing -- the key is
    supplied and ignored -- and PostprocessNN already strips those, so
    rejecting on it only throws away working models.

    What may read an undeclared name without raising:

    * ``prm.get('k', default)`` -- cannot raise, whatever is declared.
    * ``prm.setdefault('k', v)`` -- installs the key, so a later read of it
      finds a value. Order is not checked here; a read ahead of its own
      setdefault is left for the subprocess to catch, because rejecting on a
      static guess costs a whole generation and the subprocess costs seconds.
    * the framework names in FRAMEWORK_PRM.

    Static, so it runs before the subprocess and costs nothing.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"prm contract: file does not parse: {exc}"
    declared = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    try:
                        declared = set(str(k) for k in ast.literal_eval(stmt.value))
                    except Exception:
                        declared = None
                    break
    if declared is None:
        return None, "prm contract: supported_hyperparameters() is not a literal"
    indexed = set(PRM_READ.findall(code))
    defaulted = set(PRM_GET.findall(code)) | set(PRM_SETDEFAULT.findall(code))
    # The exemption comes off both sides. Taken off the reads alone it turned
    # every file that declares *and* reads epoch or transform -- which the
    # baselines do -- into a "declares without reading it" rejection.
    declared = set(declared) - FRAMEWORK_PRM
    indexed -= FRAMEWORK_PRM
    defaulted -= FRAMEWORK_PRM
    # Advisory from here on. A regex cannot see how a file reaches prm -- it
    # may splat it, iterate it, build a key, or install defaults through a
    # helper -- and a generated file is allowed to do any of that. Nothing but
    # the nn-dataset contract is fixed, so the only honest verdict on a key is
    # the one the subprocess reaches by running the constructor: a real
    # KeyError there costs seconds, while rejecting on a static guess costs
    # the generation. Measured on one cycle: 15 of 18 sound candidates thrown
    # away by this check.
    notes = []
    undeclared = sorted(indexed - declared - defaulted)
    if undeclared:
        notes.append("reads " + ", ".join(undeclared) + " without declaring it")
    unused = sorted(declared - indexed - defaulted)
    if unused:
        notes.append("declares " + ", ".join(unused) + " without reading it")
    if notes:
        return True, "prm contract: " + "; ".join(notes)
    return True, "prm contract ok"


#: Host memory one smoke child may hold. The test now runs at whatever the
#: record's transform produces -- 256 square for norm_256_flip, where a forward
#: and one backward cost some fifty times what they did at CIFAR's native 32 --
#: so a greedy candidate can take the host down with the trainer parked on it.
#: It fails its own smoke test instead, and the reason reads as environmental,
#: so the candidate is kept rather than rejected over a ceiling.
MEM_CEILING_GIB = float(os.environ.get("NNGPT_SMOKE_MEM_GIB", "4") or 4)

#: Host memory that must be free before a child is started at all. The repair
#: pass runs this while the vLLM engine is up and, from the second cycle on,
#: the fine-tuned trainer is resident too; measured on one run, generation
#: began with 6 to 7 GiB available. A ceiling alone cannot help there -- it was
#: set at 8 GiB, above the headroom, so the kernel reached the process first and
#: killed it with no traceback. The guarantee comes from the clamp below, not
#: from this floor: the child's ceiling is always set under the headroom that
#: actually exists, so it is stopped before the kernel intervenes. The floor
#: only has to leave room for a child worth starting -- below it the test is
#: skipped and the candidate kept unexamined, which costs an evaluation slot
#: rather than the run.
HOST_FLOOR_GIB = float(os.environ.get("NNGPT_SMOKE_HOST_FLOOR_GIB", "6") or 6)

#: Left for everything else while a child runs.
HOST_RESERVE_GIB = float(os.environ.get("NNGPT_SMOKE_HOST_RESERVE_GIB", "3") or 3)


def _host_available_gib() -> float:
    """MemAvailable, the kernel's own estimate of what can be handed out."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return 0.0


def _rss_gib(pid) -> float:
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return 0.0


def _run_bounded(cmd, cwd, env, timeout, ceiling=None):
    """Run the child, stopping it if it outgrows the ceiling."""
    ceiling = MEM_CEILING_GIB if ceiling is None else ceiling
    if ceiling <= 0:
        return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT), False
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, _ = proc.communicate(timeout=0.5)
            return subprocess.CompletedProcess(cmd, proc.returncode, out), False
        except subprocess.TimeoutExpired:
            pass
        if _rss_gib(proc.pid) > ceiling:
            proc.kill()
            proc.communicate()
            return subprocess.CompletedProcess(cmd, -9, b""), True
        if time.monotonic() > deadline:
            proc.kill()
            proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout)


def smoke_check(nn_file, prm=None, in_shape=None, out_shape=None,
                timeout: Optional[float] = None,
                check_learn: bool = True,
                light: bool = False) -> Tuple[Optional[bool], str]:
    """``(ok, reason)``. ``ok`` is None when the verdict is unknown.

    Runs in a subprocess so a segfault, a C-level abort or an infinite loop in
    generated code costs one candidate instead of the cycle.
    """
    if in_shape is None:
        in_shape = (SMOKE_BATCH,) + FALLBACK_IN_SHAPE
    if out_shape is None:
        out_shape = FALLBACK_OUT_SHAPE
    nn_path = Path(nn_file)
    if not nn_path.is_file():
        return None, "smoke: file not found"
    if not CHILD.is_file():                      # installed tree without it
        return None, "smoke: harness missing"

    args = {"prm": dict(prm or {}), "in_shape": list(in_shape),
            "out_shape": list(out_shape),
            "batch": 1 if light else (in_shape[0] if in_shape else 2),
            "check_learn": bool(check_learn) and not light,
            "light": bool(light), "defaults": dict(PRM_DEFAULTS)}
    # Nothing is started that the host cannot afford. The test runs at whatever
    # the record's transform produces -- 256 square for norm_256_flip -- so one
    # child can hold several GiB, and this is called once per candidate.
    available = _host_available_gib()
    if available and available < HOST_FLOOR_GIB:
        return None, (f"smoke: skipped, host has {available:.1f} GiB available "
                      f"and the floor is {HOST_FLOOR_GIB:.0f} GiB")
    ceiling = MEM_CEILING_GIB
    if available:
        ceiling = min(ceiling, max(1.0, available - HOST_RESERVE_GIB))

    env = dict(os.environ)
    # CPU only: the GPU is either held by the trainer or about to be, and a
    # smoke test that allocates on it is how a gate meant to prevent OOM causes
    # one. Threads are capped for the same reason -- 30 of these run in a row.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory(prefix="nngpt_smoke_") as tmp:
        args_path, out_path = Path(tmp) / "args.json", Path(tmp) / "out.json"
        args_path.write_text(json.dumps(args), encoding="utf-8")
        cmd = [sys.executable, "-I", str(CHILD), str(nn_path),
               str(args_path), str(out_path)]
        try:
            proc, overran = _run_bounded(cmd, tmp, env, timeout or TIMEOUT,
                                         ceiling=ceiling)
            if overran:
                return None, (f"smoke: stopped after exceeding "
                              f"{ceiling:.1f} GiB of host memory")
        except subprocess.TimeoutExpired:
            return None, f"smoke: timed out after {timeout or TIMEOUT:.0f}s"
        except Exception as exc:
            return None, f"smoke: could not run harness: {exc}"

        verdict = None
        if out_path.is_file():
            try:
                verdict = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                verdict = None
        if not isinstance(verdict, dict):
            tail = (proc.stdout or b"").decode("utf-8", "replace")[-500:].strip()
            return None, f"smoke: no verdict (rc={proc.returncode}) {tail}"

    if verdict.get("ok"):
        return True, "smoke_ok"
    reason = str(verdict.get("reason") or "smoke: unknown failure")
    if verdict.get("stage") == "harness":
        # The child could not even import torch: nothing was learned about
        # this model, so it keeps its slot.
        return None, f"smoke: harness unavailable ({reason})"
    if _is_environmental(reason):
        return None, f"smoke: environmental, kept ({reason})"
    return False, reason


TRANSFORM_CHILD = Path(__file__).with_name("_transform_child.py")

#: CIFAR-10's own statistics, so the pipeline is exercised with the shape of
#: argument the dataset really passes: two sequences, not six numbers.
PROBE_MEAN = (0.4914, 0.4822, 0.4465)
PROBE_STD = (0.2470, 0.2435, 0.2616)


def transform_check(code: str, size: int = 32,
                    timeout: Optional[float] = None) -> Tuple[Optional[bool], str]:
    """``(ok, reason)`` for a generated transform module. None is unknown.

    A generated <tr> block used to be written, installed into
    ab/nn/transform and used for a real training run without ever being
    executed. Measured consequence: a module calling
    Normalize(0.4914, 0.4822, 0.4465, 0.2470, 0.2435, 0.2616) -- six numbers
    where two sequences belong -- was installed and killed the evaluations
    that used it, with the traceback appearing only inside the worker.

    So it is built here and one image is pushed through it, which also catches
    a pipeline that forgets ToTensor and hands the trainer a PIL image.
    """
    if not code or not code.strip():
        return None, "transform: nothing to check"
    if not TRANSFORM_CHILD.is_file():
        return None, "transform: harness missing"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "2")
    with tempfile.TemporaryDirectory(prefix="nngpt_tr_") as tmp:
        tr_path = Path(tmp) / "subject.py"
        tr_path.write_text(code, encoding="utf-8")
        args_path, out_path = Path(tmp) / "args.json", Path(tmp) / "out.json"
        args_path.write_text(json.dumps(
            {"mean": list(PROBE_MEAN), "std": list(PROBE_STD), "size": size}),
            encoding="utf-8")
        cmd = [sys.executable, "-I", str(TRANSFORM_CHILD), str(tr_path),
               str(args_path), str(out_path)]
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, env=env, timeout=timeout or TIMEOUT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            return None, f"transform: timed out after {timeout or TIMEOUT:.0f}s"
        except Exception as exc:
            return None, f"transform: could not run harness: {exc}"
        verdict = None
        if out_path.is_file():
            try:
                verdict = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                verdict = None
        if not isinstance(verdict, dict):
            tail = (proc.stdout or b"").decode("utf-8", "replace")[-500:].strip()
            return None, f"transform: no verdict (rc={proc.returncode}) {tail}"
    if verdict.get("ok"):
        return True, "transform_ok"
    reason = str(verdict.get("reason") or "transform: unknown failure")
    if verdict.get("stage") == "harness" or _is_environmental(reason):
        return None, f"transform: not checked ({reason})"
    return False, reason


def smoke_check_dir(model_dir, **kwargs) -> Tuple[Optional[bool], str]:
    """Smoke-test the ``new_nn.py`` of one generation directory, with its hp.

    ``light=True`` builds the model and runs one forward without gradients, at
    batch 1. That is what "does it build" means, and it is what the repair pass
    needs: measured on one cycle, four of five failures were raised by the
    constructor or by train_setup, before a tensor was ever passed through.
    The backward pass is the expensive half -- activations at 256 square, held
    for the gradient -- and it belongs at the validity gate, which runs after
    the engine is torn down and has the memory for it.
    """
    d = Path(model_dir)
    nn_file = d / "new_nn.py"
    if not nn_file.is_file():
        cands = sorted(d.glob("*.py"))
        if not cands:
            return None, "smoke: no python file"
        nn_file = cands[0]
    prm = {}
    hp = d / "hp.txt"
    if hp.is_file():
        try:
            loaded = json.loads(hp.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prm = {k: v for k, v in loaded.items()
                       if isinstance(v, (int, float, str, bool))}
        except Exception:
            prm = {}
    # transform stays in prm. The trainer passes it, and a model is entitled
    # to read it; dropping it here made prm["transform"] raise in the gate and
    # nowhere else.
    try:
        code = nn_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        code = ""
    if code:
        ok, reason = check_prm_contract(code)
        if ok is False:
            return False, reason
    if "in_shape" not in kwargs or "out_shape" not in kwargs:
        task, dataset = _record_config(d)
        # No transform_dir: a generated transform is installed into the dataset
        # under a name derived from its own content, so the name identifies it
        # and plain lookup finds it. Passing the candidate's directory made the
        # cache key unique per candidate, and the dataset was then loaded once
        # per generation instead of once per configuration.
        shapes = trainer_shapes(task, dataset, prm.get("transform"))
        if shapes is not None:
            kwargs.setdefault("in_shape", (SMOKE_BATCH,) + tuple(shapes[0]))
            kwargs.setdefault("out_shape", tuple(shapes[1]))
    return smoke_check(nn_file, prm=prm, **kwargs)


#: Where the generation step records the baseline it was built from.
RECORD_FILE = "dataframe.df"


def _record_config(model_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """The task and dataset of the baseline this generation came from."""
    df = Path(model_dir) / RECORD_FILE
    if not df.is_file():
        return None, None
    try:
        import pandas as pd
        row = pd.read_pickle(df)
        return row.get("task"), row.get("dataset")
    except Exception:
        return None, None

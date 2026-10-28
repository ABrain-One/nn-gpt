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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

CHILD = Path(__file__).with_name("_smoke_child.py")

#: Seconds one candidate may take. A CPU forward plus one step on a CIFAR-sized
#: batch is seconds; anything near the limit is a model whose constructor is
#: doing something pathological, and the timeout is reported as "unknown" so it
#: is kept rather than rejected on a guess.
TIMEOUT = float(os.environ.get("NNGPT_SMOKE_TIMEOUT", "240") or 240)

#: Reasons that are about the environment, not the model. Seen in a reason, the
#: verdict is downgraded from "broken" to "unknown".
ENVIRONMENTAL = (
    "urlopen", "urlerror", "httperror", "connection", "timed out", "timeout",
    "download", "certificate", "ssl", "no space left", "temporarily unavailable",
    "name resolution", "proxy", "out of memory", "oom", "disk",
)


def _is_environmental(reason: str) -> bool:
    low = (reason or "").lower()
    return any(tok in low for tok in ENVIRONMENTAL)


def smoke_check(nn_file, prm=None, in_shape=(2, 3, 32, 32), out_shape=(10,),
                timeout: Optional[float] = None,
                check_learn: bool = True) -> Tuple[Optional[bool], str]:
    """``(ok, reason)``. ``ok`` is None when the verdict is unknown.

    Runs in a subprocess so a segfault, a C-level abort or an infinite loop in
    generated code costs one candidate instead of the cycle.
    """
    nn_path = Path(nn_file)
    if not nn_path.is_file():
        return None, "smoke: file not found"
    if not CHILD.is_file():                      # installed tree without it
        return None, "smoke: harness missing"

    args = {"prm": dict(prm or {}), "in_shape": list(in_shape),
            "out_shape": list(out_shape), "batch": in_shape[0] if in_shape else 2,
            "check_learn": bool(check_learn)}
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
            proc = subprocess.run(
                cmd, cwd=tmp, env=env, timeout=timeout or TIMEOUT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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


def smoke_check_dir(model_dir, **kwargs) -> Tuple[Optional[bool], str]:
    """Smoke-test the ``new_nn.py`` of one generation directory, with its hp."""
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
    prm.pop("transform", None)
    return smoke_check(nn_file, prm=prm, **kwargs)

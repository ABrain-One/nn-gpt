"""CPU pre-flight: reject candidates that cannot build or forward, BEFORE spending any GPU.

THE NUMBER THAT JUSTIFIES THIS. In the first screen of the 516 LLM candidates, ~70% failed
at build or first forward -- each after being scheduled onto a GPU, loading the dataset and
starting an epoch, at roughly 80 s a piece. That is ~8 GPU-hours spent discovering that
tensor shapes do not line up, which a CPU forward pass on one random 3x256x256 tensor
answers in well under a second.

This does NOT judge architecture and deliberately cannot: it runs no training, computes no
PSNR, and passes anything that builds, forwards to the right shape, and stays finite.
Everything it passes still goes through the real 1-epoch screen. It only removes candidates
that were never going to produce a number at all.

WHAT IT CHECKS, cheapest first, stopping at the first failure:
  parse      -> ast.parse
  static     -> the contract violations that are decidable without running anything
                (uses F. without importing it, indexes out_shape, no class Net, ...)
  build      -> Net(in_shape, out_shape, prm, device='cpu')
  forward    -> y = net(x) on a real tensor; shape must equal x, values must be finite
  train_setup-> must produce self.optimizer; self.scheduler only if learn() references it

NON-DETERMINISM CAVEAT: a model can pass at 256x256 and still fail at another resolution, and
some failures are batch-dependent. A pass here is "not obviously dead", never "will train".

MAC COST IS CHECKED HERE, and it has to be. The GA enforces the 9053 MMac edge budget on
both its seeds and its children; the LLM path enforced it NOWHERE -- screen_and_train.py has
no MAC check at all -- so LLM models entered the DB unverified. A measured audit of the first
9 trained found 2 over cap (10787 and 16968 MMac, up to 87% over), one of which had already
been selected into the evolve loop's elite pool and would have propagated an illegal
architecture into every later generation.

The prompt states the budget, but a prompt is a request, not a constraint. MACs are counted
with edge_cost._mac_hooks -- the GA's own counter -- so the numbers are comparable to the
GA's rather than a second opinion from a different profiler.

  python preflight.py --run-dirs .../cold/run .../seeded/run --out /a/mm/sweep_out/llm_pre
"""
import argparse
import ast
import builtins
import glob
import json
import os
import re
import sys
import time
import traceback
from collections import Counter

IN_SHAPE = (1, 3, 256, 256)
MAC_CAP_MMAC = 9053.0        # the edge budget; 0 disables the check
# Per-candidate caps. Sized so a legal model (<= 9053 MMac, a few Mparams) is nowhere near
# them, while a runaway allocation fails as ITS OWN result instead of OOM-killing the pod.
MEM_CAP_BYTES = 24 << 30     # floor for the per-candidate address-space cap
DEFAULT_TIMEOUT = 360        # WALL-CLOCK BACKSTOP only -- catches hangs, not slowness
DEFAULT_CPU_LIMIT = 60       # the real budget: CPU seconds (user+sys) per candidate
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 50, "transform": "denoise"}


# The single most common way an LLM candidate dies, by a wide margin. Measured over
# generation 1's 105 preflight failures:
#     33  NameError: name 'optim' is not defined     (no `import torch.optim as optim`)
#      3  uses F. but never imports torch.nn.functional as F
# 34% of all rejections, and not one of them is a modelling mistake -- the architecture is
# fine, the file just lacks a line of boilerplate. Throwing those away costs ~36 models per
# generation, more than the entire surviving yield of 22.
#
# Repairing is safe in a way that repairing logic would not be: each entry maps an alias to
# its ONE canonical import, and the repair only fires when the alias is used as
# `alias.something` and is bound nowhere in the file -- not imported, assigned, defined as a
# function/class, or taken as a parameter. Under that condition the name can only be the
# standard module, because any other meaning would already have failed with the same
# NameError.
STD_IMPORTS = {
    "torch": "import torch",
    "nn": "import torch.nn as nn",
    "F": "import torch.nn.functional as F",
    "optim": "import torch.optim as optim",
    "math": "import math",
}


def repair_imports(src):
    """-> (src, [added]). Prepend canonical imports for used-but-unbound module aliases."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, []          # a file that does not parse is not ours to fix
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                bound.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for al in n.names:
                bound.add(al.asname or al.name)
        elif isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    # Only `alias.attr` counts as use of a MODULE. A bare `optim` would be too weak a signal.
    used = {n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    add = sorted(STD_IMPORTS[k] for k in STD_IMPORTS if k in used and k not in bound)
    if not add:
        return src, []
    return "\n".join(add) + "\n" + src, add


def _class_defs(src):
    """{name: source_segment} for every top-level class. Only classes -- splicing a
    free function would be a different risk profile and is not needed."""
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            seg = ast.get_source_segment(src, n)
            if seg:
                out[n.name] = seg
    return out


def reference_classes(cand_dir):
    """Class definitions from the REFERENCE code in this candidate's own prompt.

    The prompt is written to prompt_user.txt before the model is called, and the references
    are embedded in it as ```python fences. Recovering the definitions from there means the
    repair uses exactly what this candidate was shown -- not a global pool, so it cannot
    import a class the model never saw and cannot have meant.
    """
    p = os.path.join(cand_dir, "prompt_user.txt")
    if not os.path.exists(p):
        return {}
    try:
        txt = open(p).read()
    except OSError:
        return {}
    defs = {}
    for block in re.findall(r"```(?:python)?\n(.*?)```", txt, re.S):
        defs.update(_class_defs(block))
    return defs


def repair_missing_classes(src, refs):
    """-> (src, [spliced]). Splice in helper classes the candidate uses but never defines.

    THE SINGLE LARGEST REPAIRABLE FAILURE MODE. 304 of 2364 candidates (12.9%) died at build
    with NameError on a helper class, and the missing names were not arbitrary -- they were
    'SPBlock', 'ELUAvgBlock', 'DilBlock', 'LPPoolBlock', 'HaarDown', i.e. the helper classes
    of the four reference models that every prompt carried. Another 235 (9.9%) had the
    mirror-image defect: they DEFINED one of those classes and never called it, because the
    fragment that survived was the definition rather than the call site.

    Both are the same failure -- the model reassembled a file out of reference fragments and
    dropped one half -- and in the NameError direction the missing half is sitting in the
    candidate's own prompt verbatim. Splicing it back is deterministic and changes no
    architectural decision the model made: the class it asked for is the class it gets.

    Only fires for a name that is USED as a call, is bound NOWHERE in the file, and is
    defined in this candidate's references. Under those three conditions the name has no
    other possible referent.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, []
    bound = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                bound.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for al in n.names:
                bound.add(al.asname or al.name)
        elif isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    missing = sorted(n for n in called if n not in bound and n in refs)
    if not missing:
        return src, []

    # A spliced class can itself call another helper. Resolve transitively, bounded, so a
    # chain like SPBlock -> ConvBlock does not trade one NameError for the next.
    spliced, queue, seen = [], list(missing), set(missing)
    while queue and len(spliced) < 12:
        name = queue.pop(0)
        seg = refs.get(name)
        if not seg:
            continue
        spliced.append(seg)
        try:
            sub_tree = ast.parse(seg)
        except SyntaxError:
            continue
        for n in ast.walk(sub_tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in refs and n.func.id not in seen
                    and n.func.id not in bound):
                seen.add(n.func.id)
                queue.append(n.func.id)
    if not spliced:
        return src, []

    # INSERT AFTER THE IMPORTS, not at the top of the file. A class statement evaluates its
    # bases at definition time, so a spliced `class SPBlock(nn.Module)` sitting above
    # `import torch.nn as nn` raises NameError: nn -- trading the candidate's missing class
    # for a missing module and repairing nothing. Measured: all 108 splices on generations
    # 10-16 still failed to exec until the insertion point moved below the imports.
    lines = src.splitlines(keepends=True)
    at = 0
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            at = max(at, getattr(n, "end_lineno", n.lineno))
    body = "\n\n" + "\n\n".join(spliced) + "\n\n"
    return "".join(lines[:at]) + body + "".join(lines[at:]), sorted(seen)


def strip_dead_classes(src):
    """-> (src, [removed]). Delete nn.Module classes that are defined and never used.

    STRIPPING BEATS REJECTING, which is what --reject-dead-code did. The flag exists because
    dead classes made six LLM models look like six architectures when they were one, and
    because this corpus becomes fine-tuning data where dead code teaches the next model to
    emit dead code. Removing the class serves both goals -- the file is clean AND it now
    hashes to its true architecture, so the diversity gate catches the duplicates it was
    meant to catch. Rejecting served them too, but at a cost of 235 candidates, and it threw
    away the genuinely novel ones along with the decorated copies.

    Iterated, because removing one dead class can orphan another that only it referenced.
    """
    removed = []
    for _ in range(6):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            break
        dead = None
        for n in tree.body:
            if not isinstance(n, ast.ClassDef) or n.name == "Net":
                continue
            if not any(isinstance(b, ast.Attribute) and b.attr == "Module"
                       for b in n.bases):
                continue
            uses = sum(1 for m in ast.walk(tree)
                       if isinstance(m, ast.Name) and m.id == n.name)
            if uses == 0:                       # ClassDef.name is not an ast.Name node
                dead = n
                break
        if dead is None:
            break
        seg = ast.get_source_segment(src, dead)
        if not seg or seg not in src:
            break
        src = src.replace(seg, "", 1)
        removed.append(dead.name)
    return src, removed


def static_issues(src):
    """Contract violations decidable without executing anything."""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]

    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    alias = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                alias.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                alias.add(a.asname or a.name)

    if "Net" not in names:
        out.append("no class named Net")
    if "supported_hyperparameters" not in names:
        out.append("no supported_hyperparameters()")
    for m in ("__init__", "train_setup", "forward", "learn"):
        if m not in names:
            out.append(f"missing method {m}")
    if re.search(r"\bF\.\w", src) and "F" not in alias:
        out.append("uses F. but never imports torch.nn.functional as F")
    if re.search(r"\bmath\.\w", src) and "math" not in alias:
        out.append("uses math. but never imports math")
    if re.search(r"out_shape\s*\[", src):
        out.append("indexes out_shape (harness may pass a 1-element tuple)")
    if re.search(r"self\.scheduler\b", src) and not re.search(r"self\.scheduler\s*=", src):
        out.append("references self.scheduler but never assigns it")
    if "torchvision" in src or "timm" in src:
        out.append("uses torchvision/timm (forbidden)")
    # DEAD MODULE CLASSES. Reported as a warning, never a rejection: the model still trains
    # and scores honestly, and rejecting on it would discard a working architecture over a
    # cosmetic defect. But it is the fingerprint of a candidate that LOOKS novel and is not,
    # so it belongs in the record.
    for cls in re.findall(r"^class (\w+)\s*\(\s*nn\.Module", src, re.M):
        if cls == "Net":
            continue
        body = src.replace(f"class {cls}", "")
        if not re.search(r"[^\w]" + re.escape(cls) + r"\s*\(", body):
            out.append(f"note: defines nn.Module '{cls}' but never instantiates it")
    return out


def _worker(src, strict_static, mac_cap, conn, cpu_limit=0):
    """Run the real checks in a CHILD process under a hard address-space cap.

    Isolation is not optional here. The first preflight run was OOMKilled at 200/516: some
    candidates ignore the MAC budget entirely and allocate more than the pod has, and one
    such model took down the whole pass, losing the 199 verdicts already computed. A model
    can also hang (a pathological forward on CPU), which no memory cap would catch.

    In a child, both become that candidate's own result: MemoryError -> "build/too large",
    wall-clock overrun -> the parent terminates it and records "timeout". Neither can harm
    the other 515.
    """
    # RLIMIT_AS caps VIRTUAL address space, not resident memory, and PyTorch reserves a very
    # large VA region at import. An 8 GiB cap therefore did not stop big models -- it stopped
    # THREAD CREATION (each thread stack is VA), and the pass rate collapsed from ~29% to 3%
    # as ordinary models died on "can't start new thread". The cap has to sit well above
    # torch's baseline reservation to mean "this model is too big" rather than "torch cannot
    # run"; measure the baseline once and refuse to set a cap below it.
    try:
        import resource
        base = _vsize_bytes()
        cap = max(MEM_CAP_BYTES, int(base * 1.5) + (2 << 30)) if base else MEM_CAP_BYTES
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass

    # CPU TIME, NOT WALL CLOCK. The wall-clock limit measured cluster load as much as
    # candidate cost: while an unrelated 32-shard CPU job saturated the nodes, generation 3
    # of the LLM run lost 117 of 144 candidates to "exceeded 60s" -- not one of them was
    # rejected on merit, and the generation yielded 1 model instead of ~13. A candidate's
    # verdict must not depend on what else happens to be running.
    #
    # RLIMIT_CPU counts user+sys seconds actually consumed, so contention cannot trigger it.
    # Soft limit raises SIGXCPU, which we catch to report a clean, attributable verdict;
    # the hard limit a few seconds later is the backstop if the handler cannot run.
    #
    # Threads are pinned to 1 so the limit means the same thing everywhere: RLIMIT_CPU sums
    # across threads, so an 8-thread forward burns 8 CPU-seconds per wall-second and the
    # budget would depend on how many cores the pod happened to get.
    if cpu_limit:
        try:
            import signal
            import resource

            def _xcpu(_sig, _frm):
                try:
                    conn.send((False, "cpu-timeout",
                               f"exceeded {cpu_limit}s of CPU time (not wall clock)", None))
                    conn.close()
                except Exception:
                    pass
                os._exit(0)

            signal.signal(signal.SIGXCPU, _xcpu)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
        except Exception:
            pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        res = check_one(src, strict_static, mac_cap)
        if len(res) == 3:
            res = res + (None,)
    except MemoryError:
        res = (False, "build", "MemoryError: model exceeds the address-space cap")
    except Exception as e:
        res = (False, "preflight-crash", f"{type(e).__name__}: {e}"[:200])
    try:
        conn.send(res)
    except Exception:
        pass
    finally:
        conn.close()


def _vsize_bytes():
    """Current virtual size from /proc/self/statm (pages)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def check_isolated(src, strict_static, timeout, mac_cap=MAC_CAP_MMAC, cpu_limit=0):
    """check_one in a subprocess; never lets one candidate kill the run.

    TWO limits, because neither alone is sufficient:
      cpu_limit   RLIMIT_CPU in the child -- the real budget, immune to what else runs.
      timeout     wall-clock backstop in the parent. A process blocked on I/O or deadlocked
                  burns NO cpu time and would sit under RLIMIT_CPU forever, so the wall
                  clock still has to exist -- just generously, as a hang detector rather
                  than a cost measure.
    They produce DIFFERENT verdicts on purpose: "cpu-timeout" means this candidate is
    genuinely expensive, "timeout" means it stopped making progress. Collapsing them is how
    117 healthy candidates got recorded as too slow.
    """
    import multiprocessing as mp
    ctx = mp.get_context("fork")     # torch is already imported; fork is ~free
    # Pipe, NOT Queue: Queue.put() starts a feeder THREAD, so under a tight address-space
    # limit even the error path failed with "can't start new thread" and the result was lost.
    # Connection.send() writes straight to the fd.
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker,
                    args=(src, strict_static, mac_cap, child, cpu_limit))
    p.start()
    child.close()                    # only the child keeps the write end
    wall = max(timeout, cpu_limit * 6) if cpu_limit else timeout
    got = None
    if parent.poll(wall):
        try:
            got = parent.recv()
        except EOFError:
            got = None
    parent.close()
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
        p.join(5)
        if got is None:
            return (False, "timeout",
                    f"no progress for {wall:.0f}s wall clock "
                    f"(cpu budget {cpu_limit}s was not exhausted -- hung, not slow)")
    else:
        p.join(5)
    if got is not None:
        return got
    # child died without posting: OOM killer or a segfault in a native op. Report it rather
    # than silently counting it as a pass.
    return False, "crash", f"child exited rc={p.exitcode} without a result"


_CH_RE = re.compile(
    r"Given groups=\d+, weight of size \[(\d+), (\d+), (\d+), (\d+)\], "
    r"expected input\[\d+, (\d+),")


def repair_channels(src, err, tries_left):
    """-> (src, note) or (None, None). Fix ONE conv whose in_channels disagrees with its input.

    THE LARGEST REMAINING LOSS, and it is not an architectural judgement. 55 of 138 candidates
    in the last generation failed at the first forward pass, and 31 of those were a single
    error shape:

        Given groups=1, weight of size [64, 32, 3, 3], expected input[1, 48, 64, 64]

    A Conv2d declares 32 input channels and receives 48. The model's DESIGN -- which layers,
    what topology, where the skips go -- is fine; one integer is inconsistent with the tensor
    actually arriving. Throwing the whole candidate away over that discards a working
    architecture for an arithmetic slip, exactly as the missing-import and missing-helper-class
    cases did before they were repaired.

    Repairing is narrow by construction. The error message states both the declared shape and
    the real one, so the correct value is known rather than guessed; only nn.Conv2d/
    ConvTranspose2d constructors whose in_channels literal matches the declared value AND whose
    out_channels matches are touched, and the substitution is applied once per attempt so a
    later mismatch surfaces its own message instead of being masked.

    What it does NOT do: invent channel counts, resize weights, or alter kernel size, stride or
    topology. If the message does not name both shapes, nothing happens.
    """
    m = _CH_RE.search(err or "")
    if not m or tries_left <= 0:
        return None, None
    out_c, want_in, kh, kw = (int(m.group(i)) for i in (1, 2, 3, 4))
    got_in = int(m.group(5))
    if want_in == got_in:
        return None, None
    # Match the constructor by BOTH channel literals: `out_c, want_in` is far less ambiguous
    # than in_channels alone, which repeats throughout a U-Net.
    pat = re.compile(
        r"(nn\.Conv(?:Transpose)?2d\(\s*)" + str(want_in) + r"(\s*,\s*)" + str(out_c) + r"\b")
    new, n = pat.subn(lambda mm: f"{mm.group(1)}{got_in}{mm.group(2)}{out_c}", src, count=1)
    if n == 0:
        return None, None
    return new, f"conv in_channels {want_in}->{got_in} (out={out_c}, k={kh}x{kw})"


def check_one(src, strict_static=True, mac_cap=MAC_CAP_MMAC):
    """Return (ok: bool, stage: str, detail: str)."""
    all_iss = static_issues(src)
    # "note:" findings are recorded, never fatal. Dead nn.Module classes are the fingerprint
    # of a decorated copy, but a candidate can carry dead code AND a novel architecture --
    # rejecting on it would discard working models for a cosmetic defect. Duplicates are
    # caught by the architectural id instead, which is the mechanism that actually means it.
    notes = [i for i in all_iss if i.startswith("note:")]
    iss = [i for i in all_iss if not i.startswith("note:")]
    if iss and (strict_static or any("syntax" in i or "no class" in i for i in iss)):
        return False, "static", "; ".join(iss)

    import torch
    ns = {}
    try:
        exec(compile(src, "<cand>", "exec"), ns)
    except Exception as e:
        return False, "exec", f"{type(e).__name__}: {e}"
    Net = ns.get("Net")
    if Net is None:
        return False, "exec", "no Net after exec"

    try:
        net = Net(IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")
    except Exception as e:
        return False, "build", f"{type(e).__name__}: {e}"

    # train_setup may already have run inside __init__; calling it twice must be harmless,
    # and if it was never called we still need the optimizer to exist.
    try:
        if hasattr(net, "train_setup"):
            net.train_setup(dict(PRM))
    except Exception as e:
        return False, "train_setup", f"{type(e).__name__}: {e}"
    if not hasattr(net, "optimizer"):
        return False, "train_setup", "no self.optimizer after train_setup"
    if re.search(r"self\.scheduler\b", src) and not hasattr(net, "scheduler"):
        return False, "train_setup", "learn() uses self.scheduler but it was never assigned"

    try:
        net.eval()
        x = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            y = net(x)
    except Exception as e:
        return False, "forward", f"{type(e).__name__}: {e}"
    if not torch.is_tensor(y):
        return False, "forward", f"forward returned {type(y).__name__}, not a tensor"
    if tuple(y.shape) != tuple(x.shape):
        return False, "forward", f"output shape {tuple(y.shape)} != input {tuple(x.shape)}"
    if not torch.isfinite(y).all():
        return False, "forward", "forward produced NaN/Inf on random input"

    if mac_cap and mac_cap > 0:
        try:
            from edge_cost import _mac_hooks
            handles, macs = _mac_hooks(net)
            try:
                with torch.no_grad():
                    net(torch.rand(1, 3, 256, 256))
            finally:
                for h in handles:
                    h.remove()
            mm = macs["v"] / 1e6
            if mm > mac_cap:
                return False, "mac", f"{mm:.0f} MMac > {mac_cap:.0f} cap"
        except Exception as e:
            # Unmeasurable is NOT legal. Failing open here is exactly how the over-cap
            # models got in; report it and let the caller decide.
            return False, "mac", f"unmeasurable ({type(e).__name__}: {e})"[:120]

    # A denoiser whose output leaves [0,1] was told to clamp; treat as a warning, not a
    # rejection -- it costs PSNR but it does train, and the screen will price it correctly.
    warn = "; ".join(notes)
    if float(y.min()) < -0.01 or float(y.max()) > 1.01:
        warn = ("; ".join(notes + [f"output range [{float(y.min()):.2f}, "
                                   f"{float(y.max()):.2f}] not clamped to [0,1]"]))

    # ARCHITECTURAL ID, computed here because `net` is already built and traced-ready.
    # This is the same WL graph hash the GA's diversity_gate uses and the same one the
    # professor proposes as the dataset UID: torch.fx trace -> Weisfeiler-Lehman over
    # behavioural node labels, so it is blind to naming, comments, ordering -- and to DEAD
    # CODE, which is precisely how the LLM manufactures fake diversity. Six LLM models in
    # generation 1 were one architecture; five of them differed only by defining a HaarDown
    # wavelet class they never instantiated, one by defining five such classes.
    # exact_source(src), NOT exact(net). They are not interchangeable: exact_source builds
    # the model through mutator.lemur_io.load_lemur_model, which instantiates with different
    # arguments than the (1,3,256,256) net preflight builds, so the two produce DIFFERENT
    # hashes for the SAME architecture. Using exact(net) made the cross-DB check compare
    # incomparable values and report "0 duplicates already in the dataset" for candidates
    # whose architectures were demonstrably already there -- a silent no-op that looked like
    # a clean result. The dataset's arch_uid column is written from exact_source, so this
    # must match it.
    # ONE canonical construction, shared with backfill_arch_uid.py so the values are
    # comparable. SWITCHED (2026-08-03) to arch_uid2 -- the source-only AST identity
    # (v2: per-class attribute scoping, slice ports, chained receivers). It hashes the
    # SOURCE TEXT through one pure function on both the candidate and the DB side, which
    # removes the exact()/exact_source() instantiation-mismatch class entirely, and it
    # identifies models torch.fx cannot trace (complex FFT, data-dependent control flow).
    arch = None
    try:
        from arch_uid2 import arch_uid as _uid2
        arch = _uid2(src)
    except Exception as e:
        arch = None
        warn = "; ".join([w for w in (warn, f"note: no arch id ({type(e).__name__})") if w])
    return True, "ok", warn, arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="/a/mm/sweep_out/llm_pre")
    ap.add_argument("--mac-cap", type=float, default=MAC_CAP_MMAC,
                    help="reject models above this MMac at 3x256x256 (0 disables)")
    ap.add_argument("--cpu-limit", type=float, default=DEFAULT_CPU_LIMIT,
                    help="CPU SECONDS per candidate (RLIMIT_CPU, threads pinned to 1). "
                         "Immune to cluster load, unlike the old wall-clock limit which "
                         "cost generation 3 of the LLM run 117 of 144 candidates while an "
                         "unrelated job saturated the nodes. 0 disables.")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="wall-clock seconds per candidate before it is killed")
    # ARCHITECTURAL DEDUP. The GA has gated on this hash at generation time since day one;
    # the LLM never has -- its only diversity check lives in elite_archive, which runs AFTER
    # candidates are trained and written to the DB. That asymmetry is why the worst duplicate
    # group in the corpus is six LLM models that are one architecture. Rejecting here costs
    # nothing: preflight has already built the model on CPU, and every rejection saves a full
    # 5-epoch GPU training run on a network we already have.
    ap.add_argument("--arch-dedupe", action="store_true",
                    help="run the project's DiversityGate on candidates that pass, so the "
                         "LLM is gated at generation time exactly as the GA already is")
    # GRADED is the reason to use the real gate rather than hash equality. Exact-hash only
    # rejects LITERAL graph clones; the graded operating point also rejects near-duplicates
    # at continuous-WL cosine >= cos_max. On this corpus that is the difference between
    # catching a handful and catching most of them: at the calibrated 0.95, 39% of traceable
    # LLM candidates were near-duplicates while exact-hash certified 98.6% of the cold arm as
    # "distinct". diversity_gate's own note says graded is a cross-FAMILY forcer and warns it
    # is wrong for a mutation-only generator whose mutants cluster >=0.985 -- but the LLM
    # writes from scratch and CAN leave the family, so it is the arm graded actually suits.
    ap.add_argument("--arch-graded", action="store_true",
                    help="also reject near-duplicates (continuous-WL cosine >= --cos-max)")
    ap.add_argument("--deploy-predict-thr", type=float, default=0.0,
                    help="reject candidates whose predicted full-CPU-eviction probability is "
                         "at or above this (0 disables; 0.5 is the measured operating point, "
                         "precision 0.90 / recall 0.83). Source-only: no TFLite conversion and "
                         "no device. Fails CLOSED if the frozen model is missing.")
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--arch-store", default="",
                    help="persist the accepted reference set here so LATER GENERATIONS are "
                         "gated against earlier ones, not just within their own batch")
    ap.add_argument("--arch-dedupe-db", default="",
                    help="also reject architectures already present in this LEMUR DB "
                         "(nn.arch_uid); needs add_arch_uid.py to have been run")
    # DEAD CODE MATTERS MORE THAN USUAL HERE, because this corpus is destined to be
    # FINE-TUNING DATA. A training example containing an nn.Module that is defined and never
    # instantiated teaches the next model to write exactly that. Generation 1 produced a
    # candidate carrying five such classes. Off by default (a model can have dead code and a
    # novel architecture, and the architectural id already catches the duplicates); on when
    # corpus cleanliness matters more than raw count.
    ap.add_argument("--reject-dead-code", action="store_true",
                    help="reject candidates defining an nn.Module they never instantiate. "
                         "SUPERSEDED by --strip-dead-code, which achieves the same corpus "
                         "cleanliness and the same duplicate collapse without the 9.9% "
                         "yield loss; kept for ablations")
    ap.add_argument("--strip-dead-code", action="store_true",
                    help="delete never-instantiated nn.Module classes instead of rejecting "
                         "the candidate (preferred)")
    ap.add_argument("--no-repair", action="store_true",
                    help="do not add missing standard imports; measures the arm's RAW "
                         "validity rate, which is the number to quote in an ablation")
    ap.add_argument("--chan-repair", type=int, default=3,
                    help="max Conv2d in_channels corrections per candidate. 40%% of "
                         "candidates fail the first forward and 56%% of those are one conv "
                         "declaring the wrong input width; the error message states the "
                         "value that works, so this corrects rather than guesses.")
    ap.add_argument("--no-chan-repair", action="store_true")
    ap.add_argument("--no-class-repair", action="store_true",
                    help="do not splice helper classes the candidate uses but never defines")
    ap.add_argument("--lenient-static", action="store_true",
                    help="only reject on static issues that are certainly fatal; let the "
                         "rest be decided by actually running the model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cands = []
    for r in a.run_dirs:
        for p in sorted(glob.glob(os.path.join(r, "cand", "*", "candidate.py"))):
            cands.append((os.path.basename(os.path.dirname(p)), p))
    cands.sort()
    print(f"=== preflight | {len(cands)} candidates | CPU only ===", flush=True)

    known = set()
    if a.arch_dedupe_db:
        try:
            import sqlite3
            _c = sqlite3.connect(f"file:{a.arch_dedupe_db}?mode=ro", uri=True)
            known = {r[0] for r in _c.execute(
                "SELECT DISTINCT arch_uid FROM nn WHERE arch_uid IS NOT NULL")}
            _c.close()
            print(f"  arch-dedupe: {len(known)} architectures already in the DB")
        except Exception as e:
            # Fail LOUD. Silently continuing with an empty set would look like "no
            # duplicates found" and let every one of them through.
            print(f"FATAL: --arch-dedupe-db unreadable ({type(e).__name__}: {e})")
            return 2

    gate = None
    if a.arch_dedupe:
        from diversity_gate import DiversityGate
        # ast_exact=True SO BOTH DEDUPE HALVES SHARE ONE IDENTITY.
        # The cross-corpus check above hashes with arch_uid2 (matching nn.arch_uid), but
        # DiversityGate defaults ast_exact=False, which routes its exact half through
        # arch_embed's torch.fx hash instead. That left preflight running two different
        # notions of "same architecture" at once, and the fx path FAILS OPEN on models it
        # cannot trace (FFT, data-dependent control flow) -- they were admitted with no
        # exact check at all. arch_uid2 is source-only, so it covers those too.
        #
        # HASH-SPACE CAVEAT (diversity_gate.py): a store written under the old fx hash reads
        # as EMPTY here rather than failing, silently re-admitting near-duplicates. Only
        # enable this alongside a store built from scratch -- which is the case for the
        # 4-adapter run (fresh $OUT/diversity_store.pt).
        gate = DiversityGate(graded=a.arch_graded, cos_max=a.cos_max,
                             store_path=(a.arch_store or None), log=print,
                             ast_exact=True)
        print(f"  diversity gate: graded={a.arch_graded} cos_max={a.cos_max} "
              f"ast_exact=True (arch_uid2) store={a.arch_store or '<none>'}")

    recs, t0 = [], time.time()
    stages, warns, repaired = Counter(), 0, 0
    spliced_tot, stripped_tot = 0, 0
    chan_fixed, chan_ok = 0, 0
    seen_arch, dup_batch, dup_db = {}, 0, 0
    dep_evict = 0
    for i, (cid, path) in enumerate(cands, 1):
        src = open(path).read()
        # WRITTEN BACK, not just repaired in memory. screen_and_train.py re-reads
        # candidate.py to train, so an in-memory-only fix would let a model pass preflight
        # and then die at the screen with the same NameError -- worse than not fixing it,
        # because it would burn a GPU slot to rediscover the same problem. The original is
        # kept beside it so the arm's raw output stays auditable.
        if not a.no_repair:
            fixed, added = repair_imports(src)
            notes = list(added)
            # Helper classes the candidate calls but never defines, recovered from the
            # references in its OWN prompt. 12.9% of all candidates died on exactly this.
            if not a.no_class_repair:
                fixed, spliced = repair_missing_classes(
                    fixed, reference_classes(os.path.dirname(path)))
                if spliced:
                    notes += [f"class {c}" for c in spliced]
                    spliced_tot += len(spliced)
            # Dead nn.Module classes: strip rather than reject, so the file is clean AND
            # hashes to its true architecture for the diversity gate.
            if a.strip_dead_code:
                fixed, dead = strip_dead_classes(fixed)
                if dead:
                    notes += [f"-dead {c}" for c in dead]
                    stripped_tot += len(dead)
            if notes:
                if not os.path.exists(path + ".orig"):
                    with open(path + ".orig", "w") as f:
                        f.write(src)
                with open(path, "w") as f:
                    f.write(fixed)
                src = fixed
                repaired += 1
        try:
            res = check_isolated(src, not a.lenient_static, a.timeout, a.mac_cap,
                                 cpu_limit=int(a.cpu_limit))
            if len(res) == 3:
                res = res + (None,)
            ok, stage, detail = res[0], res[1], res[2]
        except Exception:
            res = (False, "preflight-crash", traceback.format_exc()[-200:], None)
            ok, stage, detail = res[0], res[1], res[2]
        arch = res[3] if isinstance(res, tuple) and len(res) > 3 else None

        # CHANNEL REPAIR, in the loop rather than inside check_one, for the same reason the
        # import and helper-class repairs live here: the fixed source has to be WRITTEN BACK.
        # screen_and_train.py re-reads candidate.py to train, so an in-memory-only fix would
        # let a model pass preflight and then fail again on a GPU.
        if (not ok and stage == "forward" and not a.no_chan_repair):
            fixed, notes = src, []
            for _ in range(a.chan_repair):
                nxt, note = repair_channels(fixed, detail or "", 1)
                if nxt is None:
                    break
                fixed, _n = nxt, notes.append(note)
                try:
                    r2 = check_isolated(fixed, not a.lenient_static, a.timeout, a.mac_cap,
                                        cpu_limit=int(a.cpu_limit))
                except Exception:
                    break
                ok, stage, detail = r2[0], r2[1], r2[2]
                arch = r2[3] if len(r2) > 3 else None
                if ok or stage != "forward":
                    break
            if notes:
                chan_fixed += 1
                chan_ok += 1 if ok else 0
                if not os.path.exists(path + ".orig"):
                    with open(path + ".orig", "w") as f:
                        f.write(src)
                with open(path, "w") as f:
                    f.write(fixed)
                src = fixed
                detail = (detail or "") + f"  [channel repair: {'; '.join(notes)}]"
        gate_match = None
        if ok and a.reject_dead_code and "never instantiates it" in (detail or ""):
            ok, stage = False, "dead-code"
        # Exact match against the whole dataset first: cheapest, and it is the only check
        # that sees the OTHER arms' output (the gate's store only knows what it admitted).
        if ok and a.arch_dedupe and arch and arch in known:
            ok, stage, detail = False, "arch-dup-db", \
                f"architecture {arch[:16]} already in the dataset"
            dup_db += 1
        # SOURCE-LEVEL DEPLOYABILITY GATE. Placed after the build and dedup checks so it only
        # ever judges candidates that are real and novel, and before the GPU screen so a
        # predicted-evicted model never costs a training run. It predicts whole-model CPU
        # eviction from the AST (AUC 0.986 five-fold, 0.921 on a held-out generator), which is
        # the failure the operator rules miss: they explain only 66% of conversion failures.
        #
        # FAILS CLOSED, like every other gate here. A deployability filter that waves candidates
        # through when its model is missing is indistinguishable from no filter at all.
        if ok and a.deploy_predict_thr > 0:
            try:
                import deploy_gate as DPG
                d_ok, d_p = DPG.ok(src, a.deploy_predict_thr)
            except Exception as e:
                d_ok, d_p = False, None
                detail = (detail or "") + f"  [deploy gate unavailable: {type(e).__name__}]"
            if d_p is not None and not d_ok:
                ok, stage, detail = False, "deploy-evict", \
                    f"predicted CPU eviction p={d_p:.3f} >= {a.deploy_predict_thr:.2f}"
                dep_evict += 1
        # Then the real gate, on candidates already known to build -- so a pathological
        # model cannot take down the parent process, which the gate runs in.
        if ok and gate is not None:
            try:
                g_ok, fp = gate.check(src)
                gate_match = getattr(gate, "last_match", None)
                # Captured for BOTH outcomes: how close an ACCEPTED candidate came is as
                # useful to the generator as why a rejected one failed.
                match = gate_match
                if g_ok:
                    gate.add(cid, fp)
                    seen_arch[fp[0] if isinstance(fp, tuple) else arch] = cid
                else:
                    near = ""
                    if match:
                        near = (f" — closest is {match.get('key') or 'an earlier model'} "
                                f"at cosine {match['cos']:.3f}")
                    ok, stage, detail = False, "arch-dup-gate", \
                        f"DiversityGate rejected ({match['kind'] if match else 'near-duplicate'})" \
                        + near
                    dup_batch += 1
            except Exception as e:
                # Do not fail open silently: an unusable gate must be visible, because
                # "0 duplicates" and "the gate never ran" look identical in a summary.
                detail = "; ".join([d for d in (detail, f"note: gate error "
                                                f"({type(e).__name__})") if d])
        stages[stage if not ok else "ok"] += 1
        if ok and detail:
            warns += 1
        recs.append({"id": cid, "ok": ok, "stage": stage, "detail": detail,
                     "arch_uid": arch,
                     # Structured, not just embedded in the prose detail: rejection_digest.py
                     # turns these into the next generation's prompt, and parsing English
                     # would break the moment the wording changed.
                     "nearest": (gate_match or {}).get("key"),
                     "nearest_cos": (gate_match or {}).get("cos")})
        if i % 50 == 0 or i == len(cands):
            k = sum(1 for r in recs if r["ok"])
            print(f"  [{i}/{len(cands)}] pass={k} ({k/i:.1%})  {time.time()-t0:.0f}s",
                  flush=True)

    ok = [r for r in recs if r["ok"]]
    print(f"\n=== preflight summary ({time.time()-t0:.0f}s, {len(cands)} candidates) ===")
    for s, n in stages.most_common():
        print(f"  {s:<16} {n:5d}  ({n/len(cands):6.1%})")
    print(f"  passed with warning {warns}")
    print(f"  files repaired      {repaired}")
    print(f"    helper classes spliced from own prompt  {spliced_tot}")
    print(f"    dead nn.Module classes stripped         {stripped_tot}")
    print(f"    channel mismatches repaired             {chan_fixed}"
          f"  ({chan_ok} then passed)")
    if a.arch_dedupe:
        print(f"  arch duplicates     {dup_batch} by gate, {dup_db} already in dataset")
    if dep_evict:
        print(f"  deploy-gate rejects {dep_evict} predicted to be evicted to CPU")
        print(f"  distinct architectures passing  {len(seen_arch)}")
        # No explicit save: DiversityGate.add() persists on every acceptance, atomically
        # (tmp file + os.replace), so the reference set survives even a killed generation.
        if gate is not None and getattr(gate, "store_path", None):
            print(f"  gate reference set: {len(gate.accepted)} graphs"
                  + (f", {len(gate.embeds)} embeds" if gate.graded else "")
                  + f" -> {gate.store_path}")
    saved = (len(cands) - len(ok)) * 80 / 3600.0
    print(f"\n  GPU-hours avoided ~ {saved:.1f}  "
          f"(at the measured ~80 s per screened candidate)")

    print("\n=== top rejection reasons ===")
    c = Counter()
    for r in recs:
        if r["ok"]:
            continue
        c[re.sub(r"\d+", "N", (r["detail"] or "")[:80])] += 1
    for k, v in c.most_common(12):
        print(f"  {v:4d}  {k}")

    with open(os.path.join(a.out, "preflight.json"), "w") as f:
        json.dump(recs, f, indent=2)
    with open(os.path.join(a.out, "pass_ids.txt"), "w") as f:
        f.write("\n".join(r["id"] for r in ok) + ("\n" if ok else ""))
    print(f"\nwrote {a.out}/preflight.json and pass_ids.txt ({len(ok)} ids)")


if __name__ == "__main__":
    raise SystemExit(main())

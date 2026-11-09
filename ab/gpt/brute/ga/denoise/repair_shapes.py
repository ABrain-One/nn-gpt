"""Shape repair by OBSERVATION, replacing the regex-over-the-error-message repair.

WHY THE OLD ONE FIRED ON 9 OF 120 CASES AND FIXED 2. `preflight.repair_channels` parses the
CUDA/ATen error text for a conv mismatch, then rewrites the source with

    re.compile(r"(nn\\.Conv(?:Transpose)?2d\\(\\s*)" + str(want_in) + r"(\\s*,\\s*)" + str(out_c))

which requires the constructor to spell both channel counts as positional integer LITERALS.
Generated U-Nets almost never do. They write `nn.Conv2d(ch, ch * 2, 3, padding=1)`, or
`nn.Conv2d(in_channels=base, out_channels=base * 2, ...)`, or build the layer inside a
`Sequential`/`ModuleList` comprehension. The pattern misses, and the logged reasons say exactly
that: 38/120 "conv that failed already has matching channels" (the literals it found were not
the failing layer's), ~12 "could not isolate the conv inside self.<attr>", 42/120 "could not
build model".

This module changes three things.

  1. NO ERROR-STRING PARSING. A forward pre-hook records the tensor actually arriving at every
     module, so a mismatch is found by comparing `module.in_channels` against `input.shape[1]`
     directly. That is a fact, not an inference from a message, which is why the "already has
     matching channels" misdiagnosis cannot occur here -- if the declared and observed values
     agree, there is nothing to fix and the pass declines.

  2. NO SOURCE-LEVEL ISOLATION. Fixes are applied by REBUILDING the offending submodule and
     re-attaching it at its dotted path (`enc.0.conv1`, `blocks.2`), preserving every other
     hyperparameter read off the original object. Then the repair is emitted as a fixup block
     appended to `__init__` that re-assigns those paths. Because the emitted code addresses
     modules by path rather than by editing the constructor, it is completely indifferent to
     how the layer was originally written -- kwargs, variables, comprehensions, nesting. This
     is what removes the isolation failure class.

  3. ITERATION WITH VERIFICATION. Each fix is followed by a real forward pass, so a repair that
     merely moves the error is detected and the next one is attempted, up to a budget. The pass
     only claims success when a full forward runs and returns the contract shape.

WHAT IT DELIBERATELY WILL NOT DO. It never invents a channel count -- every value written is one
observed on a real tensor. It never changes kernel size, stride, padding, topology, or layer
count, so the architecture that gets trained is the architecture the generator proposed. Three
failure classes are therefore OUT OF SCOPE and reported rather than papered over:

    elementwise add/mul shape mismatch (70/434 of forward failures)   needs a projection ->
                                                                     would change the topology
    concat non-channel dim mismatch (22/434)                          needs interpolation ->
                                                                     would change the topology
    missing attribute, module never created (38/434)                  a genuine code bug

Fixing those means inserting layers the generator did not ask for, which is a different
intervention with a different risk profile, and mixing it in here would make the pass's effect
on the corpus impossible to attribute. The channel-arithmetic class it does target is
191/434 = 44% of forward failures.

VALIDATION IS BUILT IN, and it runs before this goes near the loop:
    python repair_shapes.py --validate --dirs <cand dirs...>   # fix rate on real failures
"""
import argparse
import ast
import copy
import glob
import json
import os
import sys
import traceback

IN_SHAPE = (1, 3, 256, 256)   # BATCHED (N,C,H,W) -- must match preflight.py:50
OUT_SHAPE = (3,)              # must match preflight.py:57
PRM = {"lr": 1e-3, "momentum": 0.9, "batch": 2, "dropout": 0.0}
PROBE = (2, 3, 256, 256)
MAX_FIXES = 48

FIXUP_MARK = "# --- preflight shape repair (auto-generated"


# ---------------------------------------------------------------------------------------------
# module surgery
# ---------------------------------------------------------------------------------------------
def _rebuild(mod, obs_in):
    """Return a clone of `mod` accepting `obs_in` input channels, or (None, why).

    Every hyperparameter except the input width is copied off the original object, so kernel
    size, stride, padding, dilation, padding_mode and bias presence survive untouched.
    """
    import torch.nn as nn

    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        groups = mod.groups
        # groups must keep dividing both channel counts. A depthwise conv (groups == in) has to
        # track the new width; anything else keeps its grouping only if it still divides.
        if groups == mod.in_channels and mod.out_channels % obs_in == 0:
            groups = obs_in
        elif groups != 1 and obs_in % groups != 0:
            return None, f"groups={groups} does not divide observed in_channels={obs_in}"
        if mod.out_channels % groups != 0:
            return None, f"groups={groups} does not divide out_channels={mod.out_channels}"
        cls = type(mod)
        kw = dict(kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding,
                  dilation=mod.dilation, groups=groups, bias=mod.bias is not None,
                  padding_mode=mod.padding_mode)
        if isinstance(mod, nn.ConvTranspose2d):
            kw["output_padding"] = mod.output_padding
        return cls(obs_in, mod.out_channels, **kw), None

    if isinstance(mod, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
        return type(mod)(obs_in, eps=mod.eps, momentum=mod.momentum,
                         affine=mod.affine,
                         track_running_stats=mod.track_running_stats), None

    if isinstance(mod, nn.GroupNorm):
        g = mod.num_groups
        if obs_in % g != 0:
            # fall back to the largest divisor <= the original group count; 1 always works
            g = next((d for d in range(min(g, obs_in), 0, -1) if obs_in % d == 0), 1)
        return nn.GroupNorm(g, obs_in, eps=mod.eps, affine=mod.affine), None

    if isinstance(mod, nn.Linear):
        return nn.Linear(obs_in, mod.out_features, bias=mod.bias is not None), None

    if isinstance(mod, nn.LayerNorm) and len(mod.normalized_shape) == 1:
        return nn.LayerNorm(obs_in, eps=mod.eps,
                            elementwise_affine=mod.elementwise_affine), None

    return None, f"{type(mod).__name__} is not a channel-parameterised layer"


def _declared_in(mod):
    """The input width `mod` claims to accept, or None if it does not constrain its input."""
    import torch.nn as nn
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        return mod.in_channels
    if isinstance(mod, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
        return mod.num_features
    if isinstance(mod, nn.GroupNorm):
        return mod.num_channels
    if isinstance(mod, nn.Linear):
        return mod.in_features
    if isinstance(mod, nn.LayerNorm) and len(mod.normalized_shape) == 1:
        return mod.normalized_shape[0]
    return None


def _obs_in(mod, t):
    import torch.nn as nn
    if t.dim() < 2:
        return None
    return t.shape[-1] if isinstance(mod, (nn.Linear, nn.LayerNorm)) else t.shape[1]


def _set_by_path(root, path, new):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def _spec(mod):
    """A source-emittable constructor call for `mod`."""
    import torch.nn as nn
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        extra = (f", output_padding={mod.output_padding}"
                 if isinstance(mod, nn.ConvTranspose2d) else "")
        return (f"_rnn.{type(mod).__name__}({mod.in_channels}, {mod.out_channels}, "
                f"kernel_size={mod.kernel_size}, stride={mod.stride}, padding={mod.padding}, "
                f"dilation={mod.dilation}, groups={mod.groups}, "
                f"bias={mod.bias is not None}, padding_mode={mod.padding_mode!r}{extra})")
    if isinstance(mod, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
        return (f"_rnn.{type(mod).__name__}({mod.num_features}, eps={mod.eps}, "
                f"momentum={mod.momentum}, affine={mod.affine}, "
                f"track_running_stats={mod.track_running_stats})")
    if isinstance(mod, nn.GroupNorm):
        return (f"_rnn.GroupNorm({mod.num_groups}, {mod.num_channels}, "
                f"eps={mod.eps}, affine={mod.affine})")
    if isinstance(mod, nn.Linear):
        return (f"_rnn.Linear({mod.in_features}, {mod.out_features}, "
                f"bias={mod.bias is not None})")
    if isinstance(mod, nn.LayerNorm):
        return (f"_rnn.LayerNorm({mod.normalized_shape[0]}, eps={mod.eps}, "
                f"elementwise_affine={mod.elementwise_affine})")
    raise TypeError(type(mod).__name__)


# ---------------------------------------------------------------------------------------------
# emit the fixup block
# ---------------------------------------------------------------------------------------------
def _emit(src, fixes):
    """Append a path-addressed re-assignment block to the end of Net.__init__."""
    tree = ast.parse(src)
    net = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Net"), None)
    if net is None:
        return None, "no class Net"
    init = next((n for n in net.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None or not init.body:
        return None, "no __init__ body"

    # Insert after the last statement, but BEFORE a trailing `return` -- appending after one
    # would put the repair in unreachable code and the model would build unrepaired.
    tail = init.body[-1]
    anchor = init.body[-2] if isinstance(tail, ast.Return) and len(init.body) > 1 else tail
    if isinstance(tail, ast.Return) and len(init.body) == 1:
        return None, "__init__ body is only a return"
    line = anchor.end_lineno            # 1-based, inclusive
    col = anchor.col_offset
    pad = " " * col

    body = [f"{pad}{FIXUP_MARK}, {len(fixes)} fix(es)) ---",
            f"{pad}import torch.nn as _rnn",
            f"{pad}def _rset(_root, _path, _mod):",
            f"{pad}    _p = _path.split('.'); _o = _root",
            f"{pad}    for _q in _p[:-1]:",
            f"{pad}        _o = _o[int(_q)] if _q.isdigit() else getattr(_o, _q)",
            f"{pad}    if _p[-1].isdigit(): _o[int(_p[-1])] = _mod",
            f"{pad}    else: setattr(_o, _p[-1], _mod)"]
    for f in fixes:
        body.append(f"{pad}_rset(self, {f['path']!r}, {f['spec']})   "
                    f"# {f['kind']}: {f['declared']} -> {f['observed']}")

    lines = src.splitlines()
    out = lines[:line] + body + lines[line:]
    return "\n".join(out) + ("\n" if src.endswith("\n") else ""), None


# ---------------------------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------------------------
def _build(src):
    ns = {}
    exec(compile(src, "<cand>", "exec"), ns)
    Net = ns.get("Net")
    if Net is None:
        raise RuntimeError("no Net after exec")
    return Net(IN_SHAPE, OUT_SHAPE, dict(PRM), "cpu")


def repair(src, max_fixes=MAX_FIXES, verbose=False):
    """-> dict(ok, src, fixes, stage, detail, unrepairable)

    `ok` means a full forward ran and returned the contract shape AFTER the fixes. `src` is the
    patched source when fixes were applied and verified, else the original.
    """
    import torch

    res = {"ok": False, "src": src, "fixes": [], "stage": None,
           "detail": None, "unrepairable": None}

    def _decline(**kw):
        """Return a failure, ALWAYS carrying the fixes applied so far."""
        res.update(**kw)
        res["fixes"] = list(fixes)
        return res
    if FIXUP_MARK in src:
        res["detail"] = "already carries a repair block"
        return res
    try:
        net = _build(src)
    except Exception as e:
        res["stage"] = "build"
        res["detail"] = f"{type(e).__name__}: {e}"
        return res

    fixes = []
    for _ in range(max_fixes + 1):
        seen = []
        handles = []

        def pre(mod, inp, _seen=seen):
            if inp and torch.is_tensor(inp[0]):
                _seen.append((mod, inp[0]))

        for m in net.modules():
            handles.append(m.register_forward_pre_hook(pre))
        try:
            net.eval()
            x = torch.rand(*PROBE)
            with torch.no_grad():
                y = net(x)
            err = None
        except Exception as e:
            y, err = None, e
        finally:
            for h in handles:
                h.remove()

        if err is None:
            if not torch.is_tensor(y):
                res.update(stage="forward",
                           detail=f"forward returned {type(y).__name__}, not a tensor")
                return res
            if tuple(y.shape) != tuple(x.shape):
                res.update(stage="forward",
                           detail=f"output shape {tuple(y.shape)} != input {tuple(x.shape)}")
                return res
            if not torch.isfinite(y).all():
                res.update(stage="forward", detail="forward produced NaN/Inf")
                return res
            # converged
            if not fixes:
                res.update(ok=True, detail="no repair needed")
                return res
            patched, why = _emit(src, fixes)
            if patched is None:
                res.update(stage="emit", detail=f"could not emit fixup: {why}")
                return res
            # The emitted source must itself build and run -- never hand back a patch that
            # only worked as live surgery.
            try:
                net2 = _build(patched)
                net2.eval()
                with torch.no_grad():
                    y2 = net2(torch.rand(*PROBE))
                assert torch.is_tensor(y2) and tuple(y2.shape) == PROBE
            except Exception as e:
                res.update(stage="emit",
                           detail=f"patched source failed to verify: {type(e).__name__}: {e}")
                return res
            res.update(ok=True, src=patched, fixes=fixes,
                       detail=f"{len(fixes)} fix(es), verified from source")
            return res

        # --- a module raised. The last pre-hook to fire is the module that failed, because the
        # hook runs immediately before its own forward.
        if not seen:
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable="no module was entered")
        mod, t = seen[-1]
        decl, obs = _declared_in(mod), _obs_in(mod, t)
        if decl is None or obs is None:
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable=f"failed inside {type(mod).__name__}, not a "
                                         "channel-parameterised layer -- elementwise/concat "
                                         "op or a functional (F.conv2d) call in forward()")
        if decl == obs:
            # The declared width already matches what arrived, so this is NOT the channel
            # class. Declining here is the whole point: the old pass guessed in this situation.
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable=f"{type(mod).__name__} at declared in={decl} received "
                                         f"in={obs} -- widths agree, cause is elsewhere "
                                         "(spatial size, kernel/stride, or an op in forward)")
        if len(fixes) >= max_fixes:
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable=f"fix budget {max_fixes} exhausted after "
                                         f"{len(fixes)} fix(es)")

        path = next((n for n, m in net.named_modules() if m is mod), None)
        if path is None:
            res.update(stage="forward", unrepairable="failing module has no named path")
            return res
        new, why = _rebuild(mod, obs)
        if new is None:
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable=f"cannot rebuild {path}: {why}")
        try:
            _set_by_path(net, path, new)
        except Exception as e:
            res.update(stage="forward",
                       unrepairable=f"could not re-attach at {path}: {e}")
            return res
        if any(f["path"] == path for f in fixes):
            prev = next(f for f in fixes if f["path"] == path)
            return _decline(stage="forward", detail=f"{type(err).__name__}: {err}",
                            unrepairable=f"oscillating: {path} already fixed "
                                         f"{prev['declared']}->{prev['observed']}, now asked for "
                                         f"{decl}->{obs}; the mismatch is not local to it")
        fixes.append({"path": path, "kind": type(mod).__name__,
                      "declared": decl, "observed": obs, "spec": _spec(new)})
        if verbose:
            print(f"    fix {len(fixes)}: {path} {type(mod).__name__} "
                  f"in {decl} -> {obs}", file=sys.stderr)

    return _decline(stage="forward", unrepairable="did not converge within budget")


# ---------------------------------------------------------------------------------------------
# validation harness
# ---------------------------------------------------------------------------------------------
def _worker():
    """Repair ONE file and print the row as JSON. Runs in its own process.

    ISOLATION IS NOT OPTIONAL HERE. A first version ran every candidate in-process and was
    SIGKILLed at candidate 26 of 120 with no traceback and no output file -- a generated net that
    declares enormous channel counts allocates its weights at construction, and a 256x256 probe
    on top of that exhausts memory. In-process, one such candidate destroys the whole run and
    every result already computed. Per-candidate subprocesses convert that from a lost run into
    one recorded `oom-or-killed` row, and the address-space cap makes it a catchable
    MemoryError in most cases rather than a kill.
    """
    import resource
    cap = int(os.environ.get("REPAIR_MEM_MB", "6000")) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass
    path = sys.argv[sys.argv.index("--worker") + 1]
    src = open(path, encoding="utf8", errors="replace").read()
    try:
        r = repair(src)
    except MemoryError:
        r = {"ok": False, "stage": "resource", "detail": "MemoryError",
             "fixes": [], "src": src, "unrepairable": "exceeded the memory cap"}
    except Exception:
        r = {"ok": False, "stage": "repair-crash", "detail": traceback.format_exc()[-300:],
             "fixes": [], "src": src, "unrepairable": "repair pass itself raised"}
    row = {"file": path, "ok": r["ok"], "pre_ok": r.get("detail") == "no repair needed",
           "n_fixes": len(r["fixes"]), "fixes": r["fixes"], "stage": r.get("stage"),
           "detail": (r.get("detail") or "")[:300], "unrepairable": r.get("unrepairable")}
    if r["ok"] and r["fixes"] and os.environ.get("REPAIR_WRITE") == "1":
        if not os.path.exists(path + ".orig"):
            open(path + ".orig", "w").write(src)
        open(path, "w").write(r["src"])
        row["written"] = True
    sys.stdout.write("\x01JSON\x01" + json.dumps(row) + "\n")


def _cli():
    if "--worker" in sys.argv:
        return _worker()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[],
                    help="candidate dirs to scan for *.py (or the .py files themselves)")
    ap.add_argument("--list", default=None,
                    help="file of newline-separated paths; avoids a multi-KB argv, which is "
                         "what made the first validation run unreadable in ps output")
    ap.add_argument("--only-failing", action="store_true",
                    help="skip candidates whose forward already passes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-candidate JSON here")
    ap.add_argument("--write", action="store_true",
                    help="write repaired source back (keeps a .orig)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--per-file-timeout", type=float, default=120.0,
                    help="seconds per candidate; a hung forward must not stall the sweep")
    ap.add_argument("--mem-mb", type=int, default=6000,
                    help="address-space cap per worker (RLIMIT_AS)")
    a = ap.parse_args()

    files = []
    if a.list:
        files += [l.strip() for l in open(a.list) if l.strip()]
    for d in a.dirs:
        files += [d] if d.endswith(".py") else sorted(glob.glob(os.path.join(d, "*.py")))
    if not files:
        ap.error("give --list and/or --dirs")
    if a.limit:
        files = files[:a.limit]
    print(f"scanning {len(files)} candidate file(s)", flush=True)

    import subprocess
    from collections import Counter
    tally = Counter()
    rows = []
    # results are appended as they land, so a killed driver still leaves everything computed
    jl = (a.out or "repair_results.json") + ".jsonl"
    jlf = open(jl, "a")
    env = dict(os.environ)
    env["REPAIR_MEM_MB"] = str(a.mem_mb)
    if a.write:
        env["REPAIR_WRITE"] = "1"
    for i, f in enumerate(files, 1):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", f]
        row = None
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=a.per_file_timeout, env=env)
            for ln in p.stdout.splitlines():
                if ln.startswith("\x01JSON\x01"):
                    row = json.loads(ln[6:])
            if row is None:
                row = {"file": f, "ok": False, "pre_ok": False, "n_fixes": 0, "fixes": [],
                       "stage": "killed", "detail": (p.stderr or "")[-200:],
                       "unrepairable": f"worker died (rc={p.returncode}), likely OOM"}
        except subprocess.TimeoutExpired:
            row = {"file": f, "ok": False, "pre_ok": False, "n_fixes": 0, "fixes": [],
                   "stage": "timeout", "detail": f"exceeded {a.per_file_timeout}s",
                   "unrepairable": "timeout"}
        if a.only_failing and row["pre_ok"]:
            tally["already passing (skipped)"] += 1
            continue
        if row["pre_ok"]:
            tally["already passing"] += 1
        elif row["ok"]:
            tally[f"REPAIRED ({row['n_fixes']} fix)"] += 1
        else:
            tally[f"still failing: {row.get('stage')}"] += 1
        rows.append(row)
        jlf.write(json.dumps(row) + "\n")
        jlf.flush()
        if i % 10 == 0:
            print(f"  {i}/{len(files)}  repaired so far: "
                  f"{sum(1 for r in rows if r['ok'] and not r['pre_ok'])}", flush=True)
    jlf.close()

    print("\n--- outcome ---")
    for k, v in tally.most_common():
        print(f"  {v:>5}  {k}")
    broken = [r for r in rows if not r["pre_ok"]]
    fixed = [r for r in broken if r["ok"]]
    if broken:
        print(f"\nof {len(broken)} candidates whose forward failed, "
              f"{len(fixed)} repaired = {100*len(fixed)/len(broken):.1f}%")
        print(f"  fixes applied per repaired candidate: "
              f"{sum(r['n_fixes'] for r in fixed)/max(1,len(fixed)):.1f} mean")
        reasons = Counter((r["unrepairable"] or "")[:70] for r in broken if not r["ok"])
        print("\n  why the rest were declined (this pass never guesses):")
        for k, v in reasons.most_common(8):
            print(f"    {v:>5}  {k or '(no reason recorded)'}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    _cli()

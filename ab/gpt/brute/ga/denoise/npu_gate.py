"""NPU deployability gate — reject undeployable architectures BEFORE spending 50 GPU-minutes on them.

THE PROBLEM THIS SOLVES. The pipeline gates on MMac <= 9053 as its edge-legality proxy. Measured on
128 models on a Galaxy S26 Ultra (Hexagon v81), 33 of them -- 26% -- exceed 10 ms at 256x256 while
the median is 1.72 ms, and the worst reaches 238 ms. Every one passes the MAC budget, because a
multiply-accumulate count cannot see whether an operator has an NPU kernel or whether a graph will
be shoved onto the CPU. The LLM arm is worst hit: 8 of 11 profiled LLM models run entirely off the
NPU at a median 55.21 ms against 1.58 ms for DenoiseGA. So the corpus spends ~50 GPU-minutes per
candidate training architectures that cannot ship.

WHY THE GATE IS EMPIRICAL AND NOT A STATIC OPERATOR CHECK -- THIS WAS TESTED, NOT ASSUMED.
An operator allowlist looked extremely promising at first: labelling models by npu_time_frac == 0,
the rule "uses torch.matmul or nn.PixelUnshuffle" separated all 97 labelled models perfectly, 12/12
caught and 0/85 wrongly rejected. That result is an ARTIFACT of the label. Fully-off-NPU models are
only a subset of the slow ones, and re-testing against the label that actually matters -- latency --
collapses it:

    label = latency > 10 ms (33 of 128)     recall   precision
      matmul or PixelUnshuffle                36%       50%
      any of {matmul,Softmax,normalize,sqrt,gelu,zeros}   36%   80%

It misses two of every three undeployable models, because the ~60 ms DenoiseGA cluster is slow for
reasons no operator signature can see (depth and width, not unsupported ops). A 36%-recall gate is
worse than none, since it would license the belief that survivors are deployable. The operator table
is therefore emitted as PROMPT GUIDANCE for the generator -- which is where an n=12 signal belongs --
and never as a gate.

WHY MEASURING IS AFFORDABLE. Conversion is a median 3.4 s per model on CPU, and AI Hub profile jobs
run in parallel on the farm. 250 candidates cost ~14 min on one CPU worker, ~2 min across six, and
zero GPU. Against ~50 GPU-minutes per wasted training run at a 26% failure rate, the gate pays for
itself roughly 100x.

THRESHOLD IS CHOSEN FROM THE DATA, NOT ROUND-NUMBER TASTE. Sorted latencies have their largest
multiplicative gap at 6.60 -> 17.93 ms (2.7x), so 10 ms sits inside a natural discontinuity and the
verdict is insensitive to moving it a few ms either way. npu_time_frac is recorded as the diagnostic
that explains WHY a model failed, never as the criterion: models at 0.57 fraction still take 60 ms.

  # once, anywhere:            python npu_gate.py --validate
  # per candidate batch:       python npu_gate.py --convert --src DIR --work W
  #                            python npu_gate.py --profile --work W        (login node: needs token)
  #                            python npu_gate.py --verdict --work W --max-ms 10
"""
import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys

# RESOLVE LIKE EVERY OTHER MODULE HERE. This was hardcoded to the host path, which is fine while
# the gate only ever ran on the submit node -- but it is now imported by run_seed_ga's export gate,
# which runs INSIDE a pod where the tree is mounted at /a/mm and /shared/ssd does not exist. The
# hardcoded path would have made _find_converter_py find nothing and the deployability check
# unavailable exactly where it is needed. deploy_objective.py and npu_residency.py already do this.
THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = os.path.join(THESIS, "sweep_out")

def _find_converter_py():
    """First interpreter that actually has ai_edge_torch, checked by inspecting site-packages.

    THE HARDCODED PATH WAS WRONG AND FAILED SILENTLY UNTIL RUN. This was
    `THESIS/venv_tflite/bin/python`, which exists -- so cmd_convert's os.path.exists check passed --
    but that venv holds only litert_torch; ai_edge_torch lives in `THESIS/tflite-venv` (note the
    transposed name). The gate would therefore clear its own preflight and then die inside
    reexport_convert.py on ImportError, which reads as "these candidates cannot convert" rather
    than "the converter is missing" -- the exact silent-degradation failure this file warns about
    elsewhere.

    Detection is by directory glob, not by import: importing ai_edge_torch drags in TensorFlow and
    takes 20-60 s, which is not acceptable in a preflight that runs per batch.
    """
    cands = [os.path.join(THESIS, "tflite-venv", "bin", "python"),
             os.path.join(THESIS, "venv_tflite", "bin", "python"),
             "/home/b-b-ahmed/thesis/tflite-venv/bin/python"]
    for py in cands:
        if not os.path.exists(py):
            continue
        root = os.path.dirname(os.path.dirname(py))
        if glob.glob(os.path.join(root, "lib", "python*", "site-packages", "ai_edge_torch")):
            return py
    return cands[0]          # report the preferred path in the error message rather than None


VENV_PY = os.environ.get("NPU_GATE_PY") or _find_converter_py()
DEVICE = "Samsung Galaxy S26 Ultra"      # the farm ceiling; a model that is slow here is slow anywhere
# --device overrides DEVICE for a single run. Added 2026-08-30 for the cross-device study: the
# corpus holds 2,850 measurements on the S26 Ultra and 991 on the S24 with ZERO models in common,
# because bulk_profile.py treats the two devices as parallel QUEUES and sends each model to one of
# them. That makes every cross-device statement unsupportable -- the S24/S26 median ratio of 1.63x
# is a comparison of two disjoint model populations, not of two devices. Re-profiling an existing
# S26 subset on the S24 is the only way to get paired measurements.
#
# The override is deliberately a CLI argument rather than a second constant: cmd_verdict and
# cmd_calibrate_ops both stamp DEVICE into their output and filter on it (lines ~525, ~580, ~792),
# so a run against a different device must carry that device's name through to its artifacts or
# the two sets silently pool. Callers that pass nothing keep the S26 default unchanged.
SIZE = 256                               # corpus training/export resolution
DEFAULT_MAX_MS = 10.0
OP = re.compile(r"\b(?:nn|F|torch)\.([A-Za-z_][A-Za-z0-9_]*)")

# Names that are never on-device compute: optimisers, losses, schedulers, dtypes and plumbing.
# A regex over the whole file counts these, and they then dominate the risk table purely because
# they are boilerplate -- the first version ranked SyncBatchNorm, BatchNorm3d, L1Loss and optim at
# the top, which would have pushed the generator away from BatchNorm for no reason at all.
_NOT_COMPUTE = {
    "optim", "Adam", "AdamW", "SGD", "RMSprop", "Adamax", "lr_scheduler", "CosineAnnealingLR",
    "StepLR", "ReduceLROnPlateau", "OneCycleLR", "ExponentialLR", "MSELoss", "L1Loss",
    "SmoothL1Loss", "HuberLoss", "CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss",
    "Module", "ModuleList", "Sequential", "Parameter", "ParameterList", "functional",
    "float32", "float16", "bfloat16", "tensor", "Tensor", "device", "no_grad", "cuda",
    "utils", "init", "clip_grad_norm_", "manual_seed", "Generator", "as_tensor",
}
_TRAIN_ONLY_METHODS = {"train_setup", "learn", "forward_pass", "supported_hyperparameters"}


# ---------------------------------------------------------------- unconvertible constructs
# THIS DOES NOT CONTRADICT THE DOCSTRING ABOVE. That section refuses to PREDICT LATENCY from
# operator signatures, and the numbers there stand: 36% recall, useless as a speed gate. What
# follows predicts nothing. It encodes constructs for which ai_edge_torch conversion is
# IMPOSSIBLE, so there is no TFLite graph, no AI Hub profile, and no deployment -- a precondition
# rather than a correlation. The gate already treats a conversion failure as a legitimate
# rejection (cmd_verdict, reason "convert-failed"); this only moves that same verdict earlier,
# from 3.4 s of tracing to a millisecond of parsing, and makes the reason nameable so the
# generator can be told what to stop emitting.
#
# THE BAR FOR ENTRY IS ZERO FALSE POSITIVES, MEASURED. Each rule was tested over the 2,827-model
# tflite_pool census -- 2,477 that converted, 350 that hard-failed -- and admitted only if it
# appears in NO model that converted successfully. Run `--ban-audit` to recompute the table; do
# not trust the counts written here.
#
#   rule                        in hard-failed   in converted-OK   admitted
#   AdaptiveMaxPool2d                      187                 0   yes
#   self.<attr> = ... in forward            45                 0   yes
#   torch.fft.* (any form)                  53                10   NO
#   amax / amin                             41                 3   NO
#
# THE TWO REJECTED RULES ARE THE WHOLE POINT OF THE BAR. Both look compelling -- 84% and 93%
# precision, and the converter names them explicitly ("FFT normalization != 0 is not yet
# implemented", "NHWC node rewriter not found: amax"). Banning them would still have discarded 13
# architectures that convert and profile perfectly well. Narrowing the FFT rule to calls carrying
# a norm= kwarg does not help: it matches exactly the same 53 and the same 10, so whatever
# separates them is not visible in the source. They belong in operator_risk.json as prompt
# guidance, alongside the latency skew table, and are reported by --ban-audit as WARN.
_ADAPTIVE_MAX_POOL = re.compile(r"\bAdaptiveMaxPool2d\b|\badaptive_max_pool2d\b")
_FORWARD_METHODS = ("forward", "_forward")


def _uses_adaptive_max_pool(code):
    """aten.adaptive_max_pool2d has no TFLite lowering at all -- 187 of the 350 pool failures.

    Regex rather than AST because the construct is unambiguous as a name and this must also catch
    sources that do not parse cleanly under the running interpreter.
    """
    return bool(_ADAPTIVE_MAX_POOL.search(code))


def _mutates_self_in_forward(code):
    """Assignment to a self attribute inside forward -- export refuses graph-time mutation.

    The converter reports it as `ValueError: The tensor attribute self._mid was assigned during
    export` (also seen for self._sigma and self._aux[i]). Detected structurally, covering both
    `self.x = ...` and `self.x[i] = ...`.

    KEPT DELIBERATELY BROAD. Assigning a plain None or an int to a self attribute would not upset
    the tracer, so a narrower rule is defensible -- but the broad form is the one measured at 45/0
    over the census, and narrowing it would invalidate that evidence for no observed gain. If a
    future candidate is rejected here for a non-tensor assignment, tighten the rule and re-run
    --ban-audit rather than granting an exception.

    UNLIKE THE OTHER RULE THIS IS A REPAIRABLE DEFECT, not an architectural dead end: buffers
    belong in __init__ via register_buffer. --ban-audit labels it FIXABLE so the generator can be
    taught to rewrite instead of merely avoiding.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in _FORWARD_METHODS):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assign):
                continue
            for tgt in sub.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                        and tgt.value.id == "self":
                    return True
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute) \
                        and isinstance(tgt.value.value, ast.Name) and tgt.value.value.id == "self":
                    return True
    return False


# (rule id, test, human reason, repairable?)
UNCONVERTIBLE = [
    ("adaptive-max-pool", _uses_adaptive_max_pool,
     "AdaptiveMaxPool2d/adaptive_max_pool2d has no TFLite lowering (aten.adaptive_max_pool2d)",
     False),
    ("self-mutation-in-forward", _mutates_self_in_forward,
     "assigns to a self attribute inside forward; ai_edge_torch rejects mutation during export -- "
     "declare it in __init__ with register_buffer",
     True),
]

# Constructs the census flags but that DO sometimes convert. Never gate on these.
UNCONVERTIBLE_WARN = [
    ("fft", re.compile(r"\btorch\.fft\."), "53 of 63 fft models failed to convert"),
    ("amax", re.compile(r"\.amax\(|\btorch\.amax\b|\.amin\(|\btorch\.amin\b"),
     "41 of 44 amax/amin models failed to convert (NHWC node rewriter)"),
]


def unconvertible(code):
    """[(rule_id, reason, repairable)] for every hard ban this source trips; [] if it can convert."""
    return [(rid, why, fixable) for rid, test, why, fixable in UNCONVERTIBLE if test(code)]


def unconvertible_warnings(code):
    """[(rule_id, note)] for risky-but-not-banned constructs. Advisory only."""
    return [(rid, note) for rid, rx, note in UNCONVERTIBLE_WARN if rx.search(code)]


def compute_ops(code):
    """nn/F/torch names that plausibly execute on device.

    AST-based and scoped: only names inside __init__ (module construction) and forward-like methods
    are counted, and training-only methods are skipped entirely, so an optimiser or a scheduler
    cannot enter the table. Falls back to the plain regex if the source does not parse -- a
    candidate that fails to parse never reaches the gate anyway.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {o for o in OP.findall(code) if o not in _NOT_COMPUTE}
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in _TRAIN_ONLY_METHODS:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                    and sub.value.id in ("nn", "F", "torch"):
                if sub.attr not in _NOT_COMPUTE:
                    found.add(sub.attr)
    return found


# ---------------------------------------------------------------- validate
def labelled():
    """(name, code, latency_ms, npu_frac) for every profiled model whose source we still have."""
    codes = {}
    p = os.path.join(SWEEP, "tflite_pool", "codes.jsonl")
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            codes[r["name"]] = r["code"]
    out = []
    for f in ("results_s26u.json", "results.json"):
        q = os.path.join(SWEEP, "aihub", f)
        if not os.path.exists(q):
            continue
        for m in json.load(open(q))["models"]:
            if m.get("inference_us") and m["model"] in codes:
                out.append((m["model"], codes[m["model"]], m["inference_us"] / 1000.0,
                            m.get("npu_time_frac")))
        break
    return out


def cmd_validate(a):
    rows = labelled()
    if not rows:
        sys.exit("no labelled data -- run the AI Hub sweep first")
    lat = sorted(r[2] for r in rows)
    bad = [r for r in rows if r[2] > a.max_ms]
    print(f"{len(rows)} labelled models | p50 {lat[len(lat)//2]:.2f} ms | "
          f"{len(bad)} ({len(bad)/len(rows):.0%}) exceed {a.max_ms} ms")
    off = [r for r in rows if (r[3] or 0) <= 0.001]
    print(f"{len(off)} are fully OFF the NPU -- a SUBSET of the slow ones, which is why the")
    print("operator rule looks perfect against that label and fails against latency:\n")

    ATT = {"matmul", "Softmax", "normalize", "sqrt", "gelu", "zeros"}
    tests = [("matmul or PixelUnshuffle",
              lambda o: "matmul" in o or "PixelUnshuffle" in o),
             ("any of {matmul,Softmax,normalize,sqrt,gelu,zeros}", lambda o: bool(o & ATT))]
    for lbl, key in (("npu_time_frac == 0", lambda r: (r[3] or 0) <= 0.001),
                     (f"latency > {a.max_ms} ms", lambda r: r[2] > a.max_ms)):
        print(f"  label = {lbl}")
        for name, fn in tests:
            tp = fp = tn = fn_ = 0
            for r in rows:
                y, p = key(r), fn(compute_ops(r[1]))
                tp += y and p
                fp += (not y) and p
                tn += (not y) and (not p)
                fn_ += y and (not p)
            print(f"    {name:<48} recall {tp/max(1,tp+fn_):>5.0%} "
                  f"precision {tp/max(1,tp+fp):>5.0%}  (missed {fn_}, wrongly rejected {fp})")
    print("\nVERDICT: static operator rules are NOT a usable gate. Measure instead.")

    # operator risk table -> prompt guidance for the generator
    slow = [r for r in rows if r[2] > a.max_ms]
    fast = [r for r in rows if r[2] <= a.max_ms]
    risk = []
    for op in {o for r in rows for o in compute_ops(r[1])}:
        ns = sum(1 for r in slow if op in compute_ops(r[1]))
        nf = sum(1 for r in fast if op in compute_ops(r[1]))
        if ns + nf < 3:
            continue
        risk.append((ns / max(1, len(slow)) - nf / max(1, len(fast)), op, ns, nf))
    risk.sort(reverse=True)
    print(f"\noperator risk table (share of slow models minus share of fast), top 12:")
    print(f"  {'operator':<22}{'in slow':>9}{'in fast':>9}{'skew':>8}")
    for d, op, ns, nf in risk[:12]:
        print(f"  {op:<22}{ns:>9}{nf:>9}{d:>8.2f}")
    o = os.path.join(SWEEP, "aihub", "operator_risk.json")
    json.dump({"max_ms": a.max_ms, "n_slow": len(slow), "n_fast": len(fast),
               "note": "PROMPT GUIDANCE ONLY -- measured recall as a gate is ~36%, see --validate",
               "risk": [{"op": op, "in_slow": ns, "in_fast": nf, "skew": round(d, 4)}
                        for d, op, ns, nf in risk]}, open(o, "w"), indent=1)
    print(f"\nwrote {o}  (for the generator's prompt, NOT for gating)")


# ---------------------------------------------------------------- ban audit
def census():
    """(code_by_name, converted_ok, hard_failed) from the tflite_pool conversion census.

    A model that failed in one shard and succeeded in another counts as OK: 1,847 of the 2,827
    models were processed by more than one shard (the shards overlapped), so taking failures at
    face value would invent false negatives.
    """
    P = os.path.join(SWEEP, "tflite_pool")
    code = {}
    p = os.path.join(P, "codes.jsonl")
    if not os.path.exists(p):
        sys.exit(f"FATAL no census at {p} -- cannot audit the ban list without it")
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            code[r["name"]] = r["code"]
    ok, bad = {}, {}
    for f in sorted(glob.glob(os.path.join(P, "report_shard*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            (ok if r.get("ok") else bad).setdefault(r["model"], r)
    bad = {k: v for k, v in bad.items() if k not in ok}
    return code, ok, bad


def cmd_ban_audit(a):
    """Recompute every ban rule against the census. A rule with ANY hit among converted models is
    a bug in the rule, and this exits nonzero rather than letting it keep rejecting candidates."""
    code, ok, bad = census()
    print(f"census: {len(ok)} converted OK | {len(bad)} hard-failed | {len(code)} sources")
    print(f"\n{'rule':<26}{'in FAILED':>10}{'in OK':>7}{'precision':>11}  status")
    broken, rows = [], []
    for rid, test, why, fixable in UNCONVERTIBLE:
        nf = sum(1 for m in bad if m in code and test(code[m]))
        no = sum(1 for m in ok if m in code and test(code[m]))
        prec = nf / (nf + no) if nf + no else 0.0
        bad_rule = no > 0
        if bad_rule:
            broken.append(rid)
        print(f"{rid:<26}{nf:>10}{no:>7}{prec:>10.1%}  "
              f"{'*** REJECTS WORKING MODELS ***' if bad_rule else ('FIXABLE' if fixable else 'BAN')}")
        rows.append({"rule": rid, "in_failed": nf, "in_ok": no, "precision": round(prec, 4),
                     "repairable": fixable, "valid": not bad_rule})
    print(f"\n{'advisory (never gated)':<26}{'in FAILED':>10}{'in OK':>7}{'precision':>11}")
    for rid, rx, note in UNCONVERTIBLE_WARN:
        nf = sum(1 for m in bad if m in code and rx.search(code[m]))
        no = sum(1 for m in ok if m in code and rx.search(code[m]))
        prec = nf / (nf + no) if nf + no else 0.0
        print(f"{rid:<26}{nf:>10}{no:>7}{prec:>10.1%}  WARN -- {note}")
        rows.append({"rule": rid, "in_failed": nf, "in_ok": no, "precision": round(prec, 4),
                     "advisory": True})

    # How much of the failure census do the admitted bans actually explain? The remainder is the
    # honest measure of what a static pre-check cannot catch.
    caught = {m for m in bad if m in code and unconvertible(code[m])}
    print(f"\nbans explain {len(caught)}/{len(bad)} hard failures ({len(caught)/max(1,len(bad)):.0%}); "
          f"{len(bad)-len(caught)} need a real conversion attempt")
    o = os.path.join(SWEEP, "aihub", "unconvertible_rules.json")
    json.dump({"n_ok": len(ok), "n_failed": len(bad), "explained": len(caught),
               "rules": rows}, open(o, "w"), indent=1)
    print(f"wrote {o}")
    if broken:
        sys.exit(f"FATAL rules reject models that convert: {broken} -- fix or remove them")


def cmd_ban_check(a):
    """Report which sources in --src are unconvertible, without converting anything."""
    srcs = sorted(glob.glob(os.path.join(a.src, "*.py")))
    if not srcs:
        sys.exit(f"FATAL no *.py in {a.src}")
    nbad = 0
    for p in srcs:
        c = open(p).read()
        hits, warns = unconvertible(c), unconvertible_warnings(c)
        if hits:
            nbad += 1
            for rid, why, fixable in hits:
                print(f"REJECT {os.path.basename(p)[:48]:<50} {rid} "
                      f"({'repairable' if fixable else 'dead end'})")
        for rid, note in warns:
            print(f"  warn {os.path.basename(p)[:48]:<50} {rid}: {note}")
    print(f"\n{nbad}/{len(srcs)} unconvertible by static rule")


# ---------------------------------------------------------------- graph size
# THE ONE DEPLOYABILITY RULE THAT CATCHES THE DOMINANT FAILURE. The ban rules above cover
# convertibility (6.1% of the corpus). The far bigger problem is a model that converts, profiles,
# and then executes on the CPU: 37% of 321 profiled LLM models, at a median 71 ms against 1.8 ms
# for models that stayed on the NPU. Measured over 1,350 profiled models with per-layer compute
# units (npu_residency.py), that is a GRAPH SIZE effect, not an operator effect:
#
#     off-NPU median graph ops   LLM 377, DenoiseGA 175
#     fully-on-NPU median         LLM  70, DenoiseGA  62
#     depth-alone AUC             LLM 0.965, DenoiseGA 0.973
#
# and every operator that topped the raw contrast stopped separating once models of comparable
# depth were compared -- they were markers of deep graphs, not causes. So the gate counts ops.
#
# BUT THE RULE PREDICTS SLOWNESS, NOT RESIDENCY, AND THIS WAS MEASURED THE HARD WAY. A convertible
# variant of DenoiseUnet7 (439 ops) was profiled to settle it: it ran at npu_time_frac = 1.000 --
# ENTIRELY on the NPU, no fallback at all -- and took 2,621 ms, some 1,570x the 1.67 ms of the
# 85-op champion. So a large graph does not reliably get pushed onto the CPU; it just runs slowly
# on the NPU. Against the label that actually decides deployability, >185 ops is the better rule:
#
# RECALIBRATED ON THE LATENCY LABEL, AND IT IS A WEAKER RULE THAN THE RESIDENCY FIT SUGGESTED:
#
#     label                     operating point      recall   precision   good models discarded
#     off-NPU (frac == 0)       --max-ops 185          85%        86%       25 of 839
#     latency > 10 ms           --max-ops  91 (F1)     82%        71%      210 of 722
#     latency > 10 ms           --max-ops 203 (used)   43%        96%       12 of 722
#
# THE HONEST READING: against slowness, graph size is a HIGH-PRECISION, LOW-RECALL filter. At the
# operating point in use it is right 96% of the time when it fires, and it fires on fewer than half
# the models that are actually too slow. It therefore does NOT certify that survivors are fast --
# plenty of slow models are small, being slow for reasons an op count cannot see (channel width,
# expensive single ops). The F1 optimum is deliberately NOT used: at 91 ops it would discard 210 of
# 722 deployable architectures to gain recall, and in a search loop losing an architecture is worse
# than wasting 52 min of training on one.
#
# WHY THIS RUNS AFTER CONVERSION AND NOT ON THE SOURCE. A source-side module count does not work:
# against the profiled graph size it scores Pearson r=0.298 / Spearman 0.589, and the best
# source-count threshold reaches 95% recall at only 35% precision -- it would reject 308 of 857
# deployable models. The reason is loops: `nn.Sequential(*[Block() for _ in range(12)])` is one
# textual construction and twelve times the graph. The count is only honest once the graph exists.
# Conversion already costs 3.4 s and runs BEFORE training, so this still saves the 52 min of
# full-training a model that cannot ship, and it saves an AI Hub profile job too.
DEFAULT_MAX_OPS = 203        # calibrated on the LATENCY label (>10 ms), precision-first: 96%
                             # wrongly rejected (839 on-NPU / 175 off-NPU artifacts). Re-derive
                             # with --calibrate-ops; 0 disables the check.


def count_ops(tfl_dir, out_path):
    """{model: n_ops} for every artifact, via the converter venv (this interpreter has no TF).

    Failures are recorded per model rather than aborting: an artifact whose op count cannot be read
    must not be silently treated as small, and cmd_verdict distinguishes "unknown" from "over".
    """
    arts = sorted(glob.glob(os.path.join(tfl_dir, "*.tflite")))
    if not arts:
        return {}
    code = (
        "import glob,json,os,re,sys,warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import tensorflow as tf\n"
        "out={}\n"
        "for f in sorted(glob.glob(os.path.join(sys.argv[1],'*.tflite'))):\n"
        f"    n=re.sub(r'_{SIZE}x{SIZE}_fp32\\.tflite$','',os.path.basename(f))\n"
        "    try:\n"
        "        it=tf.lite.Interpreter(model_path=f); it.allocate_tensors()\n"
        "        out[n]=len(it._get_ops_details())\n"
        "    except Exception as e:\n"
        "        out[n]=None\n"
        "json.dump(out,open(sys.argv[2],'w'))\n")
    try:
        subprocess.call([VENV_PY, "-c", code, tfl_dir, out_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  graph-size count failed to launch: {type(e).__name__}: {e}")
        return {}
    if not os.path.exists(out_path):
        print("  graph-size count produced nothing; the layer gate will report 'unknown'")
        return {}
    d = json.load(open(out_path))
    known = {k: v for k, v in d.items() if v}
    if known:
        vals = sorted(known.values())
        print(f"  graph size: {len(known)} counted, median {vals[len(vals)//2]} ops, "
              f"max {vals[-1]}")
    return d


def cmd_calibrate_ops(a):
    """Re-derive --max-ops from the profiled corpus: graph op count vs NPU residency.

    CALIBRATED ON THE COUNT THE GATE ACTUALLY SEES, which is the whole point of doing this rather
    than reusing the threshold from npu_residency.py. That analysis found >170 to be the best cut
    on AI Hub's own `n_layers`, but the gate cannot read n_layers -- that arrives only with a
    profile. It reads the TFLite op count from the artifact. The two are related but not the same
    number (AI Hub fuses and splits), so transplanting the threshold would be an assumption. This
    measures it directly on the pool's 2,511 artifacts.
    """
    pool = os.path.join(SWEEP, "tflite_pool", "tflite")
    cache = a.ops_json or os.path.join(SWEEP, "aihub", "tflite_op_counts.json")
    if not os.path.exists(cache):
        print(f"counting ops in {pool} -> {cache} (a few minutes)")
        count_ops(pool, cache)
    if not os.path.exists(cache):
        sys.exit("FATAL no op counts available; is the converter venv present?")
    ops = {k: v for k, v in json.load(open(cache)).items() if v}

    lp = os.path.join(SWEEP, "aihub", "latency_corpus.json")
    lat = json.load(open(lp)) if os.path.exists(lp) else {}

    def al(n):
        return {n, n.replace("_", "-"), n.replace("-", "_")}

    # LABEL CHOICE, AND WHY IT DEFAULTS TO LATENCY. The first calibration used NPU residency
    # (npu_time_frac == 0) because the fallback was assumed to be the failure mode. Profiling a
    # convertible DenoiseUnet7 variant disproved that: 439 ops, npu_time_frac 1.000, and 2,621 ms.
    # A large graph is not reliably evicted to the CPU; it is reliably SLOW. Deployability is a
    # latency question, so that is the label to fit. Residency stays available for comparison.
    bad, good = [], []
    for n, v in lat.items():
        if not (isinstance(v, dict) and v.get("status") == "OK"):
            continue
        if v.get("device") and v["device"] != DEVICE:
            continue
        hit = next((x for x in al(n) if x in ops), None)
        if hit is None:
            continue
        if a.calibrate_label == "residency":
            f = v.get("npu_time_frac")
            if f is None:
                continue
            (bad if f <= 0.001 else (good if f >= 0.999 else [])).append(ops[hit])
        else:
            if not v.get("ms"):
                continue
            (bad if v["ms"] > a.max_ms else good).append(ops[hit])
    lbl = (f"latency > {a.max_ms} ms" if a.calibrate_label == "latency"
           else "off-NPU (npu_time_frac == 0)")
    print(f"label: {lbl}   ->  {len(bad)} positive, {len(good)} negative")
    if len(bad) < 20 or len(good) < 20:
        sys.exit("FATAL too few labelled artifacts to calibrate")

    print(f"\n{'threshold':>10}{'recall':>9}{'precision':>11}{'F1':>7}{'good rejected':>16}")
    rows, best_f1, best_prec = [], None, None
    lo, hi = min(bad + good), max(bad + good)
    for thr in range(max(10, lo), min(hi, 1200), 1):
        tp = sum(1 for x in bad if x > thr)
        fp = sum(1 for x in good if x > thr)
        fn = len(bad) - tp
        if tp + fp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((thr, prec, rec, f1, fp))
        if best_f1 is None or f1 > best_f1[3]:
            best_f1 = (thr, prec, rec, f1, fp)
        # PRECISION-FIRST ALTERNATIVE. The two errors are not symmetric: a false negative wastes
        # 52 min of training, while a false positive discards an architecture the search may never
        # rediscover. So the highest-recall threshold that still keeps precision >= --min-precision
        # is reported alongside the F1 optimum, and is usually the one to prefer in a search loop.
        if prec >= a.min_precision and (best_prec is None or rec > best_prec[2]):
            best_prec = (thr, prec, rec, f1, fp)
    for thr, prec, rec, f1, fp in rows:
        if thr % 25 == 0:
            print(f"{thr:>10}{rec:>8.0%}{prec:>11.0%}{f1:>7.2f}{fp:>11} of {len(good)}")
    print(f"\nF1-OPTIMAL       --max-ops {best_f1[0]:<5} recall {best_f1[2]:.0%}  "
          f"precision {best_f1[1]:.0%}  discards {best_f1[4]} of {len(good)} good models")
    if best_prec:
        print(f"PRECISION-FIRST  --max-ops {best_prec[0]:<5} recall {best_prec[2]:.0%}  "
              f"precision {best_prec[1]:.0%}  discards {best_prec[4]} of {len(good)} good models"
              f"   (>= {a.min_precision:.0%} precision)")
        print(f"\nRECOMMENDED: {best_prec[0]}. The errors are asymmetric -- a missed slow model costs")
        print(f"52 min of training, a wrongly rejected fast one costs an architecture. Set")
        print(f"DEFAULT_MAX_OPS = {best_prec[0]}.")
    else:
        print(f"\nNo threshold reaches {a.min_precision:.0%} precision; use the F1 optimum "
              f"({best_f1[0]}) or lower --min-precision.")
    json.dump({"device": DEVICE, "label": lbl, "n_positive": len(bad), "n_negative": len(good),
               "f1_optimal": {"threshold": best_f1[0], "precision": best_f1[1],
                              "recall": best_f1[2], "f1": best_f1[3], "discards": best_f1[4]},
               "precision_first": (None if not best_prec else
                                   {"threshold": best_prec[0], "precision": best_prec[1],
                                    "recall": best_prec[2], "discards": best_prec[4],
                                    "min_precision": a.min_precision})},
              open(os.path.join(SWEEP, "aihub", "ops_gate_calibration.json"), "w"), indent=1)
    return 0


# ---------------------------------------------------------------- convert
def cmd_convert(a):
    # EXISTENCE IS NOT ENOUGH -- check the package, not just the interpreter. See _find_converter_py.
    if not os.path.exists(VENV_PY) or not glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(VENV_PY)), "lib", "python*",
                         "site-packages", "ai_edge_torch")):
        sys.exit(f"FATAL no interpreter with ai_edge_torch (tried {VENV_PY}) -- the ai-linux image "
                 f"ships neither litert_torch nor ai_edge_torch; build the venv or set NPU_GATE_PY")
    tfl = os.path.join(a.work, "tflite")
    os.makedirs(tfl, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(a.src, "*.py")))

    # STATIC PRE-SCREEN. Sources tripping an audited ban cannot produce a TFLite graph, so tracing
    # them only buys the same verdict 3.4 s later. They are written straight into convert.json as
    # failures with reason ids, and the converter is handed only the survivors -- in a separate
    # directory, because reexport_convert.py globs whatever --src contains.
    banned, survivors = [], []
    for p in srcs:
        hits = unconvertible(open(p).read())
        if hits:
            banned.append({"model": os.path.splitext(os.path.basename(p))[0], "ok": False,
                           "unconvertible": [h[0] for h in hits],
                           "repairable": all(h[2] for h in hits),
                           "error": "; ".join(f"{h[0]}: {h[1]}" for h in hits)})
        else:
            survivors.append(p)
    if banned:
        print(f"pre-screen: {len(banned)}/{len(srcs)} rejected as unconvertible before tracing")
        for r in banned[:10]:
            print(f"  {r['model'][:52]:<54} {','.join(r['unconvertible'])}")
        if len(banned) > 10:
            print(f"  ... and {len(banned)-10} more")

    src_dir = a.src
    if banned:
        src_dir = os.path.join(a.work, "src_convertible")
        if os.path.isdir(src_dir):
            for old in glob.glob(os.path.join(src_dir, "*.py")):
                os.remove(old)
        os.makedirs(src_dir, exist_ok=True)
        for p in survivors:
            with open(os.path.join(src_dir, os.path.basename(p)), "w") as fh:
                fh.write(open(p).read())

    report = os.path.join(a.work, "convert.json")
    rc = 0
    if survivors:
        print(f"converting {len(survivors)} candidates @ {SIZE}x{SIZE} -> {tfl}")
        conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reexport_convert.py")
        rc = subprocess.call([VENV_PY, conv, "--src", src_dir, "--out", tfl,
                              "--h", str(SIZE), "--w", str(SIZE), "--report", report])
    else:
        print("every candidate was rejected by the static pre-screen; converter not invoked")

    # MERGE, DO NOT OVERWRITE. reexport_convert.py rewrites the report from scratch, so the banned
    # rows have to be folded back in afterwards or cmd_verdict would never see them and would score
    # the batch as if those candidates had not been submitted.
    if banned:
        rows = json.load(open(report)) if os.path.exists(report) else []
        have = {r.get("model") for r in rows}
        rows.extend(r for r in banned if r["model"] not in have)
        json.dump(rows, open(report, "w"), indent=1)

    # GRAPH SIZE, COUNTED FROM THE CONVERTED ARTIFACT. This is the strongest deployability signal
    # we have and it cannot be had any earlier: see count_ops for why the source cannot supply it.
    count_ops(tfl, os.path.join(a.work, "layers.json"))

    n = len(glob.glob(os.path.join(tfl, "*.tflite")))
    print(f"{n} artifacts | {len(banned)} pre-screened out")
    # A candidate that will not even convert is not deployable either -- that is a verdict, not an
    # error, and cmd_verdict records it as such. Exit 0 when the batch was fully adjudicated, even
    # with zero artifacts: "all candidates are undeployable" is a result, whereas a converter that
    # produced nothing from convertible input is a fault worth failing on.
    sys.exit(0 if (n or not survivors) else 1)


# ---------------------------------------------------------------- profile
def cmd_profile(a):
    import qai_hub as hub
    tfl = os.path.join(a.work, "tflite")
    state_p = os.path.join(a.work, "jobs.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {}
    arts = sorted(glob.glob(os.path.join(tfl, "*.tflite")))
    for p in arts:
        name = re.sub(rf"_{SIZE}x{SIZE}_fp32\.tflite$", "", os.path.basename(p))
        if name in st:
            continue
        try:
            job = hub.submit_profile_job(model=p, device=hub.Device(DEVICE), name=f"gate {name}")
            st[name] = {"job_id": job.job_id}
            tmp = state_p + ".tmp"
            json.dump(st, open(tmp, "w"), indent=1)
            os.replace(tmp, state_p)
            print(f"  submitted {name[:44]} -> {job.job_id}", flush=True)
        except Exception as e:
            print(f"  submit FAILED {name[:44]}: {type(e).__name__}: {e}")
    print(f"{len(st)} jobs total; collecting")

    out_p = os.path.join(a.work, "profile.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    pend = 0
    for name, v in st.items():
        if name in out:
            continue
        try:
            job = hub.get_job(v["job_id"])
            s = job.get_status()
            if not s.finished:
                pend += 1
                continue
            if not s.success:
                out[name] = {"status": "PROFILE_FAILED"}
                continue
            prof = job.download_profile()
            ex = prof.get("execution_summary", {}) if isinstance(prof, dict) else {}
            us = ex.get("estimated_inference_time") or ex.get("inference_time")
            det = prof.get("execution_detail") or []
            unit = {}
            for lay in det:
                if isinstance(lay, dict):
                    cu = lay.get("compute_unit") or "UNKNOWN"
                    unit[cu] = unit.get(cu, 0) + (lay.get("execution_time") or 0)
            tot = sum(unit.values()) or 1
            out[name] = {"status": "OK" if us else "NO_LATENCY",
                         "ms": round(us / 1000.0, 3) if us else None,
                         "npu_time_frac": round(unit.get("NPU", 0) / tot, 4),
                         "device": job.device.name,
                         "fallback_units": sorted(u for u in unit if u != "NPU")}
        except Exception as e:
            print(f"  collect FAILED {name[:44]}: {type(e).__name__}: {e}")
    json.dump(out, open(out_p, "w"), indent=1)
    print(f"{len(out)} profiled | {pend} pending -> {out_p}")
    sys.exit(2 if pend else 0)          # 2 tells a caller to poll again


# ---------------------------------------------------------------- verdict
def cmd_verdict(a):
    conv = {}
    p = os.path.join(a.work, "convert.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            conv[r["model"]] = r
    prof = {}
    p = os.path.join(a.work, "profile.json")
    if os.path.exists(p):
        prof = json.load(open(p))
    layers = {}
    p = os.path.join(a.work, "layers.json")
    if os.path.exists(p):
        layers = json.load(open(p))

    names = sorted(set(conv) | set(prof))
    rows = []
    for n in names:
        c, pr = conv.get(n, {}), prof.get(n, {})
        if c.get("unconvertible"):
            # Distinct from convert-failed on purpose: this one names WHICH construct is at fault
            # and whether it is repairable, which is what the generator needs back. Collapsing it
            # into convert-failed would hide the only rejection class we can act on.
            rows.append({"model": n, "pass": False, "reason": "unconvertible-op",
                         "rules": c["unconvertible"], "repairable": c.get("repairable", False),
                         "detail": str(c.get("error"))[:160]})
        elif not c.get("ok"):
            rows.append({"model": n, "pass": False, "reason": "convert-failed",
                         "detail": str(c.get("error"))[:160]})
        elif a.max_ops and layers.get(n) and layers[n] > a.max_ops:
            # REJECTED BEFORE THE PROFILE AND BEFORE TRAINING. This is a probabilistic rule, not a
            # certainty like the convertibility bans -- at the calibrated threshold it is right
            # about 9 times in 10 -- so it is reported with the count that triggered it, and
            # --max-ops 0 disables it outright for anyone who wants the old behaviour.
            rows.append({"model": n, "pass": False, "reason": "too-deep",
                         "n_ops": layers[n], "max_ops": a.max_ops,
                         "detail": f"{layers[n]} graph ops > {a.max_ops}; models this large "
                                   f"exceeded 10 ms in 306 of 334 measured cases (91.6%)"})
        elif pr.get("status") != "OK" or pr.get("ms") is None:
            rows.append({"model": n, "pass": False, "reason": "not-profiled",
                         "detail": pr.get("status", "missing")})
        elif pr["ms"] > a.max_ms:
            rows.append({"model": n, "pass": False, "reason": "too-slow",
                         "ms": pr["ms"], "npu_time_frac": pr.get("npu_time_frac"),
                         "fallback_units": pr.get("fallback_units")})
        else:
            rows.append({"model": n, "pass": True, "ms": pr["ms"],
                         "npu_time_frac": pr.get("npu_time_frac")})
    ok = [r for r in rows if r["pass"]]
    # COVERAGE IS PART OF THE VERDICT, NOT A FOOTNOTE.
    # "profiled and too slow" and "never profiled" are completely different facts, and collapsing
    # them makes a congested device farm indistinguishable from a batch of undeployable models.
    # That is not hypothetical: this gate's own validation run had its 10 jobs queued behind a
    # 400-job bulk submission, timed out with zero profiled, and wrote "0/10 pass -- all
    # not-profiled". Fed to the pipeline that verdict would have discarded an entire generation and
    # been logged as the generator producing nothing deployable.
    # r.get, NOT r["reason"] -- PASSING rows carry no "reason" key at all, so the subscript form
    # raises KeyError the moment a single model passes. The coverage check was added and tested
    # against a batch where everything failed as "not-profiled", so the crash could not fire; the
    # first real batch with a passing model then died here, left a STALE verdict.json on disk from
    # an earlier manual run, and the comparison silently scored that stale file as 0/10.
    measured = [r for r in rows if r.get("reason") != "not-profiled"]
    coverage = len(measured) / max(1, len(rows))
    inconclusive = coverage < a.min_coverage
    o = os.path.join(a.work, "verdict.json")
    json.dump({"device": DEVICE, "size": SIZE, "max_ms": a.max_ms,
               "n": len(rows), "n_pass": len(ok),
               "n_measured": len(measured), "coverage": round(coverage, 4),
               "min_coverage": a.min_coverage, "inconclusive": inconclusive,
               "pass": [r["model"] for r in ok], "rows": rows}, open(o, "w"), indent=1)
    import collections
    print(f"{len(ok)}/{len(rows)} pass at <= {a.max_ms} ms on {DEVICE}")
    print(f"  coverage: {len(measured)}/{len(rows)} actually profiled ({coverage:.0%})")
    print("  rejections:", dict(collections.Counter(r["reason"] for r in rows if not r["pass"])))
    if inconclusive:
        print(f"  VERDICT IS INCONCLUSIVE -- coverage {coverage:.0%} < {a.min_coverage:.0%}.")
        print("  --filter-ids will FAIL OPEN (emit every id) rather than pretend the unmeasured")
        print("  candidates were rejected. Re-run --profile once the farm queue drains.")
    else:
        saved = (len(rows) - len(ok)) * 50 / 60.0
        print(f"  GPU-hours NOT spent on undeployable candidates: ~{saved:.1f}")
    print(f"wrote {o}")


# ---------------------------------------------------------------- pipeline glue
def cmd_stage_ids(a):
    """Copy the candidate.py of every id in --ids-file into <work>/src as <id>.py.

    Sources live at <run_dir>/cand/<name>/candidate.py -- the same layout preflight.py globs at
    line 684. Ids may carry a run-dir prefix (llmprod passed four run dirs, so ids came out as
    `p10_0000_seeded_...` while the directory is `0000_seeded_...`), so an exact directory match is
    tried first and a suffix match second. An id that resolves to nothing is reported rather than
    skipped silently -- a gate that quietly drops candidates would look like a strict gate.
    """
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    src = os.path.join(a.work, "src")
    os.makedirs(src, exist_ok=True)
    avail = {}
    for r in a.run_dirs:
        for p in glob.glob(os.path.join(r, "cand", "*", "candidate.py")):
            avail[os.path.basename(os.path.dirname(p))] = p
    staged, missing = 0, []
    for i in ids:
        p = avail.get(i)
        if p is None:
            hits = [v for k, v in avail.items() if i.endswith(k)]
            p = hits[0] if len(hits) == 1 else None
        if p is None:
            missing.append(i)
            continue
        safe = re.sub(r"[^A-Za-z0-9._+-]", "_", i)
        with open(os.path.join(src, safe + ".py"), "w") as fh:
            fh.write(open(p).read())
        staged += 1
    print(f"staged {staged}/{len(ids)} candidate sources -> {src}")
    if missing:
        print(f"WARNING {len(missing)} ids resolved to no candidate.py: {missing[:5]}")
    json.dump({"ids": ids, "staged": staged, "missing": missing},
              open(os.path.join(a.work, "stage_ids.json"), "w"), indent=1)
    sys.exit(0 if staged else 1)


def cmd_filter_ids(a):
    """Write the id list the downstream screen should use: verdict 'pass' only.

    Fails LOUD if the verdict is missing rather than passing everything through -- a gate that
    silently degrades to a no-op is the failure mode this project has been bitten by repeatedly
    (an `exit 0` on an empty preflight once hid a crash for five hours).
    """
    v = os.path.join(a.work, "verdict.json")
    if not os.path.exists(v):
        sys.exit(f"FATAL no verdict at {v} -- refusing to emit an unfiltered id list")
    d = json.load(open(v))
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    # FAIL OPEN ON LOW COVERAGE. A gate that did not measure the batch has no opinion about it, and
    # the honest response is to pass everything through with a loud warning -- unverified
    # deployability is recoverable, a silently discarded generation is not.
    if d.get("inconclusive"):
        with open(a.out_ids, "w") as fh:
            fh.write("\n".join(ids) + ("\n" if ids else ""))
        print(f"GATE INCONCLUSIVE (coverage {d.get('coverage', 0):.0%} < "
              f"{d.get('min_coverage', 0):.0%}) -- FAILING OPEN")
        print(f"  emitted all {len(ids)} ids unfiltered to {a.out_ids}")
        print("  deployability is UNVERIFIED for this batch; re-run --profile then --verdict")
        return
    keep = set(d["pass"])
    # verdict keys are the sanitised file stems; map back to the original ids
    out = [i for i in ids if re.sub(r"[^A-Za-z0-9._+-]", "_", i) in keep or i in keep]
    with open(a.out_ids, "w") as fh:
        fh.write("\n".join(out) + ("\n" if out else ""))
    print(f"{len(out)}/{len(ids)} ids survive the NPU gate -> {a.out_ids}")
    if ids and not out:
        print("NOTE every candidate in this batch is undeployable; the screen will have nothing "
              "to measure. That is a real verdict, not an error.")


def main():
    global DEVICE          # must precede the `default=DEVICE` read below
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="re-test static rules; print op risk table")
    ap.add_argument("--ban-audit", action="store_true",
                    help="recompute the unconvertible-op rules against the tflite_pool census; "
                         "exits nonzero if any rule rejects a model that converted")
    ap.add_argument("--ban-check", action="store_true",
                    help="report which sources in --src are unconvertible, converting nothing")
    ap.add_argument("--stage-ids", action="store_true", help="ids-file + run-dirs -> work/src")
    ap.add_argument("--filter-ids", action="store_true", help="verdict -> filtered ids for the screen")
    ap.add_argument("--ids-file")
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--out-ids")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--device", default=DEVICE,
                    help="AI Hub device to profile on. Defaults to the S26 Ultra; set to "
                         "'Samsung Galaxy S24 (Family)' for the paired cross-device study.")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--src", help="dir of candidate *.py")
    ap.add_argument("--work", default=os.path.join(SWEEP, "npu_gate"))
    ap.add_argument("--max-ms", type=float, default=DEFAULT_MAX_MS)
    ap.add_argument("--max-ops", type=int, default=DEFAULT_MAX_OPS,
                    help="reject a converted model with more than this many TFLite graph ops, "
                         "before profiling and before training. 0 disables. Graph size is the "
                         "strongest measured predictor of CPU fallback (see count_ops); calibrate "
                         "with --calibrate-ops")
    ap.add_argument("--calibrate-ops", action="store_true",
                    help="re-derive the --max-ops threshold from the profiled corpus and exit")
    ap.add_argument("--calibrate-label", choices=["latency", "residency"], default="latency",
                    help="what --calibrate-ops fits the threshold to. latency (default) uses "
                         "--max-ms, which is what deployability actually means; residency uses "
                         "npu_time_frac==0 and is kept only for comparison")
    ap.add_argument("--min-precision", type=float, default=0.95,
                    help="precision floor for the recommended threshold; the two errors are "
                         "asymmetric (see cmd_calibrate_ops)")
    ap.add_argument("--ops-json", default=None,
                    help="cached {model: n_ops} to calibrate from instead of recounting")
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="fraction of the batch that must actually be profiled for the verdict to "
                         "count; below this the gate fails OPEN instead of rejecting unmeasured "
                         "candidates (see cmd_verdict)")
    a = ap.parse_args()
    # Rebind the module constant so cmd_verdict / cmd_calibrate_ops, which stamp and filter on
    # DEVICE, agree with what was actually profiled instead of always claiming the S26.
    DEVICE = a.device
    os.makedirs(a.work, exist_ok=True)
    if a.validate:
        cmd_validate(a)
    elif a.ban_audit:
        cmd_ban_audit(a)
    elif a.calibrate_ops:
        return cmd_calibrate_ops(a)
    elif a.ban_check:
        if not a.src:
            sys.exit("--ban-check needs --src")
        cmd_ban_check(a)
    elif a.stage_ids:
        if not (a.ids_file and a.run_dirs):
            sys.exit("--stage-ids needs --ids-file and --run-dirs")
        cmd_stage_ids(a)
    elif a.filter_ids:
        if not (a.ids_file and a.out_ids):
            sys.exit("--filter-ids needs --ids-file and --out-ids")
        cmd_filter_ids(a)
    elif a.convert:
        if not a.src:
            sys.exit("--convert needs --src")
        cmd_convert(a)
    elif a.profile:
        cmd_profile(a)
    elif a.verdict:
        cmd_verdict(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

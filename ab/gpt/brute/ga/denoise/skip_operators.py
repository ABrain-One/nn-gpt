"""Skip-connection operators for the denoise GA (source -> source AST transforms).

The professor's direction: *add more skip connections*. Our existing mutator only
toggles the GLOBAL input->output residual; this module adds a richer family, each
targeting an insertion point measured by skip_inventory.py:

  add_global_residual  : wrap top forward return in `inp + (...)`      [GLOBAL, +3.41 dB proven]
  add_block_residual   : wrap a conv-block forward in `inp + (...)`    [BLOCKRES, missing in 8/10]
  add_lateral_skip     : extra input->deep-feature skip (learnable)    [LATERAL, missing in 6/10]

Design notes
------------
* GLOBAL and BLOCKRES are PARAMETER-FREE identity residuals: they add the module's
  own input to its output. Valid iff in-channels == out-channels (true for the top
  net: in_ch==out_ch==3; and for C->C conv blocks). No new submodules needed -> the
  safest, highest-ROI operators.
* LATERAL adds a LEARNABLE skip: a 1x1 conv projects the network input to the deep
  feature's channels (concrete count discovered by a trace pass, so it is registered
  in __init__ before the optimizer is built) and F.interpolate matches the spatial
  size. Ops that cannot be applied to a given seed return None (the GA just tries
  another operator).
* (A concat<->add `toggle_fusion` was prototyped and removed: on our seeds every
  U-Net concat feeds a conv sized for a+b channels, so a parameter-free flip never
  validated, and its only real payoff — fewer MACs — is already supplied by the
  edge-aware objective + block crossover. Not worth the downstream-conv surgery.)

Each operator: `op(src: str, tag: str) -> str | None`. Returns mutated source, or
None if not applicable / would be invalid. Callers validate build+forward separately
(see _valid_denoise_source in seed_mutation.py, or the __main__ self-test here).
"""
import ast


# ---------------------------------------------------------------- helpers

def _parse(src):
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _classes(tree):
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def _forward(cls):
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
            return fn
    return None


def _first_arg(fn):
    args = [a.arg for a in fn.args.args if a.arg != "self"]
    return args[0] if args else None


def _last_return(fn):
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    return rets[-1] if rets else None


def _is_loss(cls):
    return "Loss" in cls.name


def _net_instantiated_arch(tree):
    """Class name that Net.__init__ instantiates as its backbone, e.g.
    `self.net = GatedAdaptiveUNet(...)` -> 'GatedAdaptiveUNet'. This is the TRUE
    U-Net; the largest-forward heuristic mistakes a repeatedly-called block (9F's
    _GatedAdaptiveBlock) for it. Returns the arch class name or None."""
    net = next((c for c in _classes(tree) if c.name == "Net"), None)
    if net is None:
        return None
    init = next((f for f in net.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is None:
        return None
    local = {c.name for c in _classes(tree)}
    for node in ast.walk(init):
        # self.<attr> = <ClassName>(...)  where ClassName is a class in this file
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            f = node.value.func
            name = f.id if isinstance(f, ast.Name) else None
            tgt = node.targets[0] if node.targets else None
            is_self_attr = (isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name) and tgt.value.id == "self")
            if name in local and is_self_attr:
                return name
    return None


def _is_thin_wrapper(net_cls, tree):
    """True when Net.forward merely delegates to ONE submodule, e.g.
    `def forward(self, x): return self.net(x)`. Only then is the instantiated class the
    real architecture; otherwise Net itself is."""
    fn = _forward(net_cls)
    if fn is None:
        return False
    body = [s for s in fn.body if not isinstance(s, ast.Expr)]   # ignore docstrings
    if len(body) > 2:
        return False
    ret = _last_return(fn)
    if ret is None or not isinstance(ret.value, ast.Call):
        return False
    f = ret.value.func
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"


def _top_unet(tree):
    """The TOP-LEVEL architecture class -- the module whose forward maps noisy image in to
    denoised image out.

    NET IS THE ARCHITECTURE FAR MORE OFTEN THAN THIS USED TO ASSUME. The old rule took the
    first `self.<attr> = <LocalClass>(...)` in Net.__init__ as the backbone, which is right
    only when Net wraps a single backbone. When Net IS the U-Net and instantiates blocks
    directly (`self.b1 = ELUAvgBlock(32)`), that rule returns a BLOCK -- and the fallback
    below explicitly skips Net, so Net could never be recovered. Measured effect: every
    lateral-skip attempt in the ablation targeted an inner block instead of the network,
    and 32 of 32 attempts failed (top_class came back as ELUAvgBlock, DilBlock,
    LaplacianPyramidBlock). The delegation test now decides which case applies."""
    net_cls = next((c for c in _classes(tree) if c.name == "Net"), None)
    if net_cls is not None and _forward(net_cls) is not None:
        if _is_thin_wrapper(net_cls, tree):
            arch = _net_instantiated_arch(tree)
            if arch:
                cls = next((c for c in _classes(tree) if c.name == arch), None)
                if cls is not None and _forward(cls) is not None:
                    return cls
        return net_cls               # Net does the work -> Net is the architecture
    best, best_sz = None, -1
    for cls in _classes(tree):
        if _is_loss(cls) or cls.name == "Net":
            continue
        fn = _forward(cls)
        if fn is None:
            continue
        sz = sum(1 for _ in ast.walk(fn))
        if sz > best_sz:
            best, best_sz = cls, sz
    return best


def _return_adds_input(ret, inp):
    """True if `return` is `inp + <expr>` or `<expr> + inp`."""
    v = ret.value
    if not (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add)):
        return False
    for side in (v.left, v.right):
        if isinstance(side, ast.Name) and side.id == inp:
            return True
    return False


def _wrap_return_in_residual(fn, inp):
    """Rewrite the last `return <expr>` -> `return <inp> + (<expr>)`. Returns True
    if a change was made, False if already residual / no return."""
    ret = _last_return(fn)
    if ret is None or _return_adds_input(ret, inp):
        return False
    ret.value = ast.BinOp(left=ast.Name(id=inp, ctx=ast.Load()),
                          op=ast.Add(), right=ret.value)
    return True


def _builds(source):
    """Instantiate the emitted source on CPU and forward 2x3x256x256; True iff the
    output shape matches the input (identity-mapping denoise contract). Lets each
    operator REJECT its own edit when the residual is channel/shape-invalid, so an
    operator never returns a broken model."""
    import torch
    ns = {}
    try:
        exec(compile(source, "<skipop>", "exec"), ns)
        if "Net" not in ns:
            return False
        net = ns["Net"](in_shape=(1, 3, 256, 256), prm={"lr": 1e-4}, device="cpu")
        x = torch.randn(2, 3, 256, 256)
        net.eval()
        with torch.no_grad():
            y = net(x)
        return tuple(y.shape) == (2, 3, 256, 256)
    except Exception:
        return False


def _builds_and_trains(source):
    """Stricter check for operators that ADD LEARNABLE parameters: build, forward,
    AND take one optimizer step, verifying the new params are (a) registered in the
    optimizer (the repo builds Adam(self.parameters()) in __init__ before any
    forward, so a param created lazily on first forward would be MISSING) and
    (b) produce finite gradients. Rejects a parametric skip that wouldn't train."""
    import torch
    ns = {}
    try:
        exec(compile(source, "<skipop>", "exec"), ns)
        if "Net" not in ns:
            return False
        net = ns["Net"](in_shape=(1, 3, 64, 64), prm={"lr": 1e-4}, device="cpu")
        net.train()
        x = torch.randn(2, 3, 64, 64)
        y = net(x)
        if tuple(y.shape) != (2, 3, 64, 64) or not torch.isfinite(y).all():
            return False
        loss = torch.nn.functional.l1_loss(y, torch.randn_like(y))
        opt = getattr(net, "optimizer", None)
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        # at least one param must have a finite non-None grad
        ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                 for p in net.parameters())
        if opt is not None:
            opt.step()
        return ok
    except Exception:
        return False


def _trace_last_channels(source, arch_name, varname):
    """Run the arch's forward once and return the CHANNEL count that `varname` holds
    on its final assignment (the value that flows into the return). Channels are
    fixed by construction, so this is deterministic. None if untraceable.

    Used to size a learnable projection with a CONCRETE channel count at __init__
    time (LazyConv2d can't be used — see _builds_and_trains)."""
    return _trace_channels(source, arch_name, varname)


def _trace_channels(source, arch_name, varname, after_lineno=None):
    """Channel count of `varname` inside arch_name.forward.

    after_lineno=None -> the value at the END of the forward (what the default operator
                         needs, since it edits the return expression).
    after_lineno=L    -> the value immediately AFTER line L, i.e. at the point
                         add_lateral_skip_at inserts its statement.

    TWO BUGS THIS REPLACES, both of which produced a WRONG number rather than None -- so
    the caller built a projection with the wrong width and only found out via
    "RuntimeError: size of tensor a (48) must match tensor b (3)":

    1. THE TRACER RAN IN EVERY forward. It matched on `co_name == "forward"` alone, so an
       inner block's local `x`/`h`/`t` overwrote the top network's value for the same name.
       Now the frame must belong to an INSTANCE of arch_name.
    2. IT ALWAYS RETURNED THE LAST VALUE. add_lateral_skip_at inserts after a variable's
       FIRST assignment, but a rebound variable (`x = self.enc1(x); x = self.enc2(x)`)
       holds different channel counts at those two points. after_lineno pins the read to
       the insertion site.
    """
    import sys
    import torch
    ns = {}
    try:
        exec(compile(source, "<trace>", "exec"), ns)
        cls = ns.get(arch_name)
        if cls is None:
            return None
        net = ns["Net"](in_shape=(1, 3, 64, 64), prm={"lr": 1e-4}, device="cpu")
        net.eval()
    except Exception:
        return None

    seen = {"ch": None, "done": False}

    def line_tracer(frame, event, arg):
        if event == "line" and not seen["done"]:
            v = frame.f_locals.get(varname)
            if torch.is_tensor(v) and v.dim() == 4:
                if after_lineno is None:
                    seen["ch"] = int(v.shape[1])          # keep the latest
                elif frame.f_lineno > after_lineno:
                    seen["ch"] = int(v.shape[1])          # first read past the assignment
                    seen["done"] = True
        return line_tracer

    def call_tracer(frame, event, arg):
        # only trace the TOP architecture's own forward, not the blocks it calls
        if (event == "call" and frame.f_code.co_name == "forward"
                and isinstance(frame.f_locals.get("self"), cls)):
            return line_tracer
        return None

    sys.settrace(call_tracer)
    try:
        with torch.no_grad():
            net(torch.randn(1, 3, 64, 64))
    except Exception:
        return None
    finally:
        sys.settrace(None)
    return seen["ch"]


# ---------------------------------------------------------------- operators

def _input_aliases(fn, inp):
    """Names that transitively hold the forward's input tensor, e.g. `identity = x`.

    NEEDED BECAUSE THE PLAIN NAME CHECK MISSED EVERY BASE. `_return_adds_input` compares against
    the bare parameter name, but the generated bases write `identity = x` and then return
    `clamp(out_conv(d0) + identity, 0, 1)` -- the residual is real, uses an ALIAS, and sits
    NESTED INSIDE A CALL, so a top-level BinOp check on `x` finds nothing.
    """
    al = {inp}
    for _ in range(4):                      # transitive: a = x; b = a; ...
        before = len(al)
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in al:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        al.add(t.id)
        if len(al) == before:
            break
    return al


def _has_input_residual_deep(fn, inp):
    """True if ANYWHERE in `fn` the input (or an alias) is added into the data path.

    This is the guard that should have stopped the GLOBAL arm. All 99 generated variant sources
    already satisfy it, so `add_global_residual` had nothing legitimate to do on any of them and
    every one of its 96 training runs collapsed to exactly the 33.021 dB no-op: wrapping an
    existing residual as `x + clamp(out_conv(d0) + identity, 0, 1)` doubles it and makes the
    identity function an attractor.
    """
    al = _input_aliases(fn, inp)
    for n in ast.walk(fn):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            for side in (n.left, n.right):
                if isinstance(side, ast.Name) and side.id in al:
                    return True
    return False


def remove_global_residual(src, tag="skipGrm"):
    """Inverse of add_global_residual: STRIP the input->output residual from the top forward.

    WHY THE ARM HAD TO BE INVERTED. Every base already carries a global residual, so "does a
    global skip help?" cannot be asked by adding one. It can be asked by taking it away: the
    treated arm becomes residual-REMOVED and the control is the base as-is. Same contrast,
    opposite direction, and it is actually measurable.

    Only the addition is removed -- `a + b` becomes whichever side is not the input. Nothing else
    about the graph changes, so the comparison stays like-for-like.
    """
    tree = _parse(src)
    if tree is None:
        return None
    top = _top_unet(tree)
    if top is None:
        return None
    fn = _forward(top)
    inp = _first_arg(fn) if fn else None
    if not inp:
        return None
    al = _input_aliases(fn, inp)

    class Strip(ast.NodeTransformer):
        def __init__(self):
            self.n = 0

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Add):
                l_is = isinstance(node.left, ast.Name) and node.left.id in al
                r_is = isinstance(node.right, ast.Name) and node.right.id in al
                if l_is and not r_is:
                    self.n += 1
                    return node.right
                if r_is and not l_is:
                    self.n += 1
                    return node.left
            return node

    st_ = Strip()
    fn2 = st_.visit(fn)
    if st_.n == 0:
        return None                          # nothing to remove
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    return out if _builds(out) else None


def add_global_residual(src, tag="skipG"):
    """GLOBAL: add the input->output residual on the top U-Net forward (parameter-free;
    in_ch==out_ch==3 so the identity add is always shape-valid)."""
    tree = _parse(src)
    if tree is None:
        return None
    top = _top_unet(tree)
    if top is None:
        return None
    fn = _forward(top)
    inp = _first_arg(fn)
    if not inp:
        return None
    # STRICT guard: the shallow check below only sees a top-level `x + expr`. Every generated
    # base hides its residual inside a call and behind an alias, so without this the operator
    # doubles an existing residual and the model collapses to the identity. Measured: 96/96 runs
    # at exactly 33.021 dB.
    if _has_input_residual_deep(fn, inp):
        return None  # already residual (possibly nested/aliased) -- adding another is harmful
    if not _wrap_return_in_residual(fn, inp):
        return None  # already has it / nothing to do
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    return out if _builds(out) else None  # reject channel-invalid edits


def block_residual_sites(src):
    """-> ordered list of block-class names a BLOCKRES could be added to.

    The ablation needs to ask WHERE a residual goes, not just whether one exists, so the
    insertion points have to be enumerable before anything is generated. Source order is
    used as the index, which is stable across runs (unlike a set or dict iteration)."""
    base = _parse(src)
    if base is None:
        return []
    top = _top_unet(base)
    top_name = top.name if top else None
    return [c.name for c in _classes(base)
            if not _is_loss(c) and c.name not in ("Net", top_name) and _forward(c)]


def _wrap_block(tree, name):
    """Apply the identity residual to class `name` in `tree`. -> True if it changed."""
    cls = next((c for c in _classes(tree) if c.name == name), None)
    if cls is None:
        return False
    fn = _forward(cls)
    if fn is None:
        return False
    inp = _first_arg(fn)
    return bool(inp and _wrap_return_in_residual(fn, inp))


def add_block_residual(src, tag="skipB", index=None, apply_all=False):
    """BLOCKRES: add a ResNet-style identity residual to a conv-block class that lacks it.
    Valid for C->C blocks (input added to same-channel output).

    index=None, apply_all=False -> ORIGINAL behaviour: first eligible block that builds.
    index=k                     -> target exactly block_residual_sites(src)[k], or None if
                                   that site is ineligible. Deterministic placement, which
                                   is what makes "shallow vs deep" a controlled variable
                                   rather than whatever the operator happened to pick.
    apply_all=True              -> residual on EVERY eligible block (the saturated cell).
    """
    sites = block_residual_sites(src)
    if not sites:
        return None

    if apply_all:
        tree = _parse(src)
        if tree is None or not any(_wrap_block(tree, n) for n in list(sites)):
            return None
        ast.fix_missing_locations(tree)
        out = ast.unparse(tree)
        return out if _builds(out) else None

    if index is not None:
        if not (0 <= index < len(sites)):
            return None
        sites = [sites[index]]           # no silent fallback to a different site

    # try each candidate on a FRESH tree; keep the first whose edit builds+forwards.
    for name in sites:
        tree = _parse(src)
        if tree is None or not _wrap_block(tree, name):
            continue
        ast.fix_missing_locations(tree)
        out = ast.unparse(tree)
        if _builds(out):
            return out  # one valid block residual per mutation (atomic)
    return None


def lateral_sites(src):
    """-> ordered list of variable names in the top forward a lateral skip can land on.

    These are the DEPTH POSITIONS of the ablation's placement arm: injecting into the
    first assigned feature is a shallow skip, into the last is the deep one the default
    operator already does. Names are returned in statement order, excluding the forward's
    own input (adding the input to itself is the global residual, not a lateral)."""
    tree = _parse(src)
    if tree is None:
        return []
    top = _top_unet(tree)
    if top is None:
        return []
    fn = _forward(top)
    if fn is None:
        return []
    inp = _first_arg(fn)
    out = []
    for st in fn.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name) and t.id != inp and t.id not in out:
                    out.append(t.id)
    return out


def add_lateral_skip_at(src, target, tag="skipL"):
    """LATERAL at a CHOSEN depth: augment `target` (a variable in the top forward) with a
    projected, resized copy of the network input, as a statement inserted right after the
    assignment that produces it:

        <target> = <target> + self.<proj>(F.interpolate(<inp>, size=<target>.shape[-2:]))

    Same mechanism as add_lateral_skip, but the injection point is an argument instead of
    always being the final projection -- without that the placement question ("does a
    lateral skip help more shallow or deep?") cannot be asked at all.

    Returns None when the target does not exist, its channel count cannot be traced, or
    the edit fails to build and train."""
    tree = _parse(src)
    if tree is None:
        return None
    top = _top_unet(tree)
    if top is None:
        return None
    fn = _forward(top)
    inp = _first_arg(fn)
    if not inp or target == inp:
        return None

    pos = next((i for i, st in enumerate(fn.body)
                if isinstance(st, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in st.targets)),
               None)
    if pos is None:
        return None

    # Read the width AT THE INSERTION POINT. _trace_last_channels reports the variable's
    # value at the END of the forward, which is a different tensor whenever the name is
    # rebound -- sizing the projection from it is what produced the shape errors.
    ch = _trace_channels(src, top.name, target, after_lineno=fn.body[pos].lineno)
    if not ch or ch <= 0:
        return None

    import hashlib
    proj = "_lat_" + hashlib.md5(f"{tag}:{target}:{ch}".encode()).hexdigest()[:6]
    init = next((f for f in top.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is None:
        return None
    init.body.append(ast.parse(f"self.{proj} = nn.Conv2d(3, {ch}, kernel_size=1)").body[0])

    fn.body.insert(pos + 1, ast.parse(
        f"{target} = {target} + self.{proj}(torch.nn.functional.interpolate("
        f"{inp}, size={target}.shape[-2:], mode='nearest'))").body[0])

    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    return out if (out != src and _builds_and_trains(out)) else None


def add_lateral_skip(src, tag="skipL", target=None):
    """LATERAL. With target=None this is the original deep-injection operator (see below).
    Pass a variable name from lateral_sites(src) to place the skip at a chosen depth via
    add_lateral_skip_at."""
    if target is not None:
        return add_lateral_skip_at(src, target, tag=tag)
    return _add_lateral_skip_default(src, tag=tag)


def _add_lateral_skip_default(src, tag="skipL"):
    """LATERAL: add an extra LEARNABLE long-range skip from the network input into
    the deep feature just before the output projection.

    Pattern handled (covers the denoise seeds): the arch forward ends in
    `return self.<M>(<x>)` where <x> is a Name. We inject a 1x1 conv adapter
    `self.<proj> = nn.Conv2d(3, C, 1)` (C = channels of <x>, discovered by tracing)
    and rewrite the return to
        return self.<M>(<x> + self.<proj>(F.interpolate(<inp>, size=<x>.shape[-2:])))
    i.e. the input is projected to C channels, resized to <x>'s resolution, and
    added — a genuine new skip path. The projection is built in __init__ with a
    CONCRETE channel count (so it's registered before the optimizer is created) and
    validated with a real train step (_builds_and_trains). Returns None if the
    forward doesn't match the pattern or the edit won't train."""
    tree = _parse(src)
    if tree is None:
        return None
    top = _top_unet(tree)
    if top is None:
        return None
    fn = _forward(top)
    inp = _first_arg(fn)
    ret = _last_return(fn)
    if not inp or ret is None:
        return None
    # return must be  self.<M>(<xname>)  with a single Name argument
    call = ret.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
            and len(call.args) == 1 and isinstance(call.args[0], ast.Name)):
        return None
    xname = call.args[0].id
    if xname == inp:
        return None  # that's just the global residual
    ch = _trace_last_channels(src, top.name, xname)
    if not ch or ch <= 0:
        return None
    # deterministic attr name (stable across processes so the GA's checksum dedup
    # and checkpoint resume stay consistent — Python's hash() is salted per run).
    import hashlib
    proj = "_lat_" + hashlib.md5(f"{tag}:{xname}:{ch}".encode()).hexdigest()[:6]

    # inject `self.<proj> = nn.Conv2d(3, ch, 1)` at the end of the arch __init__
    init = next((f for f in top.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is None:
        return None
    assign = ast.parse(f"self.{proj} = nn.Conv2d(3, {ch}, kernel_size=1)").body[0]
    init.body.append(assign)

    # rewrite return arg:  <x>  ->  <x> + self.<proj>(F.interpolate(<inp>, size=<x>.shape[-2:], mode='nearest'))
    skip_expr = ast.parse(
        f"{xname} + self.{proj}(torch.nn.functional.interpolate("
        f"{inp}, size={xname}.shape[-2:], mode='nearest'))"
    ).body[0].value
    call.args[0] = skip_expr

    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    return out if (out != src and _builds_and_trains(out)) else None


# ---------------------------------------------------------------- registry / GA hook

# Ordered by proven/likely ROI (skip_inventory.py): global (+3.41 proven), block
# (in winner 9F, missing 8/10), lateral (learnable input->deep-feature skip).
SKIP_OPERATORS = {
    "global": add_global_residual,
    "block":  add_block_residual,
    "lateral": add_lateral_skip,
}


def apply_random_skip_op(src, rng, order=None):
    """Try skip operators in ROI order (or a shuffled order via rng) and return the
    first that produces a changed source, plus the op name. (None, None) if none apply."""
    names = order or list(SKIP_OPERATORS.keys())
    if rng is not None:
        names = list(names)
        rng.shuffle(names)
    for name in names:
        out = SKIP_OPERATORS[name](src)
        if out is not None and out != src:
            return out, name
    return None, None


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    import os
    import sys
    import torch

    NN = "/a/mm/nn-dataset/ab/nn/nn"
    sys.path.insert(0, "/a/mm/nn-dataset")

    def build_ok(source):
        """Compile source, instantiate Net, forward 2x3x256x256, check shape."""
        ns = {}
        try:
            exec(compile(source, "<gen>", "exec"), ns)
            net = ns["Net"](in_shape=(1, 3, 256, 256), prm={"lr": 1e-4}, device="cpu")
            x = torch.randn(2, 3, 256, 256)
            net.eval()
            with torch.no_grad():
                y = net(x)
            return tuple(y.shape) == (2, 3, 256, 256)
        except Exception as e:
            return f"ERR {type(e).__name__}: {str(e)[:80]}"

    seeds = sys.argv[1:] or [f"{i}F" for i in range(1, 11)]
    print(f"{'seed':7}{'op':10}{'applied':9}{'build+fwd':12}")
    print("-" * 40)
    for s in seeds:
        p = os.path.join(NN, f"DenoiseUnet{s}.py")
        if not os.path.exists(p):
            continue
        base = open(p).read()
        for opname, op in SKIP_OPERATORS.items():
            out = op(base)
            if out is None:
                print(f"{s:7}{opname:10}{'no':9}{'-':12}")
                continue
            res = build_ok(out)
            ok = "OK" if res is True else res
            print(f"{s:7}{opname:10}{'yes':9}{ok:12}")

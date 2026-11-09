"""Channel repair v2: locate the offending conv AT RUNTIME, patch its construction site.

WHY v1 ALMOST NEVER FIRES. preflight.repair_channels matches the constructor with a regex
over integer literals -- `nn.Conv2d(32, 64, ...)`. Measured over run 3's rejects, 517 of 605
Conv2d constructions in the failing candidates use VARIABLES OR EXPRESSIONS
(`nn.Conv2d(ch, ch * 2, 3)`), 44 use literals, 3 use keywords. So the pattern is blind to
~85% of the convs it exists to fix: it fired on 8 of 434 forward-stage rejects and none of
those 8 went on to pass, while 205 of those rejects are exactly the channel-mismatch error
it targets.

WHAT v2 DOES INSTEAD. The error message names the declared weight shape and the tensor that
actually arrived, so the target is identified by SHAPE, not by text:

  1. Build the model and find every Conv2d/ConvTranspose2d whose weight is [out, want_in,
     kh, kw]. Runtime knows the real numbers no matter how the source computed them.
  2. If exactly one matches, take its qualified name (`enc1`, or `enc1.0` inside a
     Sequential) and find that attribute's assignment in the AST.
  3. Overwrite ONLY that call's in_channels argument with the literal that arrived.

Ambiguity is refused, not guessed: if two convs share the shape, the module is left alone
and the candidate keeps its original error, because patching the wrong one would produce a
model that runs but is not the architecture the generator proposed.

Nothing else is touched -- not kernel size, stride, padding, out_channels, or topology.
"""
import ast
import copy


def locate_by_execution(model, size=64):
    """Run a forward and return (qualname, got_in, want_in) for the conv that actually failed.

    SHAPE MATCHING IS THE WRONG KEY. The error names the weight shape, but in a U-Net a
    shape like [32, 32, 1, 1] belongs to several convs at once, so a shape lookup finds a
    set and has to refuse -- measured as 24 of 120 candidates declining as "ambiguous",
    and it is why the first version of this module recovered only 1.

    Execution order is unambiguous. A forward_pre_hook on every conv records the module
    about to run and the channel count it is being handed; when the forward raises, the
    LAST recorded module is the one that raised, and the hook already captured the real
    input width. No parsing of the message, no guessing between identical shapes.
    """
    import torch
    import torch.nn as nn
    seen = {}
    handles = []

    def mk(name, mod):
        def hook(m, inp):
            if inp and hasattr(inp[0], "shape") and inp[0].dim() == 4:
                seen["last"] = (name, int(inp[0].shape[1]),
                                int(m.weight.shape[1] if isinstance(m, nn.Conv2d)
                                    else m.weight.shape[0]))
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_pre_hook(mk(name, m)))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 3, size, size))
        return None                      # forward succeeded; nothing to repair
    except Exception:
        return seen.get("last")
    finally:
        for h in handles:
            h.remove()


def find_conv_sites(model, out_c, want_in, kh, kw):
    """-> list of qualified module names whose weight is exactly [out_c, want_in, kh, kw]."""
    import torch.nn as nn
    hits = []
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dim() != 4:
            continue
        s = tuple(w.shape)
        # ConvTranspose2d stores weight as [in, out, kh, kw]; Conv2d as [out, in, kh, kw]
        want = (out_c, want_in, kh, kw)
        if s == want or (isinstance(m, nn.ConvTranspose2d) and s == (want_in, out_c, kh, kw)):
            hits.append(name)
    return hits


def _conv_calls(node):
    """Yield every nn.Conv2d / nn.ConvTranspose2d Call inside an AST node."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            nm = (f.attr if isinstance(f, ast.Attribute) else
                  f.id if isinstance(f, ast.Name) else None)
            if nm in ("Conv2d", "ConvTranspose2d"):
                yield n


def _assignment_for(tree, attr):
    """Find `self.<attr> = <value>` anywhere in the tree. -> the Assign node or None."""
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if (isinstance(t, ast.Attribute) and t.attr == attr
                    and isinstance(t.value, ast.Name) and t.value.id == "self"):
                return n
    return None


def _select_call(assign, rest):
    """Pick the Conv call inside `assign` addressed by the remaining qualified-name parts.

    `self.enc1 = nn.Conv2d(...)`                 -> rest == []      -> the only call
    `self.enc1 = nn.Sequential(nn.Conv2d(...),)` -> rest == ['0']   -> Sequential arg 0
    Deeper nesting falls back to "the only conv in there", and refuses if ambiguous.
    """
    calls = list(_conv_calls(assign.value))
    if not calls:
        return None
    if not rest:
        return calls[0] if len(calls) == 1 else None
    # numeric index into a Sequential-like container
    idx = None
    for part in rest:
        if part.isdigit():
            idx = int(part)
            break
    if idx is not None and isinstance(assign.value, ast.Call):
        args = assign.value.args
        if idx < len(args):
            inner = list(_conv_calls(args[idx]))
            if len(inner) == 1:
                return inner[0]
    return calls[0] if len(calls) == 1 else None


def patch_in_channels(src, qualname, got_in):
    """Rewrite the in_channels argument of the conv at `qualname` to the literal got_in.

    -> (new_source, note) or (None, reason).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return None, f"unparseable: {e}"
    parts = qualname.split(".")
    attr, rest = parts[0], parts[1:]
    assign = _assignment_for(tree, attr)
    if assign is None:
        return None, f"no assignment found for self.{attr}"
    call = _select_call(assign, rest)
    if call is None:
        return None, f"could not isolate the conv inside self.{attr}"

    new_arg = ast.Constant(value=int(got_in))
    if call.args:                                    # positional in_channels
        old = ast.unparse(call.args[0])
        call.args[0] = new_arg
    else:                                            # keyword form
        kw = next((k for k in call.keywords if k.arg == "in_channels"), None)
        if kw is None:
            return None, "conv has neither positional nor in_channels= argument"
        old = ast.unparse(kw.value)
        kw.value = new_arg
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), f"{qualname}.in_channels {old} -> {got_in}"


def repair(src, err, build_model, size=64):
    """Full v2 repair. -> (new_source, note) or (None, reason).

    Primary path is execution-order location, which needs no error text at all and cannot
    be confused by repeated shapes. The message-parsing path is kept only as a fallback for
    the case where the hook records nothing (failure before any conv runs).
    """
    try:
        model = build_model(src)
    except Exception as e:
        return None, f"could not build model: {type(e).__name__}"

    hit = locate_by_execution(model, size)
    if hit:
        qualname, got_in, want_in = hit
        if got_in == want_in:
            return None, "conv that failed already has matching channels (not a width bug)"
        return patch_in_channels(src, qualname, got_in)

    import re
    m = re.search(r"weight of size \[(\d+), (\d+), (\d+), (\d+)\].*?"
                  r"expected input\[\d+, (\d+),", err or "", re.S)
    if not m:
        return None, "no conv reached, and the message does not name both shapes"
    out_c, want_in, kh, kw, got_in = (int(m.group(i)) for i in range(1, 6))
    if want_in == got_in:
        return None, "declared and actual channels already agree"
    sites = find_conv_sites(model, out_c, want_in, kh, kw)
    if not sites:
        return None, f"no conv with weight [{out_c},{want_in},{kh},{kw}]"
    if len(sites) > 1:
        return None, f"ambiguous: {len(sites)} convs share that shape ({', '.join(sites[:3])})"
    return patch_in_channels(src, sites[0], got_in)

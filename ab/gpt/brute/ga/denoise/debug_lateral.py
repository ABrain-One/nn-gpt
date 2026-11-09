"""Why does add_lateral_skip reject every base? Reports the FAILURE STAGE per attempt.

ablation_generate.py records only valid/invalid, which cannot distinguish "the operator
declined to emit anything" from "it emitted source that would not train" -- and those have
opposite fixes. Every L=1 cell came back 0/8, so the operator is refusing uniformly rather
than hitting a model-specific edge case.

Stages, in the order they can fail:
  PATTERN  - default operator needs the top forward to end in `return self.<M>(<name>)`
  NOSITE   - no assignable target variable for the placement variant
  TRACE    - _trace_last_channels could not determine the target's channel count
  EMIT     - operator returned None for some other reason
  TRAIN    - source emitted, but _builds_and_trains rejected it
"""
import ast
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skip_operators as SO                                     # noqa: E402


def diagnose(name, src):
    out = {"model": name}
    tree = SO._parse(src)
    top = SO._top_unet(tree) if tree else None
    if top is None:
        out["default"] = "NOTOP"
        return out
    fn = SO._forward(top)
    ret = SO._last_return(fn) if fn else None
    inp = SO._first_arg(fn) if fn else None
    out["top_class"] = top.name
    out["input_arg"] = inp
    out["return_src"] = ast.unparse(ret)[:110] if ret is not None else None

    # --- default path: needs `return self.<M>(<Name>)` -------------------------
    call = ret.value if ret is not None else None
    matches = (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
               and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
               and len(call.args) == 1 and isinstance(call.args[0], ast.Name))
    if not matches:
        out["default"] = "PATTERN"
    else:
        xname = call.args[0].id
        if xname == inp:
            out["default"] = "PATTERN(returns input)"
        else:
            ch = SO._trace_last_channels(src, top.name, xname)
            out["default"] = "TRACE" if not ch else "ok(ch=%d)" % ch

    # --- placement path: per candidate target ---------------------------------
    sites = SO.lateral_sites(src)
    out["sites"] = sites[:6]
    per = {}
    for t in sites[:4]:
        ch = SO._trace_last_channels(src, top.name, t)
        if not ch:
            per[t] = "TRACE(no channels)"
            continue
        # Rebuild the edit WITHOUT the validation gate so the two failure modes separate,
        # then run the build inline and surface the real exception -- _builds_and_trains
        # catches everything and returns False, which is exactly what hid this.
        try:
            edited = _emit_only(src, top.name, t, ch)
        except Exception as e:
            per[t] = f"EMIT-RAISED {type(e).__name__}: {e}"[:160]
            continue
        if edited is None:
            per[t] = "EMIT(returned None)"
            continue
        per[t] = _why_not_trainable(edited, ch)
    out["placements"] = per
    return out


def _emit_only(src, top_name, target, ch):
    """add_lateral_skip_at's AST edit with the build/train gate removed."""
    import hashlib
    tree = SO._parse(src)
    top = next(c for c in SO._classes(tree) if c.name == top_name)
    fn = SO._forward(top)
    inp = SO._first_arg(fn)
    pos = next((i for i, st in enumerate(fn.body)
                if isinstance(st, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == target for x in st.targets)), None)
    if pos is None:
        return None
    proj = "_lat_" + hashlib.md5(f"dbg:{target}:{ch}".encode()).hexdigest()[:6]
    init = next((f for f in top.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is None:
        return None
    init.body.append(ast.parse(f"self.{proj} = nn.Conv2d(3, {ch}, kernel_size=1)").body[0])
    fn.body.insert(pos + 1, ast.parse(
        f"{target} = {target} + self.{proj}(torch.nn.functional.interpolate("
        f"{inp}, size={target}.shape[-2:], mode='nearest'))").body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _why_not_trainable(source, ch):
    """Run the _builds_and_trains steps with the exception visible."""
    import torch
    ns = {}
    try:
        exec(compile(source, "<dbg>", "exec"), ns)
        if "Net" not in ns:
            return "NO Net CLASS"
        net = ns["Net"](in_shape=(1, 3, 64, 64), prm={"lr": 1e-4}, device="cpu")
        net.train()
        x = torch.randn(2, 3, 64, 64)
        y = net(x)
        if tuple(y.shape) != (2, 3, 64, 64):
            return f"SHAPE {tuple(y.shape)} (ch traced={ch})"
        if not torch.isfinite(y).all():
            return "NON-FINITE FORWARD"
        loss = torch.nn.functional.l1_loss(y, torch.randn_like(y))
        opt = getattr(net, "optimizer", None)
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        grads = [p for p in net.parameters() if p.grad is not None]
        proj_grads = [n for n, p in net.named_parameters() if "_lat_" in n and p.grad is not None]
        if not grads:
            return "NO GRADS AT ALL"
        if not proj_grads:
            return "PROJECTION GOT NO GRAD (not in optimizer / not on the path)"
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:170]


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "/a/mm/nn-dataset/db/ab.nn.db"
    man = json.load(open("/a/mm/sweep_out/ablation/manifest.json"))
    names = []
    for r in man:
        if r["base"] not in names:
            names.append(r["base"])
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    res = []
    for n in names[:4]:
        src = c.execute("select code from nn where name=?", (n,)).fetchone()[0]
        d = diagnose(n, src)
        res.append(d)
        print(json.dumps(d, indent=1))
    json.dump(res, open("/a/mm/sweep_out/ablation/lateral_debug.json", "w"), indent=1)


if __name__ == "__main__":
    main()

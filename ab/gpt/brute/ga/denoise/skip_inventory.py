#!/usr/bin/env python3
"""
skip_inventory.py — WHERE do skip connections exist / where are they MISSING?

Statically analyses each DenoiseUnet seed and reports which skip-connection
types it has. Purpose: identify the candidate INSERTION POINTS for the new
skip operators (a skip is "useful to add" exactly where a weak seed lacks one
that the strong seeds have). Motivating evidence: 2F lacked the global residual;
the search added it -> +3.41 dB. This tool generalises that: map the whole pool.

Skip types detected (per seed, from AST of the model's forward() + block forward()):
  GLOBAL  : global input->output residual   (return inp + <out>)   [the +3.41 dB one]
  LATERAL : encoder->decoder U-Net skips     (torch.cat with a stored skip)
  BLOCKRES: intra-block residual             (y = inp + x*..., inside a block)
  DENSE   : same skip reused at >1 decoder site (nested / UNet++-style)

Usage: python skip_inventory.py [--dir <nn-dataset ab/nn/nn>] [--seeds 1F 2F ...]
Output: a table + a per-seed list of MISSING skip types = insertion points.
"""
import argparse
import ast
import os
import re

DEFAULT_DIR = "/a/mm/nn-dataset/ab/nn/nn"
DEFAULT_SEEDS = [f"{i}F" for i in range(1, 11)]


def _is_arch_class(cls):
    """True for real architecture modules; exclude loss/criterion helpers whose
    additions (l1_loss + edge, pred-target, etc.) are NOT skip connections."""
    return isinstance(cls, ast.ClassDef) and "Loss" not in cls.name


def _arch_forwards(tree):
    """Yield (class, forward_fn) for architecture classes only (no Loss classes)."""
    for cls in ast.walk(tree):
        if not _is_arch_class(cls):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                yield cls, fn


def _added_output(binop, inp_name):
    """binop is `a + b`; a real residual adds `inp` to a PROCESSED tensor
    (a Call/Attribute/Subscript), not two scalar/loss terms. Return True if one
    operand is exactly the input name and the other is a processed-tensor expr."""
    if not isinstance(binop.op, ast.Add):
        return False
    l, r = binop.left, binop.right
    def is_inp(n):
        return isinstance(n, ast.Name) and n.id == inp_name
    def is_processed(n):
        return isinstance(n, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp))
    return (is_inp(l) and is_processed(r)) or (is_inp(r) and is_processed(l))


def _has_global_residual(tree):
    """The TOP-LEVEL forward returns `inp + <processed>` — the image-in/image-out skip.

    SCOPED TO THE TOP CLASS. This used to accept a residual in ANY architecture class, so
    an intra-block `return inp + x` registered as GLOBAL as well as BLOCKRES -- which is
    why 1,831 of 2,958 corpus models appeared to have both, and why the two factors could
    not be separated. GLOBAL means the network's own input reaches its own output; a
    residual inside a block is BLOCKRES and nothing else."""
    top = _top_class_name(tree)
    for cls, fn in _arch_forwards(tree):
        if top is not None and cls.name != top:
            continue
        args = [a.arg for a in fn.args.args if a.arg != "self"]
        if not args:
            continue
        inp_name = args[0]
        for node in ast.walk(fn):
            # THE ADD NEED NOT BE THE OUTERMOST NODE OF THE RETURN. Requiring
            # `isinstance(node.value, ast.BinOp)` missed the very common
            # `return torch.clamp(x + out, 0, 1)` and `return self.act(x + out)`, so the
            # residual had to be the last operation to count. Search the whole returned
            # expression instead.
            if isinstance(node, ast.Return) and node.value is not None:
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.BinOp) and _added_output(sub, inp_name):
                        return True
    return False


def _has_lateral_by_name(src):
    """ORIGINAL name-based detector, KEPT ONLY FOR COMPARISON. Do not use for analysis.

    It requires the literal substring "skip" inside a cat([...]), so a textbook U-Net
    written `torch.cat([d1, e1], 1)` does not register. Measured over the 2,958-model
    corpus it found 12 laterals while ~78% of models contain a torch.cat -- i.e. it was
    reporting variable-naming convention, not architecture."""
    return bool(re.search(r"(torch\.)?cat\(\s*\[[^\]]*skip", src)) or \
        bool(re.search(r"cat\(\s*\[[^\]]*skips\b", src))


# statements that change spatial resolution -- the thing a lateral skip must cross
_SCALE = re.compile(r"\b(pool|Pool|interpolate|[Uu]psample|ConvTranspose|pixel_shuffle|"
                    r"PixelShuffle|pixel_unshuffle|PixelUnshuffle|adaptive_avg|adaptive_max)\b"
                    r"|stride\s*=\s*[2-9]")


def _has_lateral(src):
    """STRUCTURAL lateral-skip detector: a tensor produced BEFORE a resolution change and
    consumed AFTER it, via concat or addition.

    That crossing is what makes a skip "lateral" (encoder->decoder) rather than a local
    fusion of two same-scale branches. Working on statement order in the forward:

      idx(assign) < idx(scale change) < idx(use)

    catches `e1 = enc(x) ... p = pool(e1) ... d = up(...) ... cat([d, e1])` regardless of
    what the variable is called, and does NOT fire for `cat([a, b])` where a and b are two
    same-resolution branches -- the case a pure "is there a cat" test gets wrong.

    Falls back to the name-based test only when the source will not parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _has_lateral_by_name(src)

    # CASE 2 FIRST: the skip arrives as a BLOCK ARGUMENT. U-Nets that factor fusion into a
    # helper -- `_UpsampleConcatBlock.forward(self, x, skip)` calling `cat([x, skip])` --
    # carry the skip across a class boundary, so no single forward contains both the
    # producer and the consumer and the statement-order test below cannot see it. A module
    # taking two feature tensors and CONCATENATING OR ADDING them is a fusion point.
    for _cls, fn in _arch_forwards(tree):
        args = [a.arg for a in fn.args.args if a.arg != "self"]
        if len(args) < 2:
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and (getattr(node.func, "attr", None) == "cat"
                         or getattr(node.func, "id", None) == "cat")):
                for arg in node.args:
                    elts = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
                    nms = {e.id for e in elts if isinstance(e, ast.Name)}
                    if len(nms & set(args)) >= 2:
                        return True
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                sides = {s.id for s in (node.left, node.right) if isinstance(s, ast.Name)}
                if len(sides & set(args)) >= 2:
                    return True

    for _cls, fn in _arch_forwards(tree):
        stmts = list(fn.body)
        assigned = {}                       # name -> index of the statement assigning it
        scale_at = []                       # indices of resolution-changing statements
        for i, st in enumerate(stmts):
            try:
                text = ast.unparse(st)
            except Exception:
                text = ""
            if _SCALE.search(text):
                scale_at.append(i)
            for t in ast.walk(st):
                if isinstance(t, ast.Assign):
                    for tgt in t.targets:
                        if isinstance(tgt, ast.Name):
                            assigned.setdefault(tgt.id, i)
                # `skips.append(e1)` stores a skip for later use
                elif (isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute)
                      and t.func.attr == "append" and isinstance(t.func.value, ast.Name)):
                    assigned.setdefault(t.func.value.id, i)

        def crosses(nm, use_i):
            a = assigned.get(nm)
            return a is not None and any(a < s < use_i for s in scale_at)

        for i, st in enumerate(stmts):
            for node in ast.walk(st):
                # concat fusion
                if (isinstance(node, ast.Call)
                        and (getattr(node.func, "attr", None) == "cat"
                             or getattr(node.func, "id", None) == "cat")):
                    for arg in node.args:
                        elts = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
                        for e in elts:
                            nm = (e.id if isinstance(e, ast.Name) else
                                  e.value.id if (isinstance(e, ast.Subscript)
                                                 and isinstance(e.value, ast.Name)) else None)
                            if nm and crosses(nm, i):
                                return True
                # additive fusion: d = up(...) + e1
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    for side in (node.left, node.right):
                        if isinstance(side, ast.Name) and crosses(side.id, i):
                            return True
    return False


def _top_class_name(tree):
    """Name of the TOP-LEVEL architecture class, via the shared (corrected) rule in
    skip_operators. Falls back to the old largest-forward heuristic if that import is
    unavailable, so this module keeps working standalone."""
    try:
        import skip_operators as _SO
        top = _SO._top_unet(tree)
        if top is not None:
            return top.name
    except Exception:
        pass
    arch = [(cls, fn) for cls, fn in _arch_forwards(tree)]
    return max(arch, key=lambda cf: sum(1 for _ in ast.walk(cf[1])))[0].name if arch else None


def _has_block_residual(tree):
    """A helper block class (not Net, not a Loss, not the top U-Net) whose forward
    adds its own input to a processed tensor — the ResNet-style intra-block skip."""
    arch = [(cls, fn) for cls, fn in _arch_forwards(tree)]
    if not arch:
        return False
    top_cls = _top_class_name(tree)
    for cls, fn in arch:
        if cls.name in ("Net", top_cls):
            continue  # skip wrapper + the top U-Net (that's the global residual)
        fargs = [a.arg for a in fn.args.args if a.arg != "self"]
        if not fargs:
            continue
        inp = fargs[0]
        for node in ast.walk(fn):
            bo = None
            if isinstance(node, ast.Return) and isinstance(node.value, ast.BinOp):
                bo = node.value
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                bo = node.value
            if bo is not None and _added_output(bo, inp):
                return True
    return False


def _has_dense(src):
    """Heuristic: a stored skip indexed/reused more than once on the decoder side
    (nested skips). Rare in our seeds; flagged when a skip var is subscripted."""
    return bool(re.search(r"skips\[[^\]]+\][\s\S]{0,400}skips\[[^\]]+\]", src))


def analyse(path):
    src = open(path).read()
    tree = ast.parse(src)
    return {
        "GLOBAL": _has_global_residual(tree),
        "LATERAL": _has_lateral(src),
        "BLOCKRES": _has_block_residual(tree),
        "DENSE": _has_dense(src),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    args = ap.parse_args()

    types = ["GLOBAL", "LATERAL", "BLOCKRES", "DENSE"]
    rows = {}
    for s in args.seeds:
        p = os.path.join(args.dir, f"DenoiseUnet{s}.py")
        if not os.path.exists(p):
            print(f"  (missing {p})")
            continue
        rows[s] = analyse(p)

    # table
    hdr = "seed   " + "".join(f"{t:>10}" for t in types) + "   MISSING (=insertion points)"
    print(hdr)
    print("-" * len(hdr))
    for s, r in rows.items():
        cells = "".join(f"{'yes' if r[t] else '--':>10}" for t in types)
        missing = [t for t in types if not r[t]]
        print(f"Unet{s:<3}{cells}   {', '.join(missing) if missing else '(full skip complement)'}")

    # summary: which skips are most commonly missing across the pool
    print("\nmost-missing skip type (best operator ROI):")
    for t in types:
        miss = [s for s, r in rows.items() if not r[t]]
        print(f"  {t:9s} missing in {len(miss):2d}/{len(rows)} seeds: {', '.join(miss) if miss else '(none)'}")


if __name__ == "__main__":
    main()

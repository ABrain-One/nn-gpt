"""Component-block-swap crossover for the denoise seed GA (arm C recombination).

Realizes Prof Dymtro's decomposition "architectures = skip connections + conv
components": the child inherits Parent A's U-Net SKELETON (skip topology, the
whole `Net` + main arch class) but with A's repeated conv BLOCK replaced by
Parent B's block internals. Because the swappable blocks share a channel-
preserving contract  Block(channels) -> C->C , the swap is channel-safe by
construction, so children are valid far more often than raw source splicing.

Swappable seeds (survey 2026-07-11) all expose a `Block(channels)` residual
block: DenoiseUnet1F/6F/7F/9F/10F. The donor block may reference helper classes
(e.g. 9F's block uses _Norm2d/_GateActivation), so we transplant the block's
DEPENDENCY CLOSURE too, renamed to avoid colliding with the host.

crossover(src_A, src_B) -> child source string, or None if A has no swappable
block or B has nothing compatible to donate. The GA still validates the emitted
child (build + forward at 2x3x256x256) before trusting it.
"""
import ast

# blocks with this exact positional __init__ signature are channel-preserving
# residual blocks, safely interchangeable.
_CONTRACT = ["channels"]


def _module_classes(tree):
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def _init_args(cls):
    init = next((f for f in cls.body if isinstance(f, ast.FunctionDef)
                 and f.name == "__init__"), None)
    if not init:
        return None
    return [a.arg for a in init.args.args if a.arg != "self"]


def _swappable_block(tree):
    """Name of the class whose __init__ is exactly (self, channels) and that
    contains Conv2d(s) — i.e. the interchangeable conv block. None if absent."""
    for cls in _module_classes(tree):
        if cls.name == "Net":
            continue
        args = _init_args(cls)
        if args != _CONTRACT:
            continue
        has_conv = any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Conv2d"
                       for n in ast.walk(cls))
        if has_conv:
            return cls.name
    return None


def _referenced_names(node):
    """Top-level identifiers referenced anywhere in node (Names + Call/Base ids)."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            names.add(n.value.id)
    return names


def _closure(tree, root_name):
    """Top-level class/func defs (in this module) that root_name needs,
    transitively, excluding root itself. Used to carry a donor block's helpers."""
    defs = {n.name: n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    seen, stack = set(), [root_name]
    while stack:
        cur = stack.pop()
        node = defs.get(cur)
        if node is None:
            continue
        for ref in _referenced_names(node):
            if ref in defs and ref != cur and ref not in seen and ref != root_name:
                seen.add(ref)
                stack.append(ref)
    return [defs[n] for n in defs if n in seen]


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name in self.m:
            node.name = self.m[node.name]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.m:
            node.name = self.m[node.name]
        return node


def crossover(src_a, src_b, tag="Xb"):
    """Child = A's skeleton with A's conv block swapped for B's (renamed to A's
    block name; B's helper closure carried in, prefixed by `tag` to avoid
    collisions). Returns child source, or None if incompatible."""
    try:
        ta, tb = ast.parse(src_a), ast.parse(src_b)
    except SyntaxError:
        return None
    a_block = _swappable_block(ta)
    b_block = _swappable_block(tb)
    if not a_block or not b_block:
        return None

    # donor = B's block + its dependency closure, from B's AST
    b_defs = {n.name: n for n in tb.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    if b_block not in b_defs:
        return None
    deps = _closure(tb, b_block)
    a_names = {n.name for n in ta.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}

    # rename map: donor block -> A's block name; each helper -> tag-prefixed
    mapping = {b_block: a_block}
    for d in deps:
        new = f"{tag}_{d.name}"
        # avoid the (unlikely) case a prefixed name still collides with A
        while new in a_names or new in mapping.values():
            new = "_" + new
        mapping[d.name] = new

    import copy
    donor_nodes = [copy.deepcopy(b_defs[b_block])] + [copy.deepcopy(d) for d in deps]
    renamer = _Renamer(mapping)
    donor_nodes = [renamer.visit(n) for n in donor_nodes]
    for n in donor_nodes:
        ast.fix_missing_locations(n)

    # build child: A's body with a_block's ClassDef replaced by the donor nodes
    new_body, replaced = [], False
    for node in ta.body:
        if isinstance(node, ast.ClassDef) and node.name == a_block:
            new_body.extend(donor_nodes)   # insert transplanted block+deps here
            replaced = True
        else:
            new_body.append(node)
    if not replaced:
        return None
    ta.body = new_body
    # DEAD-CODE PRUNE: keep only top-level defs reachable from Net (+ non-def statements
    # like imports). Without this, each crossover leaves the PREVIOUS donor's closure
    # helpers behind as dead code, so the source grows unboundedly across generations
    # (measured: 5 -> 13 top-level defs over 11 crossovers) and a long run eventually
    # deadlocks (Run 2 hung ~gen 29). Pruning keeps the child the size of a single model.
    keep = {d.name for d in _closure(ta, "Net")} | {"Net"}
    ta.body = [n for n in ta.body
               if not isinstance(n, (ast.ClassDef, ast.FunctionDef)) or n.name in keep]
    ast.fix_missing_locations(ta)
    try:
        child = ast.unparse(ta)
    except Exception:
        return None
    if child == src_a:
        return None
    return child


if __name__ == "__main__":
    # tiny self-check on the swappable seeds
    import itertools, os
    NN = "/a/mm/nn-dataset/ab/nn/nn"
    seeds = ["DenoiseUnet1F", "DenoiseUnet6F", "DenoiseUnet7F",
             "DenoiseUnet9F", "DenoiseUnet10F"]
    srcs = {s: open(os.path.join(NN, s + ".py")).read() for s in seeds}
    for a, b in itertools.permutations(seeds, 2):
        c = crossover(srcs[a], srcs[b])
        ok = c is not None and c != srcs[a]
        parses = False
        if c:
            try:
                ast.parse(c); parses = True
            except SyntaxError:
                pass
        print(f"  {a:14s} x {b:14s} -> {'child' if ok else 'None':6s} parses={parses} "
              f"len={len(c) if c else 0}")

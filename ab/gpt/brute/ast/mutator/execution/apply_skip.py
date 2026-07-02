"""Skip-connection AST edits.

Rewrites a network's `forward()` return statement to add or remove a global
input->output residual connection:

    add:    return <expr>            ->   return <input_arg> + (<expr>)
    remove: return <input_arg> + X   ->   return X

For image-to-image models (denoising) the forward input and output are both
(B, 3, H, W), so the residual is always shape-safe with no projection. This is
the mutator's only `forward()`-dataflow operator (all others edit constructor
kwargs), so it operates on a `Return` node rather than a `Call` node.
"""
import ast


def _is_global_residual(ret_value: ast.AST, input_arg: str) -> bool:
    """True if ret_value is `input_arg + X` or `X + input_arg`."""
    if isinstance(ret_value, ast.BinOp) and isinstance(ret_value.op, ast.Add):
        for side in (ret_value.left, ret_value.right):
            if isinstance(side, ast.Name) and side.id == input_arg:
                return True
    return False


def apply_skip_modification(node: ast.Return, mod: dict) -> ast.Return:
    """Apply a global-residual add/remove to a `Return` node in place.

    mod keys:
      action    : 'add' | 'remove'
      input_arg : name of the forward() input argument (e.g. 'inp' or 'x')
    """
    action = mod.get("action", "add")
    input_arg = mod.get("input_arg")
    if node.value is None or not input_arg:
        return node

    if action == "add":
        if _is_global_residual(node.value, input_arg):
            return node  # already residual; no-op
        node.value = ast.BinOp(
            left=ast.Name(id=input_arg, ctx=ast.Load()),
            op=ast.Add(),
            right=node.value,
        )
    elif action == "remove":
        v = node.value
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
            if isinstance(v.left, ast.Name) and v.left.id == input_arg:
                node.value = v.right
            elif isinstance(v.right, ast.Name) and v.right.id == input_arg:
                node.value = v.left

    ast.fix_missing_locations(node)
    return node

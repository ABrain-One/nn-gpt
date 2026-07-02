"""Skip-connection planner.

Plans a global input->output residual toggle on the model's inner network
(the class assigned to `self.net` in `Net.__init__`):

  * no residual present  -> action 'add'    (return inp + <expr>)
  * residual present     -> action 'remove' (return <expr>)

This is a `forward()`-dataflow mutation, so unlike the other planners it does
not target a submodule/constructor: it parses `source_code` directly and emits a
`source_location` pointing at the forward's `return` statement. The plan is keyed
by "<InnerClass>.forward" (a descriptive key, not a real submodule name).
"""
import ast
import json
from typing import Dict, Any, Optional, Tuple

from mutator import config


class SkipPlanner:
    def __init__(self, model_planner):
        self.model_planner = model_planner

    # --- source analysis helpers -------------------------------------------
    def _tree(self) -> Optional[ast.Module]:
        src = getattr(self.model_planner, "source_code", None)
        if not src:
            return None
        try:
            return ast.parse(src)
        except SyntaxError:
            return None

    @staticmethod
    def _inner_net_class(tree: ast.Module) -> Optional[str]:
        """Class assigned to `self.net = <Class>(...)` inside class Net."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Net":
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                            and isinstance(sub.targets[0], ast.Attribute)
                            and sub.targets[0].attr == "net"
                            and isinstance(sub.value, ast.Call)):
                        f = sub.value.func
                        return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        return None

    @staticmethod
    def _forward_return(tree: ast.Module, class_name: str) -> Tuple[Optional[str], Optional[ast.Return]]:
        """(input_arg_name, last top-level return node) of class_name.forward."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        input_arg = args[0] if args else None
                        returns = [n for n in item.body if isinstance(n, ast.Return)]
                        return input_arg, (returns[-1] if returns else None)
        return None, None

    @staticmethod
    def _has_global_residual(ret: ast.Return, input_arg: str) -> bool:
        v = ret.value
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
            for side in (v.left, v.right):
                if isinstance(side, ast.Name) and side.id == input_arg:
                    return True
        return False

    def _analyze(self):
        """Return (class_name, input_arg, ret_node, has_residual) or None."""
        tree = self._tree()
        if tree is None:
            return None
        cls = self._inner_net_class(tree)
        if not cls:
            return None
        input_arg, ret = self._forward_return(tree, cls)
        if not input_arg or ret is None or ret.value is None:
            return None
        return cls, input_arg, ret, self._has_global_residual(ret, input_arg)

    # --- public API --------------------------------------------------------
    def can_apply(self) -> bool:
        return self._analyze() is not None

    def plan_skip_mutation(self) -> Dict[str, Any]:
        info = self._analyze()
        if info is None:
            if config.DEBUG_MODE:
                print("[SkipPlanner] No inner-net forward return found; skipping.")
            return {}
        cls, input_arg, ret, has_res = info
        action = "remove" if has_res else "add"

        plan = {
            f"{cls}.forward": {
                "mutation_type": "skip",
                "action": action,
                "input_arg": input_arg,
                "target_class": cls,
                "source_location": {"lineno": ret.lineno, "col_offset": ret.col_offset},
            }
        }
        self.model_planner.plan = plan
        if config.DEBUG_MODE:
            print(f"[SkipPlanner] {cls}.forward global-residual -> {action} (input='{input_arg}')")
            print(json.dumps(plan, indent=2))
        return plan

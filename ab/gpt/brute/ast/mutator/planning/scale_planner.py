"""
Scale planner — Phase 2 architectural/capacity mutation for denoising U-Nets.

Unlike the component planners (activation/kernel/stride), which tweak single leaf layers,
this grows the *capacity/topology* of a sub-network by editing scalar architectural
hyperparameters passed to a custom nn.Module constructor, e.g.:

    self.net = SomeUNet(num_filters=32, depth=4, mid_blocks=2)

- width params  (num_filters, base_channels, dim, width, ...) -> multiplied (×1.5 / ×2)
- depth params  (depth, blocks_per_level, mid_blocks, num_blocks, ...) -> incremented (+1/+2)

This biases the search toward LARGER models and changes receptive field / capacity, which
is far more likely to move PSNR than component swaps. The change is realised as a normal
keyword-arg edit (routed through CodeMutator.schedule_modification), and the emitted-code
validator in the orchestrator rejects any growth that breaks the network.
"""

import ast
import json
import random
from typing import Any, Dict, List, Tuple

from mutator import config


class ScalePlanner:
    def __init__(self, model_planner):
        self.model_planner = model_planner

    def _collect_candidates(self) -> List[Tuple[str, dict, str, int, str]]:
        """Find scalable architectural kwargs on sub-network constructor calls.

        The source map only tracks leaf nn.* modules, NOT the top-level custom sub-network
        call (e.g. `self.net = SomeUNet(num_filters=32)`) where the architectural knobs
        live. So we parse the original source STRING and match `self.<attr> = Call(...)`
        assignments whose <attr> is a real submodule (so orchestrator.get_submodule works).

        Returns list of (module_name, source_location, arg_name, current_value, kind).
        """
        sc = getattr(config, "SCALE_MUTATIONS", {})
        width_names = set(sc.get("width_params", []))
        depth_names = set(sc.get("depth_params", []))

        source_code = getattr(self.model_planner, "source_code", None)
        if not source_code:
            return []
        try:
            tree = ast.parse(source_code)
        except Exception:
            return []

        valid_submodules = {n for n, _ in self.model_planner.original_model.named_modules() if n}

        candidates: List[Tuple[str, dict, str, int, str]] = []
        for node in ast.walk(tree):
            # match:  self.<attr> = <Call>(...)
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                continue
            attr = tgt.attr
            if attr not in valid_submodules:
                continue
            call = node.value
            if not isinstance(call, ast.Call) or not hasattr(call, "keywords"):
                continue
            loc = {"lineno": call.lineno, "col_offset": call.col_offset,
                   "end_lineno": getattr(call, "end_lineno", None),
                   "end_col_offset": getattr(call, "end_col_offset", None)}
            for kw in call.keywords:
                if kw.arg is None:
                    continue
                # integer literal kwargs only (skip bools, lists, expressions)
                if (isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, int)
                        and not isinstance(kw.value.value, bool)):
                    if kw.arg in width_names:
                        candidates.append((attr, loc, kw.arg, kw.value.value, "width"))
                    elif kw.arg in depth_names:
                        candidates.append((attr, loc, kw.arg, kw.value.value, "depth"))
        return candidates

    def _grow(self, current: int, kind: str) -> int:
        sc = getattr(config, "SCALE_MUTATIONS", {})
        lo = sc.get("min_value", 1)
        if kind == "width":
            factor = random.choice(sc.get("width_factors", [1.5, 2.0]))
            new = int(round(current * factor))
            new = max(4, (new // 4) * 4)  # keep channel counts multiple-of-4 friendly
            hi = sc.get("width_max", 512)
        else:  # depth
            delta = random.choice(sc.get("depth_deltas", [1, 2]))
            new = current + delta
            hi = sc.get("depth_max", 10)
        return max(lo, min(new, hi))

    def plan_scale_mutation(self) -> Dict[str, Any]:
        viable: List[Tuple[str, dict, str, int, int, str]] = []
        for name, loc, arg, cur, kind in self._collect_candidates():
            new = self._grow(cur, kind)
            if new != cur:
                viable.append((name, loc, arg, cur, new, kind))

        if not viable:
            if config.DEBUG_MODE:
                print("[ScalePlanner] No scalable architectural kwargs found.")
            return {}

        name, loc, arg, cur, new, kind = random.choice(viable)
        plan = {
            name: {
                "mutation_type": "scale",
                "arg_name": arg,
                "new_value": new,
                "source_location": loc,
            }
        }
        self.model_planner.plan = plan
        if config.DEBUG_MODE:
            print(f"[ScalePlanner] Scale {kind}: {arg} {cur} -> {new} for '{name}'")
            print(json.dumps(plan, indent=2))
        return plan

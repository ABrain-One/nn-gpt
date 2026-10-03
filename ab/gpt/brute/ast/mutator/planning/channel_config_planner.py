"""
Channel-configuration ("bracket") mutation planner.

Models such as AirNet keep their per-stage channel widths in a configuration
container inside the ``Net`` class, e.g.::

    channels = [64, 128, 256, 512]
    init_block_channels = 64

and then build the blocks from that container inside a loop / builder function::

    for i, out_channels in enumerate(channels):
        layers.append(AirUnit(in_channels=..., out_channels=out_channels, ...))

The direct dimension planner edits the ``nn.Conv2d`` / ``nn.Linear`` call site.
For models whose layers live inside reusable block classes that call site is
shared by every block, so the edit cannot change one stage independently.

This planner instead mutates one entry of the *configuration container* so the
change flows through the loop to exactly one stage. That keeps every consumer
(the residual downsample, the batch norm, the classifier) consistent with the
new width and yields valid, independently parameterised architectures.
"""

import ast
import random
from typing import Any, Dict, List, Optional

from mutator import config


class ChannelConfigPlanner:
    """Plans mutations of channel-configuration containers."""

    def __init__(self, model_planner, source_code: Optional[str] = None):
        self.model_planner = model_planner
        self.source_code = source_code or ""
        try:
            self.tree = ast.parse(self.source_code) if self.source_code.strip() else None
        except SyntaxError:
            self.tree = None

    def has_candidates(self) -> bool:
        return bool(self.tree is not None and self._find_candidates())

    def plan_channel_config_mutation(self) -> Dict[str, Any]:
        """Return a single-entry plan mutating one channel-config value."""
        if self.tree is None:
            return {}

        candidates = self._find_candidates()
        if not candidates:
            if config.DEBUG_MODE:
                print("[ChannelConfigPlanner] No channel-config candidates found")
            return {}

        candidate = random.choice(candidates)
        possible_new_values = [
            size for size in self.model_planner.VALID_CHANNEL_SIZES
            if size != candidate["old_value"]
        ]
        if not possible_new_values:
            return {}

        new_value = random.choice(possible_new_values)
        details = {
            "mutation_type": "channel_config",
            "name": candidate["name"],
            "kind": candidate["kind"],          # 'list' | 'scalar'
            "index": candidate["index"],        # int | None
            "old_value": candidate["old_value"],
            "new_value": new_value,
            "symbolic": False,
            "source_location": candidate["location"],
        }
        if config.DEBUG_MODE:
            print(f"[ChannelConfigPlanner] Plan: {details}")

        # Single-entry plan keyed by a sentinel name so the orchestrator's
        # generic plan.values() handling keeps working.
        return {"__channel_config__": details}

    # ------------------------------------------------------------------ #
    def _find_candidates(self) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                name = target.id if isinstance(target, ast.Name) else None
                if not name or not self._name_matches(name):
                    continue
                candidates.extend(self._value_candidates(name, node.value))
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                name = target.id if isinstance(target, ast.Name) else None
                if not name or node.value is None or not self._name_matches(name):
                    continue
                candidates.extend(self._value_candidates(name, node.value))
        return candidates

    @staticmethod
    def _name_matches(name: str) -> bool:
        low = name.lower()
        return any(pattern in low for pattern in config.CHANNEL_CONFIG_NAME_PATTERNS)

    def _value_candidates(self, name: str, value: ast.AST) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(value, (ast.List, ast.Tuple)) and value.elts and all(
            isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool)
            for e in value.elts
        ):
            for index, element in enumerate(value.elts):
                out.append({
                    "name": name,
                    "kind": "list",
                    "index": index,
                    "old_value": element.value,
                    "location": self._loc(value),   # the container node
                })
        elif isinstance(value, ast.Constant) and isinstance(value.value, int) and not isinstance(value.value, bool):
            out.append({
                "name": name,
                "kind": "scalar",
                "index": None,
                "old_value": value.value,
                "location": self._loc(value),       # the Constant node
            })
        return out

    @staticmethod
    def _loc(node: ast.AST) -> Dict[str, int]:
        lineno = getattr(node, "lineno", -1)
        return {
            "lineno": lineno,
            "end_lineno": getattr(node, "end_lineno", lineno),
            "col_offset": getattr(node, "col_offset", -1),
            "end_col_offset": getattr(node, "end_col_offset", -1),
        }

"""Apply channel-configuration (bracket / scalar) modifications to AST nodes."""

import ast
from typing import Dict

from mutator import config


def apply_channel_config_modification(node: ast.AST, mod: Dict) -> bool:
    """Replace a value inside a channel-config container.

    ``node`` is either the ``ast.List``/``ast.Tuple`` container (bracket mode) or
    the ``ast.Constant`` scalar value (scalar mode).

    Returns True when a replacement was made.
    """
    new_value = mod.get("new_value")
    if new_value is None:
        return False

    if isinstance(node, (ast.List, ast.Tuple)):
        index = mod.get("index")
        if index is None or index < 0 or index >= len(node.elts):
            return False
        node.elts[index] = ast.Constant(value=new_value)
        if config.DEBUG_MODE:
            print(f"  > Modified channel-config list '{mod.get('name')}'[{index}] -> {new_value}")
        return True

    if isinstance(node, ast.Constant):
        node.value = new_value
        if config.DEBUG_MODE:
            print(f"  > Modified channel-config scalar '{mod.get('name')}' -> {new_value}")
        return True

    return False

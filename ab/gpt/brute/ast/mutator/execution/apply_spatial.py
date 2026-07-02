"""Apply spatial (kernel size, stride) modifications to AST nodes."""

import ast
from mutator import config


def apply_kernel_size_modification(node: ast.Call, mod: dict) -> None:
    new_k = mod['new_kernel_size']
    modified = False
    for kw in node.keywords:
        if kw.arg == 'kernel_size':
            kw.value = ast.Constant(value=new_k)
            modified = True
            break
    if not modified:
        if len(node.args) > 2:
            node.args[2] = ast.Constant(value=new_k)
            modified = True
    # Keep spatial size constant by adjusting padding to k//2 (valid 'same' for odd
    # kernels at stride 1). Without this, a kernel change shifts the feature-map size
    # and breaks U-Net skip-connection concatenations downstream.
    if modified:
        new_pad = new_k // 2
        pad_set = False
        for kw in node.keywords:
            if kw.arg == 'padding':
                kw.value = ast.Constant(value=new_pad)
                pad_set = True
                break
        if not pad_set:
            if len(node.args) > 4:  # positional padding (in_ch, out_ch, k, stride, padding)
                node.args[4] = ast.Constant(value=new_pad)
            else:
                node.keywords.append(ast.keyword(arg='padding', value=ast.Constant(value=new_pad)))
    if modified and config.DEBUG_MODE:
        print(f"  > Modified kernel_size to {new_k} (padding -> {new_k // 2})")


def apply_stride_modification(node: ast.Call, mod: dict) -> None:
    modified = False
    for kw in node.keywords:
        if kw.arg == 'stride':
            kw.value = ast.Constant(value=mod['new_stride'])
            modified = True
            break
    if not modified:
        if len(node.args) > 3:
            node.args[3] = ast.Constant(value=mod['new_stride'])
            modified = True
    if modified and config.DEBUG_MODE:
        print(f"  > Modified stride to {mod['new_stride']}")

"""Static architecture features for the denoise accuracy predictor.

Pure AST + cost + name features — NO instantiation, NO training, sub-millisecond.
This is the cheap half of the surrogate's input (see zero_cost.py for the
one-fwd+bwd proxies). Everything here is derived from three things the search
already has in hand for every candidate: the model SOURCE, its measured MACs and
params (the GA cost pass computes these), and its lineage NAME (encodes how the
GA built it: base seed(s), crossover 'X', mutation '+m', skip-op '+s').

The design bet (validated by skip_inventory.py): denoise quality tracks skip
TOPOLOGY (cat = lateral, add = residual, mul = gating) and block composition
(conv/norm/activation mix), which are all visible in the AST without running the
model. We count them structurally so the regressor can learn the mapping.

API: extract(source, macs=None, params=None, name=None) -> dict[str,float]
     FEATURE_NAMES  (stable ordered list, for building the design matrix)
"""
import ast
import math
import re

# nn.Module leaf types we count (attribute name after `nn.`)
_LAYER_KINDS = [
    "Conv2d", "ConvTranspose2d", "Linear",
    "BatchNorm2d", "GroupNorm", "LayerNorm", "InstanceNorm2d",
    "ReLU", "LeakyReLU", "GELU", "SiLU", "PReLU", "ELU", "Sigmoid", "Tanh",
    "PixelShuffle", "PixelUnshuffle", "Upsample",
    "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
    "Dropout", "Softmax", "Identity",
]

# forward()-level data-flow ops that signal skip topology / fusion
_FLOW_FEATS = ["n_cat", "n_add", "n_mul", "n_interpolate", "n_chunk", "n_getitem"]

FEATURE_NAMES = (
    [f"n_{k}" for k in _LAYER_KINDS]
    + _FLOW_FEATS
    + [
        "n_layers_total", "n_classes", "n_functiondefs",
        "n_init_assigns", "n_forward_stmts", "src_lines",
        "kern_max", "kern_mean", "n_conv_k1", "n_conv_k3", "n_conv_kbig",
        "ch_max", "ch_mean",
        "macs", "params", "log_macs", "log_params", "macs_per_param",
        # lineage (from name): how the GA constructed this candidate
        "lin_crossovers", "lin_mutations", "lin_skipops", "lin_depth",
    ]
)


def _num_const(node):
    """Best-effort scalar from an AST arg node (int constant or first tuple elt)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        return _num_const(node.elts[0])
    return None


def _lineage(name):
    """Parse the GA lineage name into build-op counts.
    e.g. 'DenoiseUnet7FXDenoiseUnet9F+m+s' -> crossovers=1, muts=1, skips=1."""
    if not name:
        return 0.0, 0.0, 0.0, 0.0
    crossovers = name.count("X")            # block-crossover joins
    mutations = len(re.findall(r"\+m", name))
    skipops = len(re.findall(r"\+s", name))
    depth = crossovers + mutations + skipops  # total edits from a raw seed
    return float(crossovers), float(mutations), float(skipops), float(depth)


def extract(source, macs=None, params=None, name=None):
    """Return the full static feature dict for one architecture.
    Robust: a parse failure yields the cost/name features with zeroed structure
    (so a candidate is never dropped from the matrix on a syntax quirk)."""
    feats = {k: 0.0 for k in FEATURE_NAMES}

    # --- cost features (from the GA's own cost pass) ---
    if macs:
        feats["macs"] = float(macs)
        feats["log_macs"] = math.log(float(macs) + 1.0)
    if params:
        feats["params"] = float(params)
        feats["log_params"] = math.log(float(params) + 1.0)
    if macs and params:
        feats["macs_per_param"] = float(macs) / (float(params) + 1.0)

    # --- lineage features (from name) ---
    (feats["lin_crossovers"], feats["lin_mutations"],
     feats["lin_skipops"], feats["lin_depth"]) = _lineage(name)

    # --- AST structural features ---
    try:
        tree = ast.parse(source)
    except Exception:
        return feats
    feats["src_lines"] = float(source.count("\n") + 1)

    layer_set = set(_LAYER_KINDS)
    kernels, channels = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            feats["n_classes"] += 1
        elif isinstance(node, ast.FunctionDef):
            feats["n_functiondefs"] += 1
            if node.name == "forward":
                feats["n_forward_stmts"] += float(len(node.body))
            if node.name == "__init__":
                feats["n_init_assigns"] += float(
                    sum(isinstance(s, (ast.Assign, ast.AnnAssign)) for s in ast.walk(node)))
        elif isinstance(node, ast.Call):
            fn = node.func
            attr = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if attr in layer_set:
                feats[f"n_{attr}"] += 1.0
                feats["n_layers_total"] += 1.0
                if attr in ("Conv2d", "ConvTranspose2d"):
                    # positional (in_ch, out_ch, kernel_size, ...) best-effort
                    args = node.args
                    if len(args) >= 2:
                        ci, co = _num_const(args[0]), _num_const(args[1])
                        if ci:
                            channels.append(ci)
                        if co:
                            channels.append(co)
                    k = None
                    if len(args) >= 3:
                        k = _num_const(args[2])
                    for kw in node.keywords:
                        if kw.arg == "kernel_size":
                            k = _num_const(kw.value)
                        elif kw.arg == "in_channels":
                            v = _num_const(kw.value)
                            if v:
                                channels.append(v)
                        elif kw.arg == "out_channels":
                            v = _num_const(kw.value)
                            if v:
                                channels.append(v)
                    if k:
                        kernels.append(k)
                        if k == 1:
                            feats["n_conv_k1"] += 1.0
                        elif k == 3:
                            feats["n_conv_k3"] += 1.0
                        else:
                            feats["n_conv_kbig"] += 1.0
            # data-flow / fusion ops (torch.cat, F.interpolate, .chunk, *, +)
            elif attr in ("cat", "concat"):
                feats["n_cat"] += 1.0
            elif attr == "interpolate":
                feats["n_interpolate"] += 1.0
            elif attr == "chunk":
                feats["n_chunk"] += 1.0
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                feats["n_add"] += 1.0
            elif isinstance(node.op, ast.Mult):
                feats["n_mul"] += 1.0
        elif isinstance(node, ast.Subscript):
            feats["n_getitem"] += 1.0

    if kernels:
        feats["kern_max"] = max(kernels)
        feats["kern_mean"] = sum(kernels) / len(kernels)
    if channels:
        feats["ch_max"] = max(channels)
        feats["ch_mean"] = sum(channels) / len(channels)
    return feats


def vector(source, macs=None, params=None, name=None):
    """Feature dict -> ordered list aligned to FEATURE_NAMES (design-matrix row)."""
    f = extract(source, macs, params, name)
    return [f[k] for k in FEATURE_NAMES]


if __name__ == "__main__":
    import sys
    src = open(sys.argv[1]).read()
    f = extract(src, name=sys.argv[2] if len(sys.argv) > 2 else None)
    for k in FEATURE_NAMES:
        if f[k]:
            print(f"  {k:20} {f[k]}")

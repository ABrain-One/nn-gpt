"""Vocabulary-free architecture embedding — the unified distinctness algorithm.

Two operating points from ONE computation (per Dr. Ignatov's spec):
  * exact()  -> Weisfeiler-Lehman graph hash for LITERAL clone rejection (exact equality).
  * embed()  -> continuous-WL vector; cosine = GRADED structural similarity for near-dups.

Both are derived from a single VOCABULARY-FREE featurisation of the torch.fx-traced graph:
node identity = op-type name (hashed) + ALL introspected numeric hyperparameters (hashed),
so there is NO hand-authored operation list — an unseen custom layer contributes features
with zero code change. This is what makes it a general NNGPT method (arbitrary heterogeneous
source code) rather than a denoising-specific label template.

Design decisions (all measured; see metric_ablation/DECISION_LEDGER.md):
  * acquisition by tracing -> invariant to names / style / loops / config plumbing.
  * vocabulary-free feature hashing (blake2b, stable across processes) -> deterministic;
    Python's built-in hash() is per-process salted and must NOT be used.
  * CHANNEL-INCLUSIVE: _numeric_items reads in/out_channels generically, so a width-variant
    is a distinct computation (different exact hash) and a near-dup gradedly (cosine ~0.999).
  * exact hash uses the LOSSLESS canonical descriptor string (no feature-hash collisions on
    identity); the embedding uses the feature-HASHED fixed-size vector of the same descriptor
    -> the exact hash is literally a truncation of the same featurisation.

Rejected alternatives (measured): residual-CKA / seeded-CKA / seed-averaged CKA / 100-step
proxy CKA / empirical-NTK all fail on reconstruction U-Nets (functional signals saturate);
module-signature bag is fooled by width; textual MinHash is style-sensitive + redundant.
See DECISION_LEDGER.md rows 1-11.
"""
import hashlib
import numpy as np

# feature-hashing dims. Sized for the FULL NNGPT op vocabulary (measured in metric_ablation/
# collision_test.py: 52 ecosystem archs -> 77 op-identity + 168 hyperparam-key strings; at
# D=64 the collision rate was 43%/65%). D_OP=512/D_HP=1024 -> ~4%/9% collisions AND a BETTER
# discrimination margin (denoise separation +0.064 -> +0.087). Cost is negligible (fixed-size
# vector, dot-product comparison). For a small single-task vocab, 64 would suffice.
D_OP, D_HP = 512, 1024


def _bucket(s, d):
    """Stable hash -> bucket. blake2b (NOT Python's per-process-salted hash())."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % d


def _numeric_items(obj):
    """Generic hyperparameter extraction: every public scalar / tuple-of-scalar attribute.
    Vocabulary-free — a new custom layer's hyperparameters are captured with no code change."""
    for name, val in vars(obj).items():
        if name.startswith("_"):
            continue
        if isinstance(val, bool):
            yield name, float(val)
        elif isinstance(val, (int, float)):
            yield name, float(val)
        elif isinstance(val, (tuple, list)) and val and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in val):
            for i, v in enumerate(val):
                yield f"{name}[{i}]", float(v)


def _descriptor(node, modules):
    """Vocabulary-free canonical node descriptor: (op_key, sorted numeric hyperparameters).
    Shared source of BOTH operating points."""
    if node.op == "call_module":
        m = modules.get(node.target)
        key = f"module:{type(m).__name__}"
        items = tuple(sorted(_numeric_items(m))) if m is not None else ()
    elif node.op in ("call_function", "call_method"):
        key = f"{node.op}:{getattr(node.target, '__name__', str(node.target))}"
        items = tuple(sorted((f"arg{i}", float(a)) for i, a in enumerate(node.args)
                             if isinstance(a, (int, float)) and not isinstance(a, bool)))
    else:
        key, items = node.op, ()
    return key, items


def _exact_label(node, modules):
    """Lossless canonical string for exact WL relabelling (no feature-hash collisions)."""
    key, items = _descriptor(node, modules)
    return key + "|" + ";".join(f"{n}={v:g}" for n, v in items)


def _node_features(node, modules):
    """Feature-hashed fixed-size vector of the SAME descriptor (NAME-based; retained for the
    legacy path and for comparison). Superseded by the behavioural-class encoding below."""
    key, items = _descriptor(node, modules)
    op_vec, hp_vec = np.zeros(D_OP), np.zeros(D_HP)
    op_vec[_bucket(f"{node.op}|{key}", D_OP)] = 1.0
    for name, val in items:
        hp_vec[_bucket(f"{key}.{name}", D_HP)] += np.sign(val) * np.log1p(abs(val))
    struct = np.array([np.log1p(len(node.all_input_nodes)), np.log1p(len(node.users))])
    return np.concatenate([op_vec, hp_vec, struct])


# ===================== BEHAVIOURAL-CLASS ENCODING (the open-vocabulary core) ============
# Prior art fails the open-vocabulary formulation two ways, both MEASURED here:
#   * fixed-vocabulary encoders (arch2vec/CATE/NAR-Former) cannot represent an unseen op;
#   * hashing the op NAME accepts unseen ops but encodes IDENTITY, not FUNCTION — an
#     activation swap (ReLU->GELU, 0.989) read as a BIGGER change than adding a whole U-Net
#     level (0.988), and a 3x3->5x5 conv was nearly invisible (0.998).
# We therefore never read an op name. Each node's BEHAVIOURAL CLASS is derived from what it
# DOES, via shape-propagated tracing, and that class is hashed. Consequences hold BY
# CONSTRUCTION: a never-before-seen custom layer is classified by behaviour; functionally
# equivalent ops collapse to the same bucket (act swap -> 1.000); different mechanisms stay
# orthogonal, so discrimination is preserved/improved (Lite-3F 0.846 -> 0.802,
# Lite-9F 0.658 -> 0.664) while the semantic ordering is restored.
D_CLS = 256
_RATIO_C = np.linspace(-4, 4, 9)


def _srbf(v):
    z = np.clip(np.log2(max(float(v), 1e-6)), -4, 4)
    return np.exp(-((z - _RATIO_C) ** 2) / (2 * (1.1 ** 2)))


def _shapes_of(gm, device, hw=64):
    """Propagate shapes so each node's behaviour is observable. Training-free."""
    from torch.fx.passes.shape_prop import ShapeProp
    import torch
    ShapeProp(gm).propagate(torch.randn(1, 3, hw, hw).to(device))
    out = {}
    for n in gm.graph.nodes:
        t = n.meta.get("tensor_meta")
        if t is not None and hasattr(t, "shape"):
            out[n.name] = tuple(t.shape)
    return out


def _behaviour(node, modules, shapes):
    """(class_key, continuous behavioural features) — derived purely from behaviour."""
    out = shapes.get(node.name)
    ins = [shapes.get(i.name) for i in node.all_input_nodes]
    ins = [s for s in ins if s is not None]
    cin = ins[0][1] if ins and len(ins[0]) == 4 else 0
    cout = out[1] if out is not None and len(out) == 4 else 0
    hin = ins[0][2] if ins and len(ins[0]) == 4 else 0
    hout = out[2] if out is not None and len(out) == 4 else 0
    mod = modules.get(node.target) if node.op == "call_module" else None
    npar = sum(p.numel() for p in mod.parameters(recurse=False)) if mod is not None else 0
    k = getattr(mod, "kernel_size", 1) if mod is not None else 1
    k = (k[0] if isinstance(k, (tuple, list)) else k) or 1
    grp = getattr(mod, "groups", 1) if mod is not None else 1
    parm = "P" if npar > 0 else "0"
    spat = "S" if (hin and hout and hin != hout) else ("s" if hin == hout else "?")
    chan = "same" if cin == cout else ("up" if cout > cin else "down")
    mix = "dw" if (grp > 1 and grp == cin) else ("grp" if grp > 1 else "full")
    merge = "bin" if len(node.all_input_nodes) > 1 else "un"
    kk = "1" if k == 1 else ("3" if k <= 3 else "5+")
    key = f"{parm}|{spat}|{chan}|{mix}|{merge}|k{kk}"
    macs = npar * max(hout, 1) * max(hout, 1)
    feats = np.concatenate([
        _rbf(cin), _rbf(cout), _srbf(cout / max(cin, 1)),
        _srbf(max(hout, 1) / max(hin, 1)),
        _rbf(npar), _rbf(macs), _rbf(k), _rbf(grp),
        np.array([1.0 if (npar == 0 and cin == cout and hin == hout) else 0.0,
                  1.0 if (npar > 0 and cin and cout) else 0.0,
                  1.0 if (hin and hout and hin != hout) else 0.0,
                  1.0 if npar > 0 else 0.0,
                  min(len(node.all_input_nodes), 3) / 3.0]),
    ])
    return key, feats


def _behav_node_features(node, modules, shapes):
    cls = np.zeros(D_CLS)
    key, feats = _behaviour(node, modules, shapes)
    cls[_bucket(key, D_CLS)] = 1.0                       # behavioural-CLASS bucket
    feats = feats / (np.linalg.norm(feats) + 1e-9)
    return np.concatenate([cls, 0.5 * feats])


def _trace(model):
    from torch.fx import symbolic_trace
    gm = symbolic_trace(model)
    return gm, dict(gm.named_modules()), list(gm.graph.nodes)


def _adjacency(nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for n in nodes:
        for inp in n.all_input_nodes:
            A[idx[inp], idx[n]] = 1.0
    return A


def wl_embed(A, X, iters=3, alpha=0.5):
    """Continuous Weisfeiler-Lehman: neighbourhood averaging (soft relabelling), pooled at
    every depth. Permutation-invariant, fixed-size, deterministic, training-free unit vector.
    NOTE: this is the OP-VOCABULARY half only — it pools mean/max over nodes, so it is
    TOPOLOGY-BLIND (measured: U-Nets of depth 2 vs 5 scored 0.984). embed() augments it with
    the structural block below."""
    As = A + A.T                                   # symmetric for propagation
    deg = np.maximum(As.sum(1, keepdims=True), 1.0)
    H, pooled = X, [X.mean(0), X.max(0)]
    for _ in range(iters):
        H = alpha * H + (1 - alpha) * (As @ H) / deg
        pooled += [H.mean(0), H.max(0)]
    v = np.concatenate(pooled)
    return v / (np.linalg.norm(v) + 1e-9)


# --- global TOPOLOGY block (fixes the measured topology-blindness of pooled WL) ---
# Scalars are RBF-encoded so different VALUES point in different DIRECTIONS. Encoding them
# raw fails: raw positive scalars stay collinear under normalisation, so depth-2 and depth-5
# came out MORE similar (0.993), not less. With RBF the depth-2 vs depth-5 cosine drops
# 0.984 -> 0.903 at BETA=0.4 while reload stays 1.0000, a pure width-variant stays ~1.0
# (width is not topology) and family separation is preserved (Lite-3F 0.846, Lite-9F 0.658).
_RBF_C = np.linspace(0.0, 6.0, 14)                 # centres over log1p(value)
_RBF_S = 6.0 / 13 * 1.1
BETA = 0.4                                         # topology weight vs op-vocabulary


def _rbf(v):
    z = np.log1p(max(float(v), 0.0))
    return np.exp(-((z - _RBF_C) ** 2) / (2 * _RBF_S ** 2))


def _struct_vec_graph(A):
    """RBF-encoded global graph descriptor: size, longest-path DEPTH, skip merges/branches,
    edge count, max degrees. (fx emits nodes in topological order, so the longest path is a
    single forward pass.)"""
    n = A.shape[0]
    indeg, outdeg = A.sum(0), A.sum(1)
    d = np.zeros(n)
    for j in range(n):
        pr = np.nonzero(A[:, j])[0]
        d[j] = (d[pr].max() + 1) if len(pr) else 0
    scal = [n, d.max(), (indeg >= 2).sum(), (outdeg >= 2).sum(), A.sum(),
            outdeg.max(), indeg.max()]
    return np.concatenate([_rbf(s) for s in scal])


def _combine(pooled, struct, beta=None):
    b = BETA if beta is None else beta
    p = pooled / (np.linalg.norm(pooled) + 1e-9)
    s = struct / (np.linalg.norm(struct) + 1e-9)
    v = np.concatenate([(1 - b) * p, b * s])
    return v / (np.linalg.norm(v) + 1e-9)


def embed(model, beta=None):
    """Continuous-WL over BEHAVIOURAL-CLASS node features + global topology block
    (graded operating point). cosine = dot product. Vocabulary-free: no op name is read."""
    gm, modules, _n = _trace(model)
    device = next(model.parameters()).device
    shapes = _shapes_of(gm, device)
    nodes = list(gm.graph.nodes)
    A = _adjacency(nodes)
    X = np.stack([_behav_node_features(n, modules, shapes) for n in nodes])
    return _combine(wl_embed(A, X), _struct_vec_graph(A), beta)


def exact(model):
    """Exact WL graph hash (literal-clone operating point) — a truncation of the same
    vocabulary-free featurisation (lossless canonical labels instead of hashed vectors)."""
    import networkx as nx
    _gm, modules, nodes = _trace(model)
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.name, op=_exact_label(n, modules))
        for inp in n.all_input_nodes:
            G.add_edge(inp.name, n.name)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="op")


# --- source-string convenience wrappers (load via the repo's lemur loader) ---
def _load(source):
    from mutator.lemur_io import load_lemur_model
    return load_lemur_model(source)


def embed_source(source):
    m = _load(source)
    try:
        return embed(m)
    finally:
        _cleanup(m)


def exact_source(source):
    m = _load(source)
    try:
        return exact(m)
    finally:
        _cleanup(m)


def both_source(source):
    """Return (exact_hash, embedding) — both operating points in one trace."""
    m = _load(source)
    try:
        _gm, modules, nodes = _trace(m)
        import networkx as nx
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n.name, op=_exact_label(n, modules))
            for inp in n.all_input_nodes:
                G.add_edge(inp.name, n.name)
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr="op")
        shapes = _shapes_of(_gm, next(m.parameters()).device)
        nodes = list(_gm.graph.nodes)
        A = _adjacency(nodes)
        X = np.stack([_behav_node_features(n, modules, shapes) for n in nodes])
        return h, _combine(wl_embed(A, X), _struct_vec_graph(A))
    except Exception:
        return None, None
    finally:
        _cleanup(m)


def _cleanup(m):
    info = getattr(m, "_temp_module_info", None)
    if info:
        try:
            from mutator.lemur_io import cleanup_temp_module
            cleanup_temp_module(*info)
        except Exception:
            pass

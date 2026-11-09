import ast
import hashlib
import inspect
import textwrap

import numpy as np

D_OP, D_HP = 256, 256
W_EFFECT = 0.5


def _bucket(s: str, d: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % d


def _scale(v: float) -> float:
    return float(np.sign(v) * np.log1p(abs(v)))


def _tail(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


class _Graph:
    def __init__(self):
        self.keys, self.hps, self.edges = [], [], []

    def add(self, kind: str, key: str = "") -> int:
        self.keys.append(f"{kind}|{key}")
        self.hps.append({})
        return len(self.keys) - 1

    def param(self, i: int, slot: str, val: float):
        self.hps[i][slot] = self.hps[i].get(slot, 0.0) + _scale(val)

    def link(self, src, dst, port: int = 0, w: float = 1.0):
        if src is not None and dst is not None:
            self.edges.append((src, dst, port, w))


class _Dataflow(ast.NodeVisitor):
    def __init__(self, g: _Graph, attrs: dict, uses: dict):
        self.g = g
        self.attrs = attrs
        self.uses = uses
        self.env = {}
        self.effect = None

    def _bind(self, target, node):
        if isinstance(target, ast.Name):
            self.env[target.id] = node
        elif isinstance(target, ast.Attribute):
            self.attrs.setdefault(target.attr, []).append(node)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._bind(t, node)

    def _defined(self, node) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.env
        if isinstance(node, ast.Attribute):
            return node.attr in self.attrs or node.attr in self.uses
        return False

    def _read(self, node):
        if isinstance(node, ast.Name):
            return self.env.get(node.id) or self.g.add("free")
        if isinstance(node, ast.Attribute):
            i = self.g.add("use")
            self.uses.setdefault(node.attr, []).append(i)
            return i
        return None

    def expr(self, node):
        folded = self._fold(node)
        if folded is not None:
            return folded
        if isinstance(node, ast.Call):
            f = node.func
            if self._defined(f):
                i = self.g.add("call")
                self.g.link(self._read(f), i, 0)
            else:
                i = self.g.add("call", _tail(f))
                if isinstance(f, ast.Attribute) and self._defined(f.value):
                    self.g.link(self._read(f.value), i, 0)
                elif isinstance(f, (ast.Subscript, ast.Call)):
                    self.g.link(self.expr(f), i, 0)
            for k, a in enumerate(node.args):
                self._arg(i, a, k, slot=f"a{k}")
            for kw in node.keywords:
                self._arg(i, kw.value, len(node.args), slot=kw.arg or "kw")
            return i
        if isinstance(node, (ast.Name, ast.Attribute)):
            if self._defined(node) or isinstance(node, ast.Attribute):
                return self._read(node)
            return self.g.add("free", _tail(node) if isinstance(node, ast.Name) else "")
        if isinstance(node, ast.BinOp):
            i = self.g.add("op", type(node.op).__name__)
            self.g.link(self.expr(node.left), i, 0)
            self.g.link(self.expr(node.right), i, 1)
            return i
        if isinstance(node, ast.Subscript):
            i = self.g.add("index")
            self.g.link(self.expr(node.value), i, 0)
            return i
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            i = self.g.add("pack")
            for k, e in enumerate(node.elts):
                self.g.link(self.expr(e), i, k)
            return i
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            i = self.g.add("loop")
            for c in node.generators:
                self._bind(c.target, i)
                self.g.link(self.expr(c.iter), i, 0)
            self.g.link(self.expr(node.elt), i, 1)
            return i
        if isinstance(node, ast.IfExp):
            i = self.g.add("branch")
            for k, sub in enumerate((node.test, node.body, node.orelse)):
                self.g.link(self.expr(sub), i, k)
            return i
        i = self.g.add("expr", type(node).__name__)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.g.link(self.expr(child), i, 0)
        return i

    def _fold(self, node):
        try:
            val = ast.literal_eval(node)
        except Exception:
            return None
        if isinstance(val, str):
            return self.g.add("const", val)
        if isinstance(val, bool) or isinstance(val, (int, float)):
            i = self.g.add("const")
            self.g.param(i, "v", float(val))
            return i
        if isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val
        ):
            i = self.g.add("const")
            for k, v in enumerate(val):
                self.g.param(i, f"v{k}", float(v))
            return i
        return None

    def _arg(self, call, node, port, slot):
        try:
            val = ast.literal_eval(node)
        except Exception:
            self.g.link(self.expr(node), call, port)
            return
        if isinstance(val, bool) or isinstance(val, (int, float)):
            self.g.param(call, slot, float(val))
        elif isinstance(val, str):
            self.g.param(call, f"{slot}={val}", 1.0)
        elif isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val
        ):
            for k, v in enumerate(val):
                self.g.param(call, f"{slot}[{k}]", float(v))
        else:
            self.g.link(self.expr(node), call, port)

    def visit_Assign(self, node):
        v = self.expr(node.value)
        for t in node.targets:
            self._bind(t, v)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._bind(node.target, self.expr(node.value))

    def visit_AugAssign(self, node):
        i = self.g.add("op", type(node.op).__name__)
        self.g.link(self.expr(node.target), i, 0)
        self.g.link(self.expr(node.value), i, 1)
        self._bind(node.target, i)

    def visit_Expr(self, node):
        i = self.expr(node.value)
        self.g.link(self.effect, i, 0, W_EFFECT)
        self.effect = i

    def visit_Return(self, node):
        i = self.g.add("output")
        if node.value is not None:
            self.g.link(self.expr(node.value), i, 0)

    def _loop(self, node):
        i = self.g.add("loop")
        if getattr(node, "iter", None) is not None:
            self.g.link(self.expr(node.iter), i, 0)
            self._bind(node.target, i)
        if getattr(node, "test", None) is not None:
            self.g.link(self.expr(node.test), i, 0)
        outer, self.effect = self.effect, i
        for s in node.body + getattr(node, "orelse", []):
            self.visit(s)
        self.g.link(self.effect, i, 1, W_EFFECT)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)

    visit_For = visit_While = _loop

    def visit_If(self, node):
        i = self.g.add("branch")
        self.g.link(self.expr(node.test), i, 0)
        outer, self.effect = self.effect, i
        for s in node.body + node.orelse:
            self.visit(s)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)

    def visit_With(self, node):
        i = self.g.add("ctx")
        for it in node.items:
            self.g.link(self.expr(it.context_expr), i, 0)
            if it.optional_vars is not None:
                self._bind(it.optional_vars, i)
        outer, self.effect = self.effect, i
        for s in node.body:
            self.visit(s)
        self.effect = outer
        self.g.link(outer, i, 0, W_EFFECT)

    def visit_FunctionDef(self, node):
        for s in node.body:
            self.visit(s)


def source_graph(cls):
    g, attrs, uses = _Graph(), {}, {}
    root = cls.__module__.split(".")[0]
    seen, fns = set(), []
    for klass in inspect.getmro(cls):
        if klass.__module__.split(".")[0] != root:
            continue
        for _, f in vars(klass).items():
            if inspect.isfunction(f) and id(f) not in seen:
                seen.add(id(f))
                fns.append(f)
    for f in fns:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        except (OSError, SyntaxError, IndentationError):
            continue
        _Dataflow(g, attrs, uses).visit(tree)
    for name, consumers in uses.items():
        for producer in attrs.get(name, []):
            for c in consumers:
                g.link(producer, c, 0)
    return g


def featurize(g: _Graph):
    n = len(g.keys)
    X = np.zeros((n, D_OP + D_HP + 4))
    indeg, outdeg = np.zeros(n), np.zeros(n)
    for s, d, _, _ in g.edges:
        outdeg[s] += 1
        indeg[d] += 1
    for i, key in enumerate(g.keys):
        X[i, _bucket(key, D_OP)] = 1.0
        for slot, val in g.hps[i].items():
            X[i, D_OP + _bucket(f"{key}.{slot}", D_HP)] += val
        X[i, -4:] = [np.log1p(indeg[i]), np.log1p(outdeg[i]), 1.0, 0.0]
    return X


def adjacency(g: _Graph):
    n = len(g.keys)
    F = np.zeros((n, n))
    for s, d, port, w in g.edges:
        F[s, d] += w / (1.0 + port)
    return F, F.T


def wl_embed(F, B, X, iters=3, alpha=0.5):
    dF = np.maximum(F.sum(1, keepdims=True), 1.0)
    dB = np.maximum(B.sum(1, keepdims=True), 1.0)
    H, pooled = X, [X.mean(0), X.max(0)]
    for _ in range(iters):
        H = alpha * H + (1 - alpha) * 0.5 * ((F @ H) / dF + (B @ H) / dB)
        pooled += [H.mean(0), H.max(0)]
    v = np.concatenate(pooled)
    return v / (np.linalg.norm(v) + 1e-9)


def embed_class(cls):
    g = source_graph(cls)
    return wl_embed(*adjacency(g), featurize(g))


def wl_hash(g: _Graph, iters=4) -> str:
    lab = {i: hashlib.blake2b(
        (k + repr(sorted(g.hps[i].items()))).encode(), digest_size=8).hexdigest()
        for i, k in enumerate(g.keys)}
    inc = {i: [] for i in range(len(g.keys))}
    for s, d, port, w in g.edges:
        inc[d].append((port, w, s))
    for _ in range(iters):
        lab = {i: hashlib.blake2b(
            (lab[i] + repr(sorted((p, w, lab[s]) for p, w, s in inc[i]))).encode(),
            digest_size=8).hexdigest()
            for i in range(len(g.keys))}
    return hashlib.blake2b(repr(sorted(lab.values())).encode(), digest_size=16).hexdigest()

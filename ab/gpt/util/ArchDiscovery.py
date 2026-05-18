import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_LEGACY_PATTERN_NAMES = {
    "Parallel_Triple",
    "Backbone_A_to_Fractal",
    "Backbone_B_to_Fractal",
    "Dual_Backbone_Fuse_Then_Fractal",
    "Fractal_Then_Dual_Backbone",
    "Split_Stem_Parallel_Fuse",
}


@dataclass
class ExprInfo:
    text: str
    depth: int = 0
    merges: int = 0
    max_fan_in: int = 1
    backbone_calls: int = 0
    fractal_calls: int = 0
    stem_calls: int = 0
    project_calls: int = 0
    fuse_calls: int = 0
    module_labels: Set[str] = field(default_factory=set)

    def merged_with(self, label: str, children: Sequence["ExprInfo"], *, fan_in: int = 1) -> "ExprInfo":
        depth = 1 + max((child.depth for child in children), default=0)
        merges = sum(child.merges for child in children)
        max_fan_in = max([fan_in] + [child.max_fan_in for child in children])
        backbone_calls = sum(child.backbone_calls for child in children)
        fractal_calls = sum(child.fractal_calls for child in children)
        stem_calls = sum(child.stem_calls for child in children)
        project_calls = sum(child.project_calls for child in children)
        fuse_calls = sum(child.fuse_calls for child in children)
        module_labels = set().union(*(child.module_labels for child in children))
        module_labels.add(label)

        if label.startswith("Backbone["):
            backbone_calls += 1
        if "Fractal" in label:
            fractal_calls += 1
        if label.startswith("Stem[") or label == "Stem":
            stem_calls += 1
        if label.startswith("Project[") or label in {"Project", "Bridge", "Adapter"}:
            project_calls += 1
        if label in {"Cat", "Stack", "Add", "Fuse", "Gate"} or label.startswith(("Fuse[", "Gate[")):
            merges += 1
            fuse_calls += 1

        return ExprInfo(
            text=self.text,
            depth=depth,
            merges=merges,
            max_fan_in=max_fan_in,
            backbone_calls=backbone_calls,
            fractal_calls=fractal_calls,
            stem_calls=stem_calls,
            project_calls=project_calls,
            fuse_calls=fuse_calls,
            module_labels=module_labels,
        )


@dataclass
class GraphInfo:
    pattern_name: str
    normalized_pattern_name: str
    suggested_pattern_name: str
    graph_expr: str
    graph_hash: str
    family_id: str
    family_expr: str
    family_hash: str
    cnn_expr: str
    cnn_signature: str
    descriptor_key: str
    depth: int
    merges: int
    max_fan_in: int
    backbone_calls: int
    fractal_calls: int
    stem_calls: int
    project_calls: int
    fuse_calls: int
    module_labels: Tuple[str, ...]
    is_plain_parallel_triple: bool
    has_custom_pattern_name: bool
    is_legacy_pattern_name: bool
    parse_ok: bool

    @property
    def signature(self) -> str:
        return f"{self.normalized_pattern_name}_{self.graph_hash[:6]}"


def normalize_pattern_name(name: Optional[str]) -> str:
    if not name:
        return "OpenMotif"
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not cleaned:
        return "OpenMotif"
    if cleaned[0].isdigit():
        cleaned = f"Motif_{cleaned}"
    return cleaned


def ensure_pattern_name(init_code: str, pattern_name: str) -> str:
    replacement = f"self.pattern = '{pattern_name}'"
    if re.search(r"self\.pattern\s*=\s*['\"][^'\"]+['\"]", init_code):
        return re.sub(
            r"self\.pattern\s*=\s*['\"][^'\"]+['\"]",
            replacement,
            init_code,
            count=1,
        )

    lines = init_code.splitlines()
    insert_idx = 1
    body_indent = " " * 8

    for idx, line in enumerate(lines[1:], start=1):
        if not line.strip():
            continue
        body_indent = line[: len(line) - len(line.lstrip())] or body_indent
        if "super().__init__()" in line:
            insert_idx = idx + 1
            break
        insert_idx = idx
        break

    lines.insert(insert_idx, f"{body_indent}{replacement}")
    return "\n".join(lines)


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualname(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_keyword_str(node: ast.Call, key: str) -> Optional[str]:
    for kw in node.keywords:
        if kw.arg == key:
            return _const_str(kw.value)
    return None


def _infer_attr_role(attr_name: str) -> Optional[str]:
    lowered = attr_name.lower()
    if lowered.startswith("backbone"):
        return "Backbone"
    if "stem" in lowered:
        return "Stem"
    if any(token in lowered for token in ("fuse", "mixer", "merge", "gate")):
        return "Fuse"
    if any(token in lowered for token in ("project", "adapter", "align", "bridge")):
        return "Project"
    if "classif" in lowered or lowered == "head":
        return "Classifier"
    return None


def _canonical_ctor_from_call(node: ast.Call, attr_name: str = "") -> str:
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"to", "float", "cuda", "cpu"}:
        inner = node.func.value
        if isinstance(inner, ast.Call):
            return _canonical_ctor_from_call(inner, attr_name=attr_name)

    func_name = _qualname(node.func)
    short_name = func_name.split(".")[-1]

    if short_name == "TorchVision":
        model_name = _call_keyword_str(node, "model")
        if model_name is None and node.args:
            model_name = _const_str(node.args[0])
        label = f"Backbone[{model_name or 'unknown'}]"
    elif short_name == "Sequential":
        child_names = []
        for arg in node.args:
            if isinstance(arg, ast.Call):
                child_names.append(_qualname(arg.func).split(".")[-1])
        child_sig = "+".join(child_names[:4]) if child_names else "Block"
        label = f"Sequential[{child_sig}]"
    elif short_name in {"Linear", "Conv2d", "ConvTranspose2d", "MaxPool2d", "AdaptiveAvgPool2d"}:
        label = short_name
    elif short_name:
        label = short_name
    else:
        label = "Module"

    role = _infer_attr_role(attr_name)
    if role and not label.startswith(f"{role}[") and label != role and not label.startswith("Backbone["):
        return f"{role}[{label}]"
    return label


def _extract_attr_types(init_code: str) -> Dict[str, str]:
    try:
        tree = ast.parse(init_code)
    except Exception:
        return {}

    attr_types: Dict[str, str] = {}
    fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if fn is None:
        return attr_types

    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Attribute):
            continue
        target = node.targets[0]
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            continue
        attr_name = target.attr
        value = node.value
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Attribute) and value.func.attr in {"to", "float", "cuda", "cpu"}:
                wrapped = value.func.value
                if isinstance(wrapped, ast.Attribute) and isinstance(wrapped.value, ast.Name) and wrapped.value.id == "self":
                    if wrapped.attr in attr_types:
                        attr_types[attr_name] = attr_types[wrapped.attr]
                        continue
                if isinstance(wrapped, ast.Call):
                    attr_types[attr_name] = _canonical_ctor_from_call(wrapped, attr_name=attr_name)
                    continue

            if _qualname(value.func).endswith("convert_sync_batchnorm") and value.args:
                arg0 = value.args[0]
                if isinstance(arg0, ast.Attribute) and isinstance(arg0.value, ast.Name) and arg0.value.id == "self":
                    if arg0.attr in attr_types:
                        attr_types[attr_name] = attr_types[arg0.attr]
                        continue
                if isinstance(arg0, ast.Call):
                    attr_types[attr_name] = _canonical_ctor_from_call(arg0, attr_name=attr_name)
                    continue

            attr_types[attr_name] = _canonical_ctor_from_call(value, attr_name=attr_name)
        else:
            role = _infer_attr_role(attr_name)
            if role:
                attr_types[attr_name] = role

    return attr_types


def _leaf_info(text: str) -> ExprInfo:
    return ExprInfo(text=text, depth=0, module_labels={text})


def _expr_info(node: ast.AST, env: Dict[str, ExprInfo], attr_types: Dict[str, str]) -> ExprInfo:
    if isinstance(node, ast.Name):
        if node.id == "x":
            return _leaf_info("Input")
        return env.get(node.id, _leaf_info(f"Var[{node.id}]"))

    if isinstance(node, ast.Constant):
        return _leaf_info(repr(node.value))

    if isinstance(node, ast.List):
        items = [_expr_info(item, env, attr_types) for item in node.elts]
        text = "[" + ", ".join(item.text for item in items) + "]"
        return ExprInfo(text=text, depth=max((item.depth for item in items), default=0), module_labels=set().union(*(item.module_labels for item in items)))

    if isinstance(node, ast.Tuple):
        items = [_expr_info(item, env, attr_types) for item in node.elts]
        text = "(" + ", ".join(item.text for item in items) + ")"
        return ExprInfo(text=text, depth=max((item.depth for item in items), default=0), module_labels=set().union(*(item.module_labels for item in items)))

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _expr_info(node.left, env, attr_types)
        right = _expr_info(node.right, env, attr_types)
        info = ExprInfo(text=f"Add({left.text}, {right.text})")
        return info.merged_with("Add", [left, right], fan_in=2)

    if isinstance(node, ast.Call):
        func_name = _qualname(node.func)

        if func_name == "adaptive_pool_flatten":
            child = _expr_info(node.args[0], env, attr_types) if node.args else _leaf_info("Unknown")
            info = ExprInfo(text=f"PoolFlat({child.text})")
            return info.merged_with("PoolFlat", [child])

        if func_name in {"torch.cat", "cat"}:
            parts = []
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                parts = [_expr_info(item, env, attr_types) for item in node.args[0].elts]
            else:
                parts = [_expr_info(arg, env, attr_types) for arg in node.args]
            info = ExprInfo(text=f"Cat({', '.join(part.text for part in parts)})")
            return info.merged_with("Cat", parts, fan_in=max(2, len(parts)))

        if func_name == "torch.stack":
            parts = []
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                parts = [_expr_info(item, env, attr_types) for item in node.args[0].elts]
            info = ExprInfo(text=f"Stack({', '.join(part.text for part in parts)})")
            return info.merged_with("Stack", parts, fan_in=max(2, len(parts)))

        if isinstance(node.func, ast.Attribute) and node.func.attr in {"flatten", "view", "reshape", "contiguous", "float", "to"}:
            child = _expr_info(node.func.value, env, attr_types)
            return child

        if isinstance(node.func, ast.Attribute) and node.func.attr == "mean":
            child = _expr_info(node.func.value, env, attr_types)
            info = ExprInfo(text=f"Mean({child.text})")
            return info.merged_with("Mean", [child])

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            attr_name = node.func.attr
            label = attr_types.get(attr_name, attr_name)
            children = [_expr_info(arg, env, attr_types) for arg in node.args]
            info = ExprInfo(text=f"{label}({', '.join(child.text for child in children)})")
            return info.merged_with(label, children, fan_in=max(1, len(children)))

        children = [_expr_info(arg, env, attr_types) for arg in node.args]
        label = func_name.split(".")[-1] if func_name else "Call"
        info = ExprInfo(text=f"{label}({', '.join(child.text for child in children)})")
        return info.merged_with(label, children, fan_in=max(1, len(children)))

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        label = attr_types.get(node.attr, node.attr)
        return _leaf_info(label)

    return _leaf_info(type(node).__name__)


def _looks_like_plain_parallel_triple(
    forward_code: str,
    *,
    graph_expr: str = "",
    info: Optional[ExprInfo] = None,
) -> bool:
    squashed = re.sub(r"\s+", " ", forward_code)
    checks = [
        "self.features(x)",
        "self.backbone_a(x)",
        "self.backbone_b(x)",
        "torch.cat([",
        "self.classifier(",
    ]
    if all(token in squashed for token in checks):
        return True

    if not info or not graph_expr:
        return False

    compact = re.sub(r"\s+", "", graph_expr)
    classifier_head = compact.startswith("classifier(Cat(") or (
        compact.startswith("Classifier[") and "(Cat(" in compact
    )
    pure_parallel_fuse = (
        info.backbone_calls == 3
        and info.project_calls == 0
        and info.stem_calls == 0
        and info.fractal_calls == 0
        and info.merges == 1
        and info.max_fan_in == 3
        and compact.count("Backbone[") == 3
        and "Cat(" in compact
        and classifier_head
    )
    return pure_parallel_fuse


def _build_descriptor_key(info: ExprInfo) -> str:
    return "|".join(
        [
            f"d{min(info.depth, 9)}",
            f"m{min(info.merges, 6)}",
            f"bb{min(info.backbone_calls, 4)}",
            f"fr{min(info.fractal_calls, 4)}",
            f"st{min(info.stem_calls, 4)}",
            f"pr{min(info.project_calls, 4)}",
            f"fu{min(info.fuse_calls, 4)}",
            f"fan{min(info.max_fan_in, 6)}",
        ]
    )


def _split_top_level_args(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
        elif ch == "," and paren_depth == 0 and bracket_depth == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_family_tree(text: str):
    text = text.strip()
    if not text:
        return ""
    open_idx = text.find("(")
    if open_idx <= 0 or not text.endswith(")"):
        return text

    name = text[:open_idx].strip()
    inner = text[open_idx + 1:-1]
    children = [_parse_family_tree(part) for part in _split_top_level_args(inner)]
    return (name, children)


def _canonical_family_label(label: str) -> str:
    clean = re.sub(r"\s+", "", label)
    if clean.startswith("Var[") or clean in {"Input", "Var"}:
        return "Var"
    if clean.startswith("Backbone[") or clean == "Backbone":
        return "Backbone"
    if clean.startswith("Project[") or clean in {"Project", "Bridge", "Adapter"}:
        return "Project"
    if clean.startswith("Stem[") or clean == "Stem":
        return "Stem"
    if clean.startswith("Fuse[") or clean == "Fuse":
        return "Fuse"
    if clean.startswith("Gate[") or clean == "Gate":
        return "Gate"
    if clean.startswith("Classifier[") or clean.lower() == "classifier":
        return "Classifier"
    if clean.startswith("Sequential["):
        inner = clean[len("Sequential["):-1] if clean.endswith("]") else clean
        if "Fractal" in inner:
            return "Fractal"
        if "Stem" in inner:
            return "Stem"
        if any(token in inner for token in ("Project", "Bridge", "Adapter", "Align")):
            return "Project"
        return "Block"
    if clean == "Sequential":
        return "Block"
    if "Fractal" in clean:
        return "Fractal"
    if clean in {"PoolFlat", "Mean"}:
        return "Pool"
    return clean


def _render_canonical_family_tree(node) -> str:
    if isinstance(node, str):
        return _canonical_family_label(node)

    label, children = node
    canon_label = _canonical_family_label(label)
    canon_children = [_render_canonical_family_tree(child) for child in children]
    canon_children = [child for child in canon_children if child]

    if canon_label == "Pool":
        return canon_children[0] if canon_children else "Pool"
    if canon_label in {"Cat", "Add"}:
        canon_children = sorted(canon_children)

    if not canon_children:
        return canon_label
    return f"{canon_label}({', '.join(canon_children)})"


def _build_family_expr(graph_expr: str) -> str:
    try:
        family_expr = _render_canonical_family_tree(_parse_family_tree(graph_expr))
    except Exception:
        family_expr = graph_expr
        family_expr = re.sub(r"(Backbone|Project|Stem|Fuse|Gate|Sequential|Classifier)\[[^\]]+\]", r"\1", family_expr)
        family_expr = re.sub(r"Var\[[^\]]+\]", "Var", family_expr)
    family_expr = re.sub(r"\s+", " ", family_expr).strip()
    return family_expr


def _render_canonical_cnn_tree(node) -> str:
    if isinstance(node, str):
        canon_label = _canonical_family_label(node)
        if canon_label == "Classifier":
            return ""
        if canon_label == "Pool":
            return "Pool"
        if canon_label == "Var":
            return "Input"
        return canon_label

    label, children = node
    canon_label = _canonical_family_label(label)
    canon_children = [_render_canonical_cnn_tree(child) for child in children]
    canon_children = [child for child in canon_children if child]

    if canon_label in {"Classifier", "Pool"}:
        if len(canon_children) == 1:
            return canon_children[0]
        if canon_children:
            return f"Ctx({', '.join(canon_children)})"
        return ""
    if canon_label in {"Cat", "Add", "Stack"}:
        canon_children = sorted(canon_children)
    if not canon_children:
        return canon_label
    return f"{canon_label}({', '.join(canon_children)})"


def _build_cnn_expr(graph_expr: str) -> str:
    try:
        cnn_expr = _render_canonical_cnn_tree(_parse_family_tree(graph_expr))
    except Exception:
        cnn_expr = graph_expr
    cnn_expr = re.sub(r"\s+", " ", str(cnn_expr or "")).strip()
    return cnn_expr or "IncompleteCNN"


def _build_family_id(
    info: ExprInfo,
    *,
    parse_ok: bool,
    is_plain_parallel_triple: bool,
) -> str:
    if not parse_ok:
        return "Incomplete"
    if is_plain_parallel_triple:
        return "ParallelTriple_Shallow"

    if info.backbone_calls >= 3 and info.max_fan_in >= 3 and info.merges == 1 and info.fuse_calls == 1:
        if info.project_calls == 0 and info.stem_calls == 0 and info.fractal_calls == 0:
            return "TripleBackboneFuse_Shallow"
    if info.backbone_calls == 2 and info.merges == 1 and info.fuse_calls == 1:
        if info.project_calls == 0 and info.stem_calls == 0 and info.fractal_calls == 0:
            return "DualBackboneFuse_Shallow"

    parts: List[str] = []
    if info.stem_calls:
        parts.append("Stem")
    if info.project_calls:
        parts.append("Project")
    if info.fractal_calls:
        parts.append("Fractal")

    if info.backbone_calls >= 3:
        parts.append("TriBackbone")
    elif info.backbone_calls == 2:
        parts.append("DualBackbone")
    elif info.backbone_calls == 1:
        parts.append("SingleBackbone")

    if info.fuse_calls or info.merges:
        parts.append("Fuse")
    if info.max_fan_in >= 3:
        parts.append("Wide")
    if info.depth >= 5:
        parts.append("Deep")

    if not parts:
        parts.append("OpenFamily")
    return "_".join(parts[:5])


def suggest_pattern_name(graph_hash: str, info: ExprInfo) -> str:
    parts: List[str] = []
    if info.stem_calls:
        parts.append("Stem")
    if info.backbone_calls >= 2:
        parts.append("DualBackbone")
    elif info.backbone_calls == 1:
        parts.append("SingleBackbone")
    if info.fractal_calls:
        parts.append("Fractal")
    if info.fuse_calls or info.merges:
        parts.append("Fuse")
    if info.max_fan_in >= 3:
        parts.append("Wide")
    if info.depth >= 5:
        parts.append("Deep")
    if not parts:
        parts.append("OpenMotif")
    return normalize_pattern_name("_".join(parts[:4]) + f"_{graph_hash[:6]}")


def extract_graph_info(
    init_code: str,
    forward_code: str,
    *,
    legacy_patterns: Optional[Iterable[str]] = None,
) -> GraphInfo:
    legacy_set = {normalize_pattern_name(name) for name in (legacy_patterns or DEFAULT_LEGACY_PATTERN_NAMES)}
    raw_pattern = None
    match = re.search(r"self\.pattern\s*=\s*['\"]([^'\"]+)['\"]", init_code)
    if match:
        raw_pattern = match.group(1)
    normalized_pattern = normalize_pattern_name(raw_pattern)
    is_legacy = normalized_pattern in legacy_set

    attr_types = _extract_attr_types(init_code)

    parse_ok = True
    graph_expr = "IncompleteGraph"
    info = _leaf_info("IncompleteGraph")

    try:
        tree = ast.parse(forward_code)
        fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if fn is None:
            raise ValueError("forward function missing")

        env: Dict[str, ExprInfo] = {}
        for node in fn.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                env[node.targets[0].id] = _expr_info(node.value, env, attr_types)
            elif isinstance(node, ast.Return):
                info = _expr_info(node.value, env, attr_types)
                graph_expr = re.sub(r"\s+", " ", info.text).strip()
                break
        else:
            raise ValueError("forward return missing")
    except Exception:
        parse_ok = False

    graph_hash = hashlib.sha1(graph_expr.encode("utf-8")).hexdigest()
    family_expr = _build_family_expr(graph_expr)
    plain_parallel = _looks_like_plain_parallel_triple(
        forward_code,
        graph_expr=graph_expr,
        info=info,
    )
    family_id = _build_family_id(
        info,
        parse_ok=parse_ok,
        is_plain_parallel_triple=plain_parallel,
    )
    family_hash = hashlib.sha1(f"{family_id}|{family_expr}".encode("utf-8")).hexdigest()
    cnn_expr = _build_cnn_expr(graph_expr)
    cnn_signature = hashlib.sha1(cnn_expr.encode("utf-8")).hexdigest()
    suggested_name = suggest_pattern_name(graph_hash, info)
    has_custom_name = bool(raw_pattern) and not is_legacy and normalized_pattern != "OpenMotif"

    return GraphInfo(
        pattern_name=raw_pattern or "",
        normalized_pattern_name=normalized_pattern if raw_pattern else suggested_name,
        suggested_pattern_name=suggested_name,
        graph_expr=graph_expr,
        graph_hash=graph_hash,
        family_id=family_id,
        family_expr=family_expr,
        family_hash=family_hash,
        cnn_expr=cnn_expr,
        cnn_signature=cnn_signature,
        descriptor_key=_build_descriptor_key(info),
        depth=info.depth,
        merges=info.merges,
        max_fan_in=info.max_fan_in,
        backbone_calls=info.backbone_calls,
        fractal_calls=info.fractal_calls,
        stem_calls=info.stem_calls,
        project_calls=info.project_calls,
        fuse_calls=info.fuse_calls,
        module_labels=tuple(sorted(info.module_labels)),
        is_plain_parallel_triple=plain_parallel,
        has_custom_pattern_name=has_custom_name,
        is_legacy_pattern_name=is_legacy,
        parse_ok=parse_ok,
    )

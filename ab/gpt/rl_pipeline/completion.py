"""Shared completion parsing helpers for TuneRL pipelines."""

from __future__ import annotations

import hashlib
import re
import textwrap
from typing import Dict, List, Sequence, Tuple

REQUIRED_BACKBONE_NAMES = ("backbone_a", "backbone_b")
BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"

_BLOCKED_ATTRS = {
    "device",
    "use_amp",
    "_input_spec",
    "pattern",
    "classifier",
    "infer_dimensions_dynamically",
    "train_setup",
    "learn",
    "criterion",
    "optimizer",
    "_scaler",
}

_EXTRACTION_META_CACHE: Dict[str, Dict[str, object]] = {}


def clear_extraction_meta_cache() -> None:
    _EXTRACTION_META_CACHE.clear()


def _strip_outer_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_block(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```python\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return _clean_block(match.group(1)) if match else ""


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _completion_cache_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _infer_attr_role(attr_name: str) -> str:
    lowered = attr_name.lower()
    if "fractal" in lowered:
        return "fractal"
    if lowered.startswith("backbone"):
        return "backbone"
    if "stem" in lowered:
        return "stem"
    if any(token in lowered for token in ("project", "bridge", "adapter", "align")):
        return "project"
    if any(token in lowered for token in ("fuse", "merge", "gate", "mixer")):
        return "fuse"
    return "generic"


def _has_structural_attr(attrs: Sequence[str]) -> bool:
    return any(
        _infer_attr_role(attr) in {"stem", "project", "fuse", "backbone", "fractal"}
        for attr in attrs
    )


def _scan_raw_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in _BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    return _dedupe_keep_order(attrs)


def _prepare_completion_for_xml(completion: str) -> str:
    stripped = _strip_outer_code_fences(completion or "").lstrip()
    if "<block>" not in stripped and "</block>" in stripped and "<init>" in stripped:
        return stripped
    if "<block>" not in stripped and "</block>" not in stripped and "<init>" in stripped:
        init_pos = stripped.find("<init>")
        pre_init = stripped[:init_pos].strip()
        rest = stripped[init_pos:]
        if pre_init:
            return f"<block>\n{BLOCK_SIGNATURE}\n{pre_init}\n</block>\n{rest}"
        return (
            f"<block>\n{BLOCK_SIGNATURE}\n"
            "    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias))\n"
            f"</block>\n{rest}"
        )
    return stripped


def _normalize_required_function(code: str, fn_name: str, signature: str) -> str:
    code = _strip_outer_code_fences(code)
    if not code:
        return ""
    code = textwrap.dedent(code).strip()
    if not code:
        return ""

    lines = code.splitlines()
    if lines and re.match(rf"^\s*def {re.escape(fn_name)}\s*\(", lines[0]):
        body_lines = lines[1:]
    else:
        body_lines = lines

    body_text = textwrap.dedent("\n".join(body_lines)).strip("\n")
    if not body_text.strip():
        return ""

    normalized_body = [f"    {line}" if line.strip() else "" for line in body_text.splitlines()]
    return f"{signature}\n" + "\n".join(normalized_body)


def _normalize_block_code(block_code: str) -> str:
    return _normalize_required_function(block_code, "drop_conv3x3_block", BLOCK_SIGNATURE)


def _find_last_body_line_index(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    last_index = 0
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            last_index = index + 1
    return last_index


def _repair_init_abi(init_code: str) -> str:
    if not init_code:
        return ""
    lines = init_code.splitlines()
    if not lines:
        return init_code

    signature = lines[0]
    body_lines = list(lines[1:])
    if not body_lines:
        return init_code

    repaired_body: List[str] = []
    for raw_line in body_lines:
        if raw_line.strip().startswith("self.infer_dimensions_dynamically("):
            continue
        repaired_body.append(raw_line)

    if not any(line.strip().startswith("self.device =") for line in repaired_body):
        insert_at = _find_last_body_line_index(repaired_body, ("super().__init__()",))
        repaired_body.insert(insert_at, "    self.device = device")

    if not any(line.strip().startswith("self.use_amp =") for line in repaired_body):
        insert_at = _find_last_body_line_index(repaired_body, ("super().__init__()", "self.device ="))
        repaired_body.insert(insert_at, "    self.use_amp = torch.cuda.is_available()")

    if not any("self._input_spec" in line and "=" in line for line in repaired_body):
        insert_at = _find_last_body_line_index(
            repaired_body,
            ("super().__init__()", "self.device =", "self.use_amp =", "self.pattern ="),
        )
        repaired_body.insert(insert_at, "    self._input_spec = tuple(in_shape[1:])")

    while repaired_body and not repaired_body[-1].strip():
        repaired_body.pop()
    repaired_body.append("    self.infer_dimensions_dynamically(out_shape[0])")
    return "\n".join([signature, *repaired_body])


def _normalize_init_code(init_code: str) -> str:
    normalized = _normalize_required_function(init_code, "__init__", INIT_SIGNATURE)
    return _repair_init_abi(normalized)


def _normalize_forward_code(forward_code: str) -> str:
    return _normalize_required_function(forward_code, "forward", FORWARD_SIGNATURE)


def _extract_defined_backbones(init_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\s*=", init_code or ""))


def _extract_used_backbones(forward_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\b", forward_code or ""))


def _extract_backbone_model_names(init_code: str) -> List[str]:
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in REQUIRED_BACKBONE_NAMES if name in matches]


def _count_xml_tags(text: str, tag: str) -> Tuple[int, int]:
    return (
        len(re.findall(rf"<{tag}>", text, re.IGNORECASE)),
        len(re.findall(rf"</{tag}>", text, re.IGNORECASE)),
    )


def _build_extraction_meta(
    completion: str,
    candidate: str,
    block_code: str,
    init_code: str,
    forward_code: str,
) -> Dict[str, object]:
    xml_tag_count = sum(bool(code) for code in (block_code, init_code, forward_code))
    xml_counts = {tag: _count_xml_tags(candidate, tag) for tag in ("block", "init", "forward")}
    class_count = len(re.findall(r"^\s*class\s+\w+", candidate, re.MULTILINE))
    import_count = len(re.findall(r"^\s*(?:from|import)\s+\w+", candidate, re.MULTILINE))
    bad_signature_count = len(re.findall(r"\)\s*-\s*:", candidate))
    raw_attrs = _scan_raw_attrs(candidate, block_code, init_code, forward_code)
    structural_attr_detected = _has_structural_attr(raw_attrs)

    defined_backbones = _extract_defined_backbones(init_code)
    used_backbones = _extract_used_backbones(forward_code)
    backbone_model_names = _extract_backbone_model_names(init_code)
    required_backbone_set = set(REQUIRED_BACKBONE_NAMES)
    dual_backbone_init_ok = set(defined_backbones) == required_backbone_set and len(defined_backbones) == 2
    dual_backbone_forward_ok = required_backbone_set.issubset(set(used_backbones)) and len(set(used_backbones)) == 2
    dual_backbone_ok = dual_backbone_init_ok and dual_backbone_forward_ok

    exact_xml = all(start_count == 1 and end_count == 1 for start_count, end_count in xml_counts.values())
    exact_signatures = {
        "block": block_code.startswith(BLOCK_SIGNATURE),
        "init": init_code.startswith(INIT_SIGNATURE),
        "forward": forward_code.startswith(FORWARD_SIGNATURE),
    }

    quality_score = 0
    quality_score += 2 if exact_xml else 0
    quality_score += sum(1 for ok in exact_signatures.values() if ok)
    quality_score += 2 if dual_backbone_ok else 0
    quality_score += 1 if structural_attr_detected else 0
    quality_score -= min(class_count, 2)
    quality_score -= min(import_count, 2)
    quality_score -= min(bad_signature_count, 2)

    return {
        "xml_tag_count": xml_tag_count,
        "xml_tag_exact": exact_xml,
        "xml_counts": xml_counts,
        "class_count": class_count,
        "import_count": import_count,
        "bad_signature_count": bad_signature_count,
        "structural_attr_detected": structural_attr_detected,
        "quality_score": quality_score,
        "exact_block_signature": exact_signatures["block"],
        "exact_init_signature": exact_signatures["init"],
        "exact_forward_signature": exact_signatures["forward"],
        "defined_backbones": defined_backbones,
        "used_backbones": used_backbones,
        "backbone_model_names": backbone_model_names,
        "dual_backbone_init_ok": dual_backbone_init_ok,
        "dual_backbone_forward_ok": dual_backbone_forward_ok,
        "dual_backbone_ok": dual_backbone_ok,
        "candidate_line_count": len(candidate.splitlines()),
    }


def extract_completion_payload_tolerant(completion: str) -> Tuple[Tuple[str, str, str], Dict[str, object]]:
    cache_key = _completion_cache_key(completion or "")
    cached = _EXTRACTION_META_CACHE.get(cache_key)
    if cached:
        return (
            (cached["block_code"], cached["init_code"], cached["forward_code"]),
            dict(cached["meta"]),
        )

    candidate = _prepare_completion_for_xml(completion or "")
    block_code = _normalize_block_code(_extract_xml_tag(candidate, "block"))
    init_code = _normalize_init_code(_extract_xml_tag(candidate, "init"))
    forward_code = _normalize_forward_code(_extract_xml_tag(candidate, "forward"))
    meta = _build_extraction_meta(completion or "", candidate, block_code, init_code, forward_code)

    _EXTRACTION_META_CACHE[cache_key] = {
        "block_code": block_code,
        "init_code": init_code,
        "forward_code": forward_code,
        "meta": meta,
    }
    return ((block_code, init_code, forward_code), meta)


def extract_completion_blocks_tolerant(completion: str) -> Tuple[str, str, str]:
    blocks, _ = extract_completion_payload_tolerant(completion)
    return blocks


def extract_completion_meta(completion: str) -> Dict[str, object]:
    _, meta = extract_completion_payload_tolerant(completion)
    return meta

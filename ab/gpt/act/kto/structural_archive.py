"""Structural identity, graded rarity and descriptors for generated architectures.

One trace per candidate yields:
  * the Weisfeiler-Lehman hash, identical to graph_novelty_audit.graph_hash_for_file
    (same tracer, labels and refinement), used for exact structural duplicates;
  * the WL-subtree feature multiset (labels of every refinement round), whose
    cosine similarity is the WL subtree kernel, used for a graded rarity score;
  * cheap architecture descriptors (size, depth, operator mix) that are not
    optimized by either signal, for independent diversity measures.

StructuralArchive keeps the seen hashes and an inverted index over features, so
rarity = 1 - max cosine similarity to anything already archived.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ab.gpt.act.kto.graph_novelty_audit import (
    _DefaultPrm, _graph_from_gm, _h, _load_net, _wl_hash,
)

_OP_GROUPS = (
    ("conv", ("convolution",)),
    ("linear", ("addmm", "aten.mm", "linear", "bmm", "matmul")),
    ("norm", ("batch_norm", "layer_norm", "group_norm", "instance_norm")),
    ("pool", ("pool",)),
    ("act", ("relu", "gelu", "silu", "sigmoid", "tanh", "hardswish", "elu", "mish", "softmax")),
    ("add", ("aten.add.",)),
    ("cat", ("aten.cat.",)),
)


def wl_features(labels: List[str], adj_in: List[List[int]], adj_out: List[List[int]],
                rounds: int) -> Counter:
    """WL-subtree feature multiset: refined labels of rounds 0..K (mirrors _wl_hash)."""
    cur = [_h(l) for l in labels]
    feats = Counter(cur)
    for _ in range(rounds):
        nxt = []
        for i in range(len(cur)):
            ins = ",".join(sorted(cur[j] for j in adj_in[i]))
            outs = ",".join(sorted(cur[j] for j in adj_out[i]))
            nxt.append(_h(f"{cur[i]}<{ins}>{outs}"))
        cur = nxt
        feats.update(cur)
    return feats


def descriptors(labels: List[str], adj_in: List[List[int]], adj_out: List[List[int]],
                n_params: int) -> Dict[str, int]:
    depth = [1] * len(labels)
    for i in range(len(labels)):          # fx nodes are topologically ordered
        if adj_in[i]:
            depth[i] = 1 + max(depth[j] for j in adj_in[i])
    d = {"nodes": len(labels), "edges": sum(len(a) for a in adj_out),
         "depth": max(depth) if depth else 0, "params": int(n_params)}
    counts = Counter()
    for lab in labels:
        for name, keys in _OP_GROUPS:
            if any(k in lab for k in keys):
                counts[name] += 1
                break
        else:
            counts["other"] += 1
    for name, _ in _OP_GROUPS:
        d[name] = counts[name]
    d["other"] = counts["other"]
    return d


def graph_signature_for_file(path: Path, sizes: List[int], in_ch: int, n_cls: int,
                             granularity: str, wl_rounds: int) -> Tuple[Optional[dict], str]:
    """(signature, status); signature = {"h", "f", "d"} or None when untraceable."""
    import torch

    name = None
    try:
        Net, name = _load_net(path)
        if Net is None:
            return None, "no_Net_class"
    except Exception as e:  # noqa: BLE001
        if name:
            sys.modules.pop(name, None)
        return None, f"exec_error:{type(e).__name__}"
    last = "unknown"
    try:
        for s in sizes:
            try:
                model = Net((1, in_ch, s, s), (n_cls,), _DefaultPrm(), torch.device("cpu"))
                model.eval()
                x = torch.randn(1, in_ch, s, s)
                with torch.no_grad():
                    from torch.fx.experimental.proxy_tensor import make_fx
                    gm = make_fx(model, tracing_mode="real")(x)
                labels, ai, ao = _graph_from_gm(gm, "aten", granularity)
                n_params = sum(p.numel() for p in model.parameters())
                return {"h": _wl_hash(labels, ai, ao, wl_rounds),
                        "f": dict(wl_features(labels, ai, ao, wl_rounds)),
                        "d": descriptors(labels, ai, ao, n_params)}, "ok"
            except Exception as e:  # noqa: BLE001
                last = type(e).__name__
        return None, f"trace_error:{last}"
    finally:
        sys.modules.pop(name, None)


class StructuralArchive:
    """Seen WL hashes plus a cosine index over WL-subtree features."""

    def __init__(self) -> None:
        self.seen: set = set()
        self.postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.n_items = 0

    @staticmethod
    def _normalized(feats: Dict[str, int]) -> Dict[str, float]:
        norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
        return {k: v / norm for k, v in feats.items()}

    def add(self, h: Optional[str], feats: Optional[Dict[str, int]]) -> None:
        if h:
            self.seen.add(h)
        if feats:
            idx = self.n_items
            for k, w in self._normalized(feats).items():
                self.postings[k].append((idx, w))
            self.n_items += 1

    def is_dup(self, h: Optional[str]) -> bool:
        return bool(h) and h in self.seen

    def rarity(self, feats: Optional[Dict[str, int]]) -> Optional[float]:
        """1 - max cosine similarity to the archive (1.0 when the archive is empty)."""
        if not feats:
            return None
        if self.n_items == 0:
            return 1.0
        dots: Dict[int, float] = defaultdict(float)
        for k, w in self._normalized(feats).items():
            for idx, v in self.postings.get(k, ()):
                dots[idx] += w * v
        best = max(dots.values()) if dots else 0.0
        return max(0.0, 1.0 - min(1.0, best))

    def load_corpus(self, path: Path) -> Tuple[int, int]:
        """Seed from a build_struct_corpus file. Returns (#entries, #traced)."""
        n = traced = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n += 1
                if rec.get("h"):
                    self.add(rec["h"], rec.get("f"))
                    traced += 1
        return n, traced

    def extend(self, items: Iterable[Tuple[Optional[str], Optional[Dict[str, int]]]]) -> None:
        for h, f in items:
            self.add(h, f)

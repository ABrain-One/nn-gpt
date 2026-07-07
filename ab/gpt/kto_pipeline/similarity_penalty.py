"""Near-duplicate similarity penalty for KTO (MinHash-Jaccard).

Self-contained (KTO-only): builds a MinHash-LSH index over a reference corpus
(LEMUR DB models + our own accepted generations) and scores each generated model
by its nearest-neighbour Jaccard similarity. A shaped penalty is then subtracted
from the KTO chosen reward (see ab/gpt/util/KTO.py).

Reference corpus:
  * LEMUR DB code via ab.nn.api.data(...)["nn_code"]  (static)
  * our own accepted generations, added cumulatively across cycles ("previous gens")
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

from datasketch import MinHash, MinHashLSH


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def _tokenize(code: str) -> List[str]:
    return _TOKEN_RE.findall(code or "")


def _shingles(tokens: List[str], n: int) -> set:
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def to_minhash(code: str, num_perm: int = 128, n: int = 5) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(_tokenize(code), n):
        mh.update(sh.encode("utf-8"))
    return mh


def shape_penalty(s: float, threshold: float = 0.85, shape: str = "exponential",
                  gamma: float = 4.0) -> float:
    """Map a Jaccard similarity s -> penalty in [0, 1]. Zero below threshold.

    linear:      ramps linearly from 0 (at threshold) to 1 (at s=1).
    exponential: mild for moderate similarity, harsh near exact copies.
    """
    if s <= threshold or threshold >= 1.0:
        return 0.0
    t = max(0.0, min(1.0, (s - threshold) / (1.0 - threshold)))
    if shape == "linear":
        return t
    return (math.exp(gamma * t) - 1.0) / (math.exp(gamma) - 1.0)


class SimilarityIndex:
    """MinHash-LSH over a code corpus; nearest-neighbour Jaccard queries."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128, shingle_n: int = 5):
        # LSH threshold slightly below the penalty threshold so near-threshold
        # neighbours are still retrieved as candidates (exact Jaccard is recomputed).
        self.penalty_threshold = threshold
        self.num_perm = num_perm
        self.shingle_n = shingle_n
        lsh_thr = max(0.3, threshold - 0.1)
        self.lsh = MinHashLSH(threshold=lsh_thr, num_perm=num_perm)
        self.id2mh = {}
        self._next = 0
        self.size = 0

    def add(self, code: str) -> None:
        if not code or not code.strip():
            return
        mh = to_minhash(code, self.num_perm, self.shingle_n)
        key = f"m{self._next}"
        self._next += 1
        try:
            self.lsh.insert(key, mh)
        except Exception:  # noqa: BLE001 — duplicate key / index hiccup is non-fatal
            return
        self.id2mh[key] = mh
        self.size += 1

    def add_codes(self, codes: List[str]) -> int:
        before = self.size
        for c in codes:
            self.add(c)
        return self.size - before

    def add_db(self, task: str = "img-classification", dataset: str = "cifar-10",
               unique_nn: bool = True) -> int:
        """Index the LEMUR DB model code (via ab.nn.api.data). Returns #added."""
        from ab.nn.api import data
        df = data(task=task, dataset=dataset, unique_nn=unique_nn)
        if "nn_code" not in getattr(df, "columns", []):
            return 0
        codes = [c for c in df["nn_code"].tolist() if isinstance(c, str) and c.strip()]
        return self.add_codes(codes)

    def nearest_jaccard(self, code: str) -> float:
        if not code or self.size == 0:
            return 0.0
        mh = to_minhash(code, self.num_perm, self.shingle_n)
        best = 0.0
        for k in self.lsh.query(mh):
            j = mh.jaccard(self.id2mh[k])
            if j > best:
                best = j
        return best

    def penalty_for(self, code: str, shape: str = "exponential") -> float:
        """Shaped penalty in [0, 1] for a code vs the indexed corpus (excludes alpha)."""
        return shape_penalty(self.nearest_jaccard(code), self.penalty_threshold, shape)

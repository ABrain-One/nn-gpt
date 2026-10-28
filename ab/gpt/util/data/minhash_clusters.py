"""Global similarity clustering of every model in the dataset, with no queries.

``ab/nn/stat/nn/<name>.json`` carries a ``code_minhash`` block for each model:
128 permutations over 7-shingles of its source. That is a complete similarity
structure sitting on disk, so the clustering this pipeline needs does not have
to be discovered by probing the database one band at a time -- and it does not
have to fall back on the shape of a model's name, which is not a property of
the model at all. One generator gives every batch its own prefix, another puts
1,900 near-identical models under a single one; neither tells you what the code
looks like.

What this replaces, concretely: the anchor loop used to pick candidates by name
prefix and then spend a database join per candidate to find out whether it had
any neighbours. With the signatures read up front, the clusters are known
before the first query, so anchors can be one per cluster, largest first.

Jaccard is estimated the usual way -- the fraction of the 128 positions where
two signatures agree -- and candidate pairs are found by LSH banding so nothing
is compared against everything. Clusters are the connected components of the
verified pairs.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Rows per LSH band. With 128 permutations this gives 16 bands, and two
#: models land in the same bucket with probability s**8 summed over bands, so
#: the band structure starts admitting candidates around s = 0.84 -- just below
#: the verification threshold, which is where it should be.
ROWS_PER_BAND = int(os.environ.get("NNGPT_MINHASH_ROWS", "8") or 8)

#: Estimated Jaccard two models must reach to be called the same cluster.
THRESHOLD = float(os.environ.get("NNGPT_MINHASH_THRESHOLD", "0.8") or 0.8)

#: Set to 0 to ignore the signatures and use the previous behaviour.
ENABLED_ENV = "NNGPT_MINHASH_CLUSTERS"

_CACHE: Dict[Tuple, Dict[str, str]] = {}


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def stat_nn_dir() -> Optional[Path]:
    """Where the installed dataset keeps its per-model statistics."""
    override = os.environ.get("NNGPT_NN_STAT_DIR", "").strip()
    if override:
        d = Path(override)
        return d if d.is_dir() else None
    try:
        import importlib.util
        spec = importlib.util.find_spec("ab.nn")
    except Exception:
        return None
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "stat" / "nn"
    return d if d.is_dir() else None


def load_signatures(names: Iterable[str],
                    stat_dir: Optional[Path] = None) -> Dict[str, Tuple[int, ...]]:
    """MinHash signature per model, for the models that have one on disk."""
    stat_dir = stat_dir or stat_nn_dir()
    if stat_dir is None:
        return {}
    out: Dict[str, Tuple[int, ...]] = {}
    for name in names:
        f = stat_dir / f"{name}.json"
        if not f.is_file():
            continue
        try:
            block = (json.loads(f.read_text(encoding="utf-8")) or {}).get("code_minhash")
        except Exception:
            continue
        if not isinstance(block, dict) or not block.get("available"):
            continue
        values = block.get("hashvalues")
        if not isinstance(values, list) or not values:
            continue
        out[name] = tuple(int(v) for v in values)
    return out


def _agreement(a: Sequence[int], b: Sequence[int]) -> float:
    n = min(len(a), len(b))
    if not n:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def cluster(names: Sequence[str], threshold: float = None,
            rows: int = None, stat_dir: Optional[Path] = None,
            verbose: bool = True) -> Dict[str, str]:
    """``{name: cluster id}`` for the models whose signature is on disk.

    Names with no signature are absent from the result; the caller decides
    what to do with them, which is the honest division of labour -- this
    module knows about similarity and nothing about fallbacks.
    """
    names = sorted(set(str(n) for n in names))
    if not names or not enabled():
        return {}
    threshold = THRESHOLD if threshold is None else threshold
    rows = ROWS_PER_BAND if rows is None else rows

    key = (len(names), hash(tuple(names)), threshold, rows)
    if key in _CACHE:
        return _CACHE[key]

    sigs = load_signatures(names, stat_dir)
    if not sigs:
        if verbose:
            print("[minhash] no signatures found under ab/nn/stat/nn; "
                  "falling back to the band-probing clustering", flush=True)
        _CACHE[key] = {}
        return {}

    # LSH banding: identical in any band of `rows` consecutive hashes makes a
    # candidate. Verification below decides.
    buckets: Dict[Tuple, List[str]] = collections.defaultdict(list)
    for name, sig in sigs.items():
        for band in range(len(sig) // rows):
            buckets[(band, sig[band * rows:(band + 1) * rows])].append(name)

    parent: Dict[str, str] = {n: n for n in sigs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    compared = merged = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Against the bucket's representative, not against every other member:
        # a bucket of 1,900 identical signatures would otherwise be 1.8M
        # comparisons, and the components come out the same because similarity
        # to the representative is what the bucket already asserts.
        rep = members[0]
        for other in members[1:]:
            if find(rep) == find(other):
                continue
            compared += 1
            if _agreement(sigs[rep], sigs[other]) >= threshold:
                union(rep, other)
                merged += 1

    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for n in sigs:
        groups[find(n)].append(n)
    label = {root: f"mh:{sorted(members)[0]}" for root, members in groups.items()}
    out = {n: label[find(n)] for n in sigs}

    if verbose:
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        singles = sum(1 for s in sizes if s == 1)
        print(f"[minhash] {len(groups)} cluster(s) over {len(sigs)} of "
              f"{len(names)} model(s) at Jaccard >= {threshold:.2f} "
              f"({compared} pair(s) verified, {merged} merged); "
              f"largest {sizes[:6]}, {singles} singleton(s)", flush=True)
    _CACHE[key] = out
    return out
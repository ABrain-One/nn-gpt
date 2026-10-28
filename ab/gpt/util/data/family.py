"""Architecture families, and how to spread a sample across them.

Every selector in this pipeline -- band anchors, the holdout, the pinned test
set, the SFT subsample -- was uniform over *model names*. The dataset is not
uniform over families: a search that produced hundreds of near-identical
variants (``rl-bb-init-<hash>``) contributes hundreds of names, while a
hand-written architecture contributes one. Uniform sampling therefore returns
mostly the former, and the measured consequence was a 30-model test set of
which 29 came from one family and 60 "novel" generations that were all the same
template with a different backbone string.

A family here is the model name with its content hash removed. That is the
grouping LEMUR's own naming already encodes (``<family>-<32 hex>``), it needs
no database query, and it is stable across cycles -- which matters because the
cap has to mean the same thing in cycle 0 and cycle 15.
"""

from __future__ import annotations

import collections
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

#: LEMUR appends a content hash to every generated name. Also matches the
#: shorter hashes some generators use, down to 8 hex digits.
_HASH_TAIL = re.compile(r"-[0-9a-f]{8,}$", re.I)

#: A trailing numeric variant index, as in ``AlexNet-3``.
_INDEX_TAIL = re.compile(r"-\d+$")


def family_of(nn_name) -> str:
    """The family a model name belongs to.

    ``rl-bb-init-1d4ad77a...`` -> ``rl-bb-init``; ``llm-088e98...`` -> ``llm``;
    ``AlexNet`` -> ``AlexNet``. An unparseable name is its own family, which is
    the safe default: it can only ever use one slot of a per-family cap.
    """
    name = str(nn_name or "").strip()
    if not name:
        return "?"
    stripped = _HASH_TAIL.sub("", name)
    if stripped != name:
        return stripped or name
    stripped = _INDEX_TAIL.sub("", name)
    return stripped or name


def group_by_family(items: Iterable, key: Optional[Callable] = None,
                    raw_key: bool = False) -> Dict[str, List]:
    """Items bucketed by group, each bucket in the order it was seen.

    ``raw_key`` says that ``key`` already returns the group itself -- a MinHash
    cluster id, a code-shape tuple -- rather than a name to derive a family
    from. Without it a non-name key would be run through the name parser, which
    is meaningless for anything that is not a model name.
    """
    out: Dict[str, List] = collections.defaultdict(list)
    for it in items:
        if key is None:
            group = family_of(it)
        elif raw_key:
            group = key(it)
        else:
            group = family_of(key(it))
        out[group].append(it)
    return dict(out)


def stratified_sample(items: Sequence, n: int, rng, key: Optional[Callable] = None,
                      min_members: int = 1, biggest_first: bool = False,
                      raw_key: bool = False) -> List:
    """``n`` items spread as evenly as possible over the families present.

    Round-robin over families, shuffling within each so the pick is not the
    database's own order, and taking one per family per pass. A family with k
    members can contribute at most ceil(n / n_families) before every other
    family has had its turn, so the big families stop crowding the sample out
    without anything being excluded outright: when the small families run dry
    the remaining slots do go back to the big ones.
    """
    if n <= 0 or not items:
        return []
    buckets = group_by_family(items, key, raw_key=raw_key)
    if min_members > 1:
        # A family too small to be useful to the caller is pure cost: it is
        # visited, queried and discarded. Dropping it here keeps the sample
        # just as diverse -- it still spans families -- and much cheaper.
        buckets = {f: m for f, m in buckets.items() if len(m) >= min_members}
        if not buckets:
            buckets = group_by_family(items, key, raw_key=raw_key)
    for members in buckets.values():
        rng.shuffle(members)
    if biggest_first:
        # Visit the best-connected families first. The round-robin still takes
        # one member per family per pass, so the head of the sample is still
        # n distinct families -- but they are the n families most likely to
        # satisfy whatever the caller needs them for.
        order = sorted(buckets, key=lambda f: (-len(buckets[f]), repr(f)))
    else:
        order = sorted(buckets, key=repr)
        rng.shuffle(order)
    out: List = []
    cursor = 0
    while len(out) < n:
        took = 0
        for fam in order:
            members = buckets[fam]
            if cursor < len(members):
                out.append(members[cursor])
                took += 1
                if len(out) >= n:
                    break
        if not took:
            break
        cursor += 1
    return out


def cap_per_family(items: Sequence, max_per_family: int,
                   key: Optional[Callable] = None,
                   rank: Optional[Callable] = None) -> List:
    """At most ``max_per_family`` items from any one family, order preserved.

    ``rank`` decides which members of an over-represented family survive --
    pass a key that sorts the ones worth keeping first (accuracy, say).
    """
    if not max_per_family or max_per_family < 1:
        return list(items)
    keep = set()
    for fam, members in group_by_family(items, key).items():
        chosen = sorted(members, key=rank) if rank else members
        for it in chosen[:max_per_family]:
            keep.add(id(it))
    return [it for it in items if id(it) in keep]


def family_histogram(names: Iterable) -> Dict[str, int]:
    """How many of ``names`` each family holds, biggest first."""
    counts = collections.Counter(family_of(n) for n in names)
    return dict(counts.most_common())


def describe(names: Iterable, limit: int = 6) -> str:
    """One line for a log: how concentrated a selection is."""
    hist = family_histogram(names)
    total = sum(hist.values()) or 1
    head = ", ".join(f"{k}:{v}" for k, v in list(hist.items())[:limit])
    top = next(iter(hist.values()), 0)
    return (f"{len(hist)} famil{'y' if len(hist) == 1 else 'ies'} over {total} "
            f"model(s), largest {100.0 * top / total:.0f}% [{head}"
            + (", ..." if len(hist) > limit else "") + "]")
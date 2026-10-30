"""Which corpus rows one cycle trains on.

The sampler was uniform over rows. Two things follow from that, both measured.

Family. The corpus inherits the dataset's family mix, so a uniform sample is
~90% one search family's near-identical variants. The fine-tune then reproduces
that family's template: 60 of 60 "novel" generations in one run shared the same
class skeleton and differed by a backbone string. Balancing over families costs
nothing -- the corpus is not reduced, only the mix of what is drawn from it --
and is the one change that does not trade diversity against corpus size.

Gain. Every improving pair is equally likely to be drawn, so a pair that gained
0.005 teaches as loudly as one that gained 0.15. The objective is the best
model a cycle produces, and an imitation objective over all improvements is not
that objective: training loss sat at 0.050 and flat while cycle_max never beat
what the base model managed before any fine-tuning. Weighting the draw by the
measured gain is the cheapest step toward "learn the big improvements" that
does not require a preference objective; it changes the distribution the SFT
loss averages over, not the loss itself.

Set ``NNGPT_SFT_SAMPLER=uniform`` to get the previous behaviour back, which is
what an ablation of this needs.
"""

from __future__ import annotations

import collections
import math
import os
import random
from typing import Dict, List, Optional, Sequence

from ab.gpt.util.data.family import family_of

#: How much a measured gain raises a row's weight. w = 1 + SCALE * gain, so at
#: 12 a 0.15 gain is drawn ~2.8x as often as a 0.005 one. Deliberately mild: a
#: harder weighting narrows the corpus to a handful of rows and reintroduces the
#: memorisation the per-target cap in edit_pairs exists to prevent.
GAIN_SCALE = float(os.environ.get("NNGPT_SFT_GAIN_SCALE", "12") or 12)

#: No family may take more than this share of a cycle's rows while any other
#: family still has rows to offer.
MAX_FAMILY_SHARE = float(os.environ.get("NNGPT_SFT_MAX_FAMILY_SHARE", "0.25") or 0.25)


def _row_family(row) -> str:
    """Which group a pair belongs to, best evidence first.

    The MinHash cluster the pair was drawn from is the real answer: it is a
    similarity measure over the architecture graph. The name prefix is a
    fallback for corpora built before clusters were recorded, and a poor one --
    one generator gives every batch its own prefix, another puts 1,900
    near-identical models under a single one.
    """
    meta = row.get("meta") if isinstance(row, dict) else None
    if isinstance(meta, dict):
        cluster = meta.get("baseline_cluster")
        if cluster:
            return str(cluster)
        fam = meta.get("baseline_family")
        if fam:
            return str(fam)
        if meta.get("baseline_nn"):
            return family_of(meta["baseline_nn"])
    return "?"


def _row_gain(row) -> float:
    meta = row.get("meta") if isinstance(row, dict) else None
    if isinstance(meta, dict):
        try:
            return max(0.0, float(meta.get("observed_gain") or 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _weighted_order(rows: Sequence, rng: random.Random,
                    gain_scale: float) -> List:
    """Rows in weighted-random order, without replacement.

    Efraimidis-Spirakis: a key of u**(1/w) ranks a weighted sample without
    replacement in one pass, so the whole family can be ordered once and then
    consumed by the round-robin below.
    """
    keyed = []
    for row in rows:
        w = 1.0 + max(0.0, gain_scale) * _row_gain(row)
        u = max(rng.random(), 1e-12)
        keyed.append((math.log(u) / w, row))
    keyed.sort(key=lambda t: -t[0])
    return [r for _k, r in keyed]


def _task_of(row) -> str:
    meta = row.get("meta") if isinstance(row, dict) else None
    if isinstance(meta, dict) and meta.get("task"):
        return str(meta["task"])
    return "improve"


def sample_rows(rows: Sequence, n: int, seed: int = 0,
                gain_scale: Optional[float] = None,
                max_family_share: Optional[float] = None,
                verbose: bool = True) -> List:
    """``n`` rows: the corpus's task mix, each task spread over its clusters."""
    rows = list(rows)
    if n is None or n >= len(rows) or n <= 0:
        return rows
    rng = random.Random(seed)
    mode = os.environ.get("NNGPT_SFT_SAMPLER", "balanced").strip().lower()
    if mode == "uniform":
        picked = rng.sample(rows, n)
        if verbose:
            print(f"[SFT] uniform sample of {n} of {len(rows)} rows", flush=True)
        return picked

    # The task mix is decided when the corpus is built, and it has to survive
    # this draw. It would not otherwise: an auxiliary row has no measured gain,
    # so the weighting below gives it the floor weight of 1.0 while an
    # improvement row with a 0.15 gain gets 2.8 -- the auxiliary tasks would be
    # built at a 40% share and then trained at a fraction of it. So each task
    # is allocated its share of n first, and the weighting operates inside a
    # task, where every row is the same kind of answer and the comparison is
    # meaningful.
    by_task: Dict[str, List] = collections.defaultdict(list)
    for row in rows:
        by_task[_task_of(row)].append(row)
    if len(by_task) > 1:
        picked: List = []
        order = sorted(by_task, key=lambda t: -len(by_task[t]))
        allocated = 0
        for i, task in enumerate(order):
            quota = (n - allocated if i == len(order) - 1
                     else int(round(n * len(by_task[task]) / len(rows))))
            quota = max(0, min(quota, len(by_task[task]), n - allocated))
            if quota:
                picked.extend(_sample_one_task(
                    by_task[task], quota, rng, gain_scale, max_family_share))
                allocated += quota
        # A task with fewer rows than its quota leaves slots; the largest task
        # takes them, which is the improvement task in any sane corpus.
        if allocated < n:
            have = {id(r) for r in picked}
            rest = [r for r in rows if id(r) not in have]
            rng.shuffle(rest)
            picked.extend(rest[:n - allocated])
        rng.shuffle(picked)
        if verbose:
            got = collections.Counter(_task_of(r) for r in picked)
            print(f"[SFT] balanced sample of {len(picked)} of {len(rows)} rows "
                  f"over {len(by_task)} task(s): "
                  + ", ".join(f"{k}={v}" for k, v in got.most_common()),
                  flush=True)
        return picked
    return _sample_one_task(rows, n, rng, gain_scale, max_family_share,
                            verbose=verbose)


def _sample_one_task(rows: Sequence, n: int, rng: random.Random,
                     gain_scale: Optional[float],
                     max_family_share: Optional[float],
                     verbose: bool = False) -> List:
    """Cluster-balanced, gain-weighted draw within a single task."""
    rows = list(rows)
    if n >= len(rows):
        return rows

    gain_scale = GAIN_SCALE if gain_scale is None else gain_scale
    share = MAX_FAMILY_SHARE if max_family_share is None else max_family_share
    cap = max(1, int(math.ceil(share * n))) if share and share > 0 else n

    buckets: Dict[str, List] = collections.defaultdict(list)
    for row in rows:
        buckets[_row_family(row)].append(row)
    ordered = {fam: _weighted_order(members, rng, gain_scale)
               for fam, members in buckets.items()}
    families = sorted(ordered)
    rng.shuffle(families)

    picked: List = []
    taken = collections.Counter()
    # Pass one honours the cap, so no family can fill the cycle on its own.
    # Pass two ignores it, because leaving slots empty to enforce a share
    # would shrink the corpus the trainer sees -- the cap is about the mix.
    for respect_cap in (True, False):
        depth = 0
        while len(picked) < n:
            progress = False
            for fam in families:
                if len(picked) >= n:
                    break
                members = ordered[fam]
                if depth >= len(members):
                    continue
                if respect_cap and taken[fam] >= cap:
                    continue
                picked.append(members[depth])
                taken[fam] += 1
                progress = True
            if not progress:
                break
            depth += 1
        if len(picked) >= n:
            break

    if verbose:
        gains = [_row_gain(r) for r in picked]
        top = taken.most_common(5)
        print(f"[SFT] balanced sample of {len(picked)} of {len(rows)} rows: "
              f"{len(taken)} famil{'y' if len(taken) == 1 else 'ies'}, "
              f"largest {100.0 * (top[0][1] if top else 0) / max(1, len(picked)):.0f}% "
              f"({', '.join(f'{k}:{v}' for k, v in top)}), "
              f"mean measured gain {sum(gains) / max(1, len(gains)):+.4f}",
              flush=True)
    return picked

"""
Segment-order permutations of already-evaluated schedules.

Source selection. 
Sources are chosen by STRATIFIED accuracy sampling: 
all evaluated schedules with >= 2 segments are split into top/medium/weak, 
and the --per_tier highest-accuracy schedules are taken from each tier. 

Sampling. A schedule with n segments has n! orderings. n=2 -> 2, n=3 -> 6, n=4 -> 24. 
The identity ordering is dropped. For n<=3 every remaining ordering is emitted; for n=4 a
random sample of --max_per_source is drawn.

Every reordering is re-validated with policy='reject' and re-hashed with ConfigGen.
Anything that collides with an already-evaluated config_id is dropped.

"""

import argparse
import glob
import itertools
import json
import os
import random
from pathlib import Path

from ab.gpt.brute.trans.augment.ScheduleValidate import validate_schedule
from ab.gpt.brute.trans.augment.config.ConfigGen import config_hash
from ab.gpt.util.Const import trans_dir, out_dir


RESULT_DIR = trans_dir / 'augment/gen_result'
CONFIG_DIR = trans_dir / 'augment/config'
RUN_MINUTES_PER_CONFIG = 52.6  # observed median, 30-epoch ImageNette/ResNet


def _method_seq(segments):
    return [m for m, a, f in segments]


def load_evaluated(result_dir: Path):
    out = {}
    for f in glob.glob(str(result_dir / "**" / "*seed42.json"), recursive=True):
        if "training_summary" in f or os.path.getsize(f) == 0:
            continue
        try:
            j = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        cid = j.get("config_id")
        if cid and "augment" in j and "accuracy" in j:
            out[cid] = {"augment": j["augment"], "accuracy": j["accuracy"]}
    return out


def known_config_ids(config_dir: Path, evaluated: dict):
    ids = set(evaluated)
    for f in glob.glob(str(config_dir / "aug_config_*.json")):
        d = json.load(open(f))
        for entry in d.values():
            cid = entry.get("config_id") if isinstance(entry, dict) else None
            if cid:
                ids.add(cid)
    return ids


def select_sources(evaluated: dict, per_tier: int, source_ids=None):
    """Stratified sampling"""
    eligible = {cid: v for cid, v in evaluated.items() if len(v["augment"]) >= 2}
    if source_ids:
        missing = [c for c in source_ids if c not in eligible]
        if missing:
            raise ValueError(f"--source_ids not found: {missing}")
        return [(c, eligible[c], "manual") for c in source_ids]

    ranked = sorted(eligible.items(), key=lambda kv: -kv[1]["accuracy"])
    n = len(ranked)
    if n == 0:
        return []
    b1, b2 = n // 3, (2 * n) // 3
    tiers = [
        ("top", ranked[:b1] or ranked[:1]),
        ("medium", ranked[b1:b2] or ranked[b1:b1 + 1]),
        ("weak", ranked[b2:] or ranked[-1:]),
    ]

    sources = []
    for tier_name, members in tiers:
        for cid, rec in members[:per_tier]:
            sources.append((cid, rec, tier_name))
    return sources


def permute_source(segments, rng, max_per_source):
    """All non-identity orderings for n<=3"""
    n = len(segments)
    identity = tuple(range(n))
    all_perms = [p for p in itertools.permutations(range(n)) if p != identity]
    if len(all_perms) > max_per_source:
        all_perms = rng.sample(all_perms, max_per_source)
    out = []
    for perm in all_perms:
        reordered = [segments[i] for i in perm]
        canonical, _ = validate_schedule(reordered, policy="reject")
        out.append((perm, canonical))
    return out


def generate(result_dir: Path, config_dir: Path, per_tier: int,
             max_per_source: int, seed: int, source_ids=None):
    evaluated = load_evaluated(result_dir)
    if not evaluated:
        raise RuntimeError(f"No evaluated schedules with full history found in {result_dir}")
    seen_ids = known_config_ids(config_dir, evaluated)
    rng = random.Random(seed)

    sources = select_sources(evaluated, per_tier, source_ids)
    skipped_no_order = sum(1 for v in evaluated.values() if len(v["augment"]) < 2)

    entries = []
    per_source_counts = {}
    for source_id, src, tier in sources:
        segments = src["augment"]
        candidates = permute_source(segments, rng, max_per_source)
        kept = 0
        for perm, canonical in candidates:
            cid = config_hash(canonical)
            if cid == source_id or cid in seen_ids:
                continue
            seen_ids.add(cid)
            entries.append({
                "config_id": cid,
                "augment_configs": canonical,
                "source_config_id": source_id,
                "source_tier": tier,
                "source_order": _method_seq(segments),
                "permutation": list(perm),
            })
            kept += 1
        per_source_counts[source_id] = (tier, len(candidates), kept)

    return entries, sources, per_source_counts, skipped_no_order


def save(entries, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aug_config_perm_seed{seed}_n{len(entries)}.json")
    output = {str(i + 1): e for i, e in enumerate(entries)}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Generate segment-order permutations of evaluated schedules")
    ap.add_argument("--result_dir", type=str, default=str(RESULT_DIR))
    ap.add_argument("--config_dir", type=str, default=str(CONFIG_DIR))
    ap.add_argument("--per_tier", type=int, default=5,
                    help="highest-accuracy evaluated schedules to permute from")
    ap.add_argument("--max_per_source", type=int, default=5,
                    help="cap on new orderings per source")
    ap.add_argument("--source_ids", type=str, default=None,
                    help="comma-separated config_ids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    source_ids = args.source_ids.split(",") if args.source_ids else None
    entries, sources, per_source_counts, skipped = generate(
        Path(args.result_dir), Path(args.config_dir),
        args.per_tier, args.max_per_source, args.seed, source_ids)

    print(f"Sources considered: {len(sources)}  "
          f"(skipped {skipped} evaluated schedules with < 2 segments)")
    for source_id, (tier, available, kept) in per_source_counts.items():
        print(f"  [{tier}] {source_id}: {available} orderings tried, {kept} new "
              f"({available - kept} duplicated an existing config)")

    if not entries:
        print("No new permutations")
        return

    out_path = save(entries, args.config_dir, args.seed)
    est_hours = len(entries) * RUN_MINUTES_PER_CONFIG / 60
    print(f"\nTotal new permutations: {len(entries)}")
    print(f"Estimated cost at ~{RUN_MINUTES_PER_CONFIG:.1f} min/run (30-epoch fidelity): "
          f"{est_hours:.1f} GPU-hours")
    

if __name__ == "__main__":
    main()

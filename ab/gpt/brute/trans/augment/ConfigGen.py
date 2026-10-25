"""
generates augmentation-schedule configs 
  - Type sequences are shuffled under a fixed seed before enumeration.
  - The all-'none' baseline config [["none", null, 1.0]] is now emitted first.
"""

import argparse
import hashlib
import itertools
import json
import os
import random

from ab.gpt.brute.trans.augment.ScheduleValidate import validate_schedule, METHODS



AUG_TYPES = [m for m in METHODS if m != "none"] + ["none"]
# grid alphas per method
# cutout/erasing are fractions of image side
ALPHA_OPTIONS = {
    "none":          [None],
    "cutmix":        [0.5, 1.0, 2.0],
    "mixup":         [0.2, 0.5, 1.0],
    "cutout":        [0.1, 0.2, 0.35],
    "erasing":       [0.1, 0.2, 0.35],
    "gridmask":      [0.4, 0.5, 0.6],
    "hide_and_seek": [0.1, 0.25, 0.4],
    "resizemix":     [0.2, 0.4, 0.6],
}

PROB_SPLITS = {
    1: [(1.0,)],
    2: [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
        (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1),
        (0.25, 0.75), (0.75, 0.25), (1 / 3, 2 / 3), (2 / 3, 1 / 3)],
    3: [(1 / 3, 1 / 3, 1 / 3),
        (0.25, 0.25, 0.50), (0.25, 0.50, 0.25), (0.50, 0.25, 0.25),
        (0.20, 0.40, 0.40), (0.40, 0.20, 0.40), (0.40, 0.40, 0.20),
        (0.125, 0.125, 0.75), (0.125, 0.75, 0.125), (0.75, 0.125, 0.125),
        (0.25, 0.375, 0.375), (0.375, 0.25, 0.375), (0.375, 0.375, 0.25),
        (0.1, 0.3, 0.6), (0.2, 0.3, 0.5), (0.1, 0.4, 0.5),
        (0.20, 0.20, 0.60), (0.20, 0.60, 0.20), (0.60, 0.20, 0.20),
        (0.30, 0.30, 0.40), (0.30, 0.40, 0.30), (0.40, 0.30, 0.30)],
    4: [(0.25, 0.25, 0.25, 0.25),
        (0.10, 0.15, 0.25, 0.50), (0.50, 0.25, 0.15, 0.10),
        (0.15, 0.35, 0.35, 0.15),
        (0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)],
}

for n_segs, splits in PROB_SPLITS.items(): 
    assert len(set(splits)) == len(splits), f"duplicate split in PROB_SPLITS[{n_segs}]"
    for split in splits:
        assert abs(sum(split) - 1.0) < 1e-6, \
            f"PROB_SPLITS[{n_segs}] entry {split} sums to {sum(split):.6f}"

BASELINE = [["none", None, 1.0]]  # the no-augmentation baseline, always config 1


def config_hash(config) -> str:
    raw = json.dumps(config, sort_keys=False, separators=(",", ":")).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def _finalize(raw_config, seen):
    """Validate, canonicalize, dedupe, Returns config or None"""
    canonical, report = validate_schedule(raw_config, policy="renormalize")
    if canonical is None:
        return None
    h = config_hash(canonical)
    if h in seen:
        return None
    seen.add(h)
    return canonical


def generate_grid_configs(n_aug_types, n_configs, seed=0):
    """ shuffled ordered type sequences x alphas x splits."""
    if n_aug_types not in range(1, 5):
        raise ValueError("n_aug_types must be 1..4")
    rng = random.Random(seed)

    type_sequences = [seq for seq in itertools.permutations(AUG_TYPES, n_aug_types)
                      if not all(t == "none" for t in seq)]
    rng.shuffle(type_sequences)  

    seen, configs = set(), []
    if n_aug_types == 1:  
        base = _finalize([list(s) for s in BASELINE], seen)
        if base:
            configs.append(base)

    for type_seq in type_sequences:
        alpha_combos = list(itertools.product(*[ALPHA_OPTIONS[t] for t in type_seq]))
        rng.shuffle(alpha_combos)
        for alphas in alpha_combos:
            splits = list(PROB_SPLITS[n_aug_types])
            rng.shuffle(splits)
            for split in splits:
                cfg = _finalize([[t, a, p] for t, a, p in zip(type_seq, alphas, split)],
                                seen)
                if cfg:
                    configs.append(cfg)
                if len(configs) >= n_configs:
                    return configs
    return configs


def generate_offgrid_configs(n_configs, seed=0, max_segments=4):
    """Uniformly random configs: random methods, continuous alphas within
    the validator's ranges, Dirichlet-style random fractions."""
    rng = random.Random(seed)
    seen, configs = set(), []
    attempts = 0
    while len(configs) < n_configs and attempts < n_configs * 50:
        attempts += 1
        n = rng.randint(1, max_segments)
        methods = [rng.choice(AUG_TYPES) for _ in range(n)]
        if all(m == "none" for m in methods):
            continue  # the baseline exists once
        raw = [rng.random() + 1e-3 for _ in range(n)]  
        total = sum(raw)
        segs = []
        for m, r in zip(methods, raw):
            if m == "none":
                a = None
            else:
                lo, hi, _ = METHODS[m]
                a = round(rng.uniform(lo, hi), 4)
            segs.append([m, a, r / total])
        cfg = _finalize(segs, seen)
        if cfg:
            configs.append(cfg)
    return configs


def save(configs, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aug_config_{tag}_n{len(configs)}.json")
    output = {str(i + 1): {"config_id": config_hash(c), "augment_configs": c}
              for i, c in enumerate(configs)}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(configs)} configs -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate augmentation-schedule configs")
    ap.add_argument("-t", "--n_aug_types", type=int, default=4,
                    help="segments per config for grid mode (1-4)")
    ap.add_argument("-n", "--n_configs", type=int, default=60)
    ap.add_argument("-o", "--output_dir", type=str, default="ab/gpt/brute/trans/augment/config")
    ap.add_argument("--offgrid", action="store_true",
                    help="emit uniformly random configs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.offgrid:
        cfgs = generate_offgrid_configs(args.n_configs, seed=args.seed)
        save(cfgs, args.output_dir, f"offgrid_seed{args.seed}")
    else:
        cfgs = generate_grid_configs(args.n_aug_types, args.n_configs, seed=args.seed)
        save(cfgs, args.output_dir, f"seg{args.n_aug_types}_seed{args.seed}")


if __name__ == "__main__":
    main()
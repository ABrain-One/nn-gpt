"""Count which normalisation layers a corpus of models actually INSTANTIATES.

WHY AST AND NOT GREP. The first attempt at this measurement was a grep for `nn.BatchNorm2d`,
and it reported "GA output is 100% BatchNorm". The true answer was the opposite -- 88
norm-free, 1 BatchNorm out of 89. The grep was matching the NaN-guard

    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

which mentions the class without ever constructing one. Nearly every denoise template
contains a line like that, so the false positive was near-universal.

Here only `ast.Call` nodes count -- i.e. `nn.BatchNorm2d(...)`, an actual construction. A
bare reference inside an isinstance tuple is a Name/Attribute node and cannot be counted by
construction, not by a filter that might be forgotten. That is the whole point of using the
parser: the class of error above becomes unrepresentable rather than merely guarded against.

A model is classified by the SET of norms it constructs; "none" means it constructs no
normalisation layer at all. Reported per corpus so the GA's generated sources can be compared
against its seeds and against the previous run.

  python norm_census.py --glob '.../seed_ga_NORMv3_models/*.py' --label normv3
  python norm_census.py --glob 'A/*.py' --label A --glob 'B/*.py' --label B
"""
import argparse
import ast
import glob as globmod
import os
from collections import Counter

NORMS = {
    "BatchNorm1d": "BatchNorm", "BatchNorm2d": "BatchNorm", "BatchNorm3d": "BatchNorm",
    "SyncBatchNorm": "BatchNorm",
    "GroupNorm": "GroupNorm",
    "LayerNorm": "LayerNorm",
    "InstanceNorm1d": "InstanceNorm", "InstanceNorm2d": "InstanceNorm",
    "InstanceNorm3d": "InstanceNorm",
    "LocalResponseNorm": "LocalResponseNorm",
    "RMSNorm": "RMSNorm",
}
# Mobile-NPU reality, not a style preference: LayerNorm/GroupNorm/InstanceNorm force a CPU
# fallback on the TFLite GPU delegate (edge_deploy.py treats them as SOFT violations).
EDGE_SAFE = {"BatchNorm", "none", "RMSNorm-custom"}


def norms_of(path):
    """Set of norm families CONSTRUCTED in this file, or None if unparseable."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return None
    found = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name in NORMS:
            found.add(NORMS[name])
    return found


def census(paths, label):
    per_model, bad = Counter(), 0
    families = Counter()
    for p in paths:
        s = norms_of(p)
        if s is None:
            bad += 1
            continue
        key = "+".join(sorted(s)) if s else "none"
        per_model[key] += 1
        for f in s:
            families[f] += 1
    n = sum(per_model.values())
    print(f"\n=== {label} | {n} parsed" + (f", {bad} unparseable" if bad else "") + " ===")
    if not n:
        return per_model
    for k, v in per_model.most_common():
        print(f"  {k:<34} {v:5d}  ({v/n:6.1%})")
    nf = per_model.get("none", 0)
    edge_ok = sum(v for k, v in per_model.items()
                  if all(part in EDGE_SAFE for part in k.split("+")))
    print(f"  {'-'*34}")
    print(f"  norm-free                          {nf:5d}  ({nf/n:6.1%})")
    print(f"  uses >=1 norm                      {n-nf:5d}  ({(n-nf)/n:6.1%})")
    print(f"  edge-safe norm profile             {edge_ok:5d}  ({edge_ok/n:6.1%})"
          f"   [BatchNorm or none]")
    return per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="glob of .py model sources; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="label per --glob, in order")
    a = ap.parse_args()

    results = {}
    for i, g in enumerate(a.glob):
        label = a.label[i] if i < len(a.label) else g
        # _temp_model_*.py are LEMUR scratch files, not corpus members; including them
        # double-counts whatever happened to be training when the glob ran.
        paths = [p for p in sorted(globmod.glob(g))
                 if not os.path.basename(p).startswith(("_temp_model_", "__"))]
        if not paths:
            print(f"\n=== {label} === no files matched {g}")
            continue
        results[label] = census(paths, label)

    if len(results) == 2:
        (la, ca), (lb, cb) = results.items()
        na, nb = sum(ca.values()), sum(cb.values())
        if na and nb:
            fa, fb = ca.get("none", 0) / na, cb.get("none", 0) / nb
            print(f"\n=== {la} -> {lb} ===")
            print(f"  norm-free share {fa:.1%} -> {fb:.1%}   (delta {fb-fa:+.1%})")


if __name__ == "__main__":
    raise SystemExit(main())

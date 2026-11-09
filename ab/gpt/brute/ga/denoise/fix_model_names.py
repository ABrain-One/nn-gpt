"""Rename denoise models so '_' does not appear in the model name, and move their stat dirs.

WHY. LEMUR encodes a training-stat directory as <task>_<dataset>_<metric>_<model> joined by
`config_splitter = '_'` (Const.py:89) and parses it back with a plain 4-way split
(Util.py:61, `tuple(c.split(config_splitter))`). Any '_' inside a model name therefore adds
segments and the importer fails with "too many values to unpack (expected 4)".

Our 372 LLM models are named `LLM_Pg23_seeded-<hash>` -- two underscores -- so their dirs split
into 6 and every one of their stat files is skipped on import.

SAFE TO DO AS A PURE RENAME. Verified before writing this: the model name does not appear inside
its own .py, nor anywhere in the stat JSON payloads (whose keys are accuracy/batch/best_accuracy/
metric_psnr/... and whose only id field, `uid`, refers to the prm row, not the model). So nothing
inside any file needs rewriting.

WHAT THIS DOES NOT FIX -- and it must be said plainly: 111 further dirs are broken by the DATASET
name, not the model name. `img-denoising_denoise_awgn_psnr_DenoiseGA-<hash>` splits into 5 because
the dataset is `denoise_awgn`. Renaming models cannot help those; they need the dataset renamed to
`denoise-awgn` or the dirs withdrawn. That is a semantic decision about published data, so this
script reports them and changes nothing.

NOT TOUCHED: `ast-layer_type-AlexNet-*` (9 models). They also contain '_' and would break the same
parser, but they are upstream classification models, not ours.

    python fix_model_names.py            # dry run: mapping + collision check
    python fix_model_names.py --apply
"""
import argparse
import collections
import os
import re
import sys

REPO = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset"
NN = os.path.join(REPO, "ab", "nn", "nn")
STAT = os.path.join(REPO, "ab", "nn", "stat", "train")
# only OUR denoise producers; ast-layer_* etc. are upstream and out of scope
OURS = ("LLM_",)


def new_name(old):
    return old.replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    models = sorted(f[:-3] for f in os.listdir(NN)
                    if f.endswith(".py") and f.startswith(OURS) and "_" in f[:-3])
    print(f"models to rename: {len(models)}")
    if models:
        print(f"  e.g. {models[0]}  ->  {new_name(models[0])}")

    # a rename that lands on an existing name would silently merge two models' stats
    existing = {f[:-3] for f in os.listdir(NN) if f.endswith(".py")}
    coll = [m for m in models if new_name(m) in existing]
    if coll:
        sys.exit(f"ABORT: {len(coll)} renames collide with existing files, e.g. {coll[:3]}")
    tgt = collections.Counter(new_name(m) for m in models)
    dup = [k for k, v in tgt.items() if v > 1]
    if dup:
        sys.exit(f"ABORT: {len(dup)} target names are not unique, e.g. {dup[:3]}")
    print("  collision check: OK (no target exists, all targets unique)")

    # stat dirs keyed by model, matched on the metric segment rather than by splitting on '_'
    bymodel = collections.defaultdict(list)
    for d in os.listdir(STAT):
        m = re.search(r"_(psnr|psnrrgb|ssimrgb)_(.+)$", d)
        if d.startswith("img-denoising") and m:
            bymodel[m.group(2)].append(d)

    moves_py, moves_dir = [], []
    for m in models:
        moves_py.append((os.path.join(NN, m + ".py"), os.path.join(NN, new_name(m) + ".py")))
        for d in bymodel.get(m, []):
            # replace ONLY the trailing model segment, never the task/dataset/metric prefix
            nd = d[: len(d) - len(m)] + new_name(m)
            moves_dir.append((os.path.join(STAT, d), os.path.join(STAT, nd)))
    print(f"  .py renames:       {len(moves_py)}")
    print(f"  stat dir renames:  {len(moves_dir)}")

    missing = [m for m in models if not bymodel.get(m)]
    if missing:
        print(f"  NOTE {len(missing)} model(s) have no stat dir; the .py is still renamed")

    if a.apply:
        done = 0
        for src, dst in moves_py + moves_dir:
            if not os.path.exists(src):
                continue
            if os.path.exists(dst):
                sys.exit(f"ABORT: destination already exists: {dst}")
            os.rename(src, dst)
            done += 1
        print(f"\n  renamed {done} path(s)")
    else:
        print("\nDRY RUN -- rerun with --apply")

    # the OTHER cause, reported not fixed
    five = [d for d in os.listdir(STAT)
            if d.startswith("img-denoising") and len(d.split("_")) == 5]
    if five:
        ds = collections.Counter("_".join(d.split("_")[1:3]) for d in five)
        print(f"\n  STILL BROKEN, different cause: {len(five)} dirs whose DATASET contains '_'")
        print(f"    dataset segment: {dict(ds)}")
        print(f"    e.g. {five[0]}")
        print("    Renaming models cannot fix these. Either rename the dataset to 'denoise-awgn'")
        print("    or withdraw the dirs. Left untouched -- it is a decision about published data.")

    bad = [d for d in os.listdir(STAT)
           if d.startswith("img-denoising") and len(d.split("_")) != 4]
    print(f"\n  denoise stat dirs still not splitting into 4: {len(bad)}")


if __name__ == "__main__":
    main()

"""Build a small, SYMLINKED MIDD subset so 40 architectures can be ranked in GPU-hours.

WHY A SUBSET. The full pre-tiled MIDD set is 20 sensors x ~34.7k pairs (~694k). The other
session measures ~33 min/epoch on it, so 40 models x 10 epochs would be ~220 GPU-hours. The
research question here is NOT "what absolute PSNR do we reach on MIDD" -- it is "does an
architecture's AWGN ranking predict its ranking on real sensor noise". Rank correlation is
unaffected by training every candidate on the same reduced set, so the subset costs nothing
that matters and buys a ~40x speedup.

WHY SYMLINKS INTO A NEW ROOT, AND NOT AN EDIT TO midd.py. ab/nn/loader/midd.py has a
MIDD_SENSORS filter but no per-sensor cap. Adding one would mean editing a loader that ANOTHER
SESSION's live jobs (midd-rank-x20, midd-x30) import -- and their next launch would silently
pick up different data. That loader's own docstring warns about exactly this failure: a run
that "saw 14 sensors and a 1,221-image test set" produced numbers that were "quietly
incomparable". So nothing is edited and nothing is copied: this builds a parallel tree of
symlinks in the layout midd.py already expects, and the job points MIDD_ROOT at it. Their data
is read-only throughout.

SENSOR DIVERSITY IS THE POINT OF MIDD. Sampling 6 sensors across four vendors keeps the
"real noise, many devices" property that distinguishes MIDD from synthetic AWGN. Taking one
sensor and more patches from it would be cheaper and would answer a different, weaker question.

  python midd_subset.py --train-per-sensor 3000 --test-per-sensor 200
"""
import argparse
import glob
import json
import os
import random

SRC_DEFAULT = "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/data/midd/_patches20"
# One per vendor family where possible; sorted() elsewhere keeps the choice reproducible.
SENSORS_DEFAULT = ("Sony_IMX766", "Sony_IMX586", "ISOCELL_GN1",
                   "ISOCELL_3T2", "OmniVision_OV64B", "Hynix_SL846")


def link_split(src_sensor, dst_sensor, split, cap, rng):
    ndir = os.path.join(src_sensor, split, "original")
    cdir = os.path.join(src_sensor, split, "denoised")
    if not (os.path.isdir(ndir) and os.path.isdir(cdir)):
        return 0
    # 'blurry' files are excluded by the authors' evaluate.py and by midd.py's _collect; drop
    # them here too so the cap is spent on usable pairs rather than silently discarded ones.
    clean = {f for f in os.listdir(cdir) if "blurry" not in f}
    names = sorted(f for f in os.listdir(ndir) if "blurry" not in f and f in clean)
    rng.shuffle(names)                      # spread across scenes, not the first N filenames
    names = names[:cap]
    n = 0
    for side, sdir in (("original", ndir), ("denoised", cdir)):
        out = os.path.join(dst_sensor, split, side)
        os.makedirs(out, exist_ok=True)
        for f in names:
            dst = os.path.join(out, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(sdir, f), dst)
        n = len(names)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    # TEST PAIRS LIVE IN A DIFFERENT TREE. _patches20 is pre-tiled TRAINING patches only --
    # every sensor there has training_set/ and no test_set/, which is why the first build
    # produced 18,000 train pairs and 0 test pairs. The evaluation images are the full-res
    # originals under data/midd/<sensor>/test_set/.
    ap.add_argument("--test-src", default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/data/midd")
    ap.add_argument("--dst", default="/shared/ssd/home/b-b-ahmed/thesis/sweep_out/midd_subset")
    ap.add_argument("--sensors", default=",".join(SENSORS_DEFAULT))
    # MATCHED TO MAI, WHICH IS THE SMALLEST OF THE THREE DATASETS: 658 train / 50 val pairs.
    # A cross-dataset comparison run at MIDD's or SIDD's native size would confound the noise
    # source with the data budget -- more training pairs is a different experiment from harder
    # noise. Totals, not per-sensor counts, so adding sensors widens device diversity without
    # changing how much data any model sees.
    ap.add_argument("--train-total", type=int, default=658)
    ap.add_argument("--test-total", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    want = [s.strip() for s in a.sensors.split(",") if s.strip()]
    have = {os.path.basename(p) for p in glob.glob(os.path.join(a.src, "*"))}
    missing = [s for s in want if s not in have]
    if missing:
        print(f"WARNING sensors not present, skipping: {missing}")
    want = [s for s in want if s in have]
    if not want:
        raise SystemExit(f"no requested sensors found under {a.src}")

    os.makedirs(a.dst, exist_ok=True)
    tot_tr = tot_te = 0
    manifest = {"src": a.src, "test_src": a.test_src, "seed": a.seed, "sensors": {},
                "train_total": a.train_total, "test_total": a.test_total}
    # Spread the total evenly, giving the remainder to the first sensors, so no sensor is
    # silently dropped when the total does not divide by the sensor count.
    k = len(want)
    tr_caps = [a.train_total // k + (1 if i < a.train_total % k else 0) for i in range(k)]
    te_caps = [a.test_total // k + (1 if i < a.test_total % k else 0) for i in range(k)]
    for i, s in enumerate(want):
        tr = link_split(os.path.join(a.src, s), os.path.join(a.dst, s),
                        "training_set", tr_caps[i], rng)
        te = link_split(os.path.join(a.test_src, s), os.path.join(a.dst, s),
                        "test_set", te_caps[i], rng)
        manifest["sensors"][s] = {"train": tr, "test": te}
        tot_tr += tr
        tot_te += te
        print(f"  {s:<22} train {tr:>6}  test {te:>5}")

    # midd.py reads corrupt*.json from MIDD_ROOT to skip pairs that fail to decode. Without
    # these the subset would re-hit files the other session already identified as broken.
    for j in ("corrupt.json", "corrupt_train.json"):
        srcj = os.path.join(os.path.dirname(a.src), j)
        dstj = os.path.join(a.dst, j)
        if os.path.exists(srcj) and not os.path.lexists(dstj):
            os.symlink(srcj, dstj)
            print(f"  linked {j}")

    manifest["total_train"], manifest["total_test"] = tot_tr, tot_te
    json.dump(manifest, open(os.path.join(a.dst, "subset_manifest.json"), "w"), indent=1)
    print(f"\n{len(want)} sensors | {tot_tr} train pairs | {tot_te} test pairs -> {a.dst}")
    print(f"matched to MAI: 658 train / 50 val")
    print("\nUse with:  MIDD_ROOT=<dst> MIDD_PATCH_DIR=.  (the subset IS the patch tree)")


if __name__ == "__main__":
    main()

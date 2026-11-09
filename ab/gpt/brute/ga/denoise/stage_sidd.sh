#!/usr/bin/env bash
# Extract SIDD from the NAS archive into a usable tree. No GPU.
#
# WHY THIS IS NEEDED. nn-dataset/sidd/ currently holds 320 BROKEN symlinks pointing at
# /home/s493053/thesis/... -- the pre-migration path from before the tree moved to
# /shared/ssd/home/b-b-ahmed. `ls` shows 320 entries and 490K, so it looks populated; every
# read fails. The real data is /shared/hdd/data/denoising_datasets.zip (25 GB, owner
# a-f-vasluianu, world-readable).
#
# STAGED TO A NEW DIRECTORY, NOT OVER THE OLD ONE. nn-dataset/sidd is inside the repo and its
# broken links are still referenced by ab/nn/loader/sidd.py's hardcoded
# SIDD_ROOT = os.path.join(os.getcwd(), 'sidd'). Jobs run from /scratch/root with symlinks
# into the repo, so pointing /scratch/root/sidd at THIS tree needs no edit to the loader and
# destroys nothing -- if the staging is wrong, the old tree is still there to compare against.
#
# NAME MAPPING. The archive uses input_crops/target_crops; the loader expects original/
# denoised. sidd.py sorts both sides and pairs by index, and raises on a count mismatch, so
# the two directories must stay in lockstep -- hence the count assertion at the end rather
# than trusting the unzip.
set -uo pipefail

ZIP=/shared/hdd/data/denoising_datasets.zip
DEST=${DEST:-/a/mm/data_sidd}
LOG=${LOG:-/a/mm/sweep_out/sidd_stage.log}
mkdir -p "$DEST" "$(dirname "$LOG")"

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
say "=== staging SIDD from $ZIP -> $DEST ==="
[ -r "$ZIP" ] || { say "FATAL cannot read $ZIP"; exit 1; }

# HEREDOC MUST BIND TO python3, NOT TO tee. Written as
#     python3 - "$ZIP" "$DEST" 2>&1 | tee -a "$LOG" <<'PY'
# bash attaches the heredoc to the LAST command in the pipeline -- tee -- which then printed
# the Python source to the log while python3 read an empty stdin and did nothing. The job
# "succeeded" through every step and staged 0 files. Placing the heredoc before the pipe binds
# it to python3, where it belongs.
python3 - "$ZIP" "$DEST" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, zipfile, time
zpath, dest = sys.argv[1], sys.argv[2]
# train crops are the training pool; val crops are held out by the loader's own index split,
# so both are staged into one tree and sidd.py partitions it. The .mat validation blocks are
# the OFFICIAL SIDD benchmark and are kept separately for a like-for-like published number.
MAP = {"train/SIDD/input_crops": "original", "train/SIDD/target_crops": "denoised",
       "val/SIDD/input_crops": "val_original", "val/SIDD/target_crops": "val_denoised"}
CAP = int(os.environ.get("SIDD_CAP", "3000"))   # per directory; 658 needed + margin
z = zipfile.ZipFile(zpath)
todo = []
# EXTRACTING ALL 30,609 PAIRS WOULD BE WASTED WORK. The cross-dataset comparison is run at
# MATCHED SIZE against MAI, which has only 658 train / 50 val pairs, so SIDD contributes 658.
# A cap keeps the unzip to ~1 min instead of pulling 25 GB for data that is then discarded.
# Members are taken in archive order, which is scene order, so the cap is a contiguous slice
# rather than a random sample -- the equal-size subset built later re-samples properly.
seen = {}
for i in z.infolist():
    if i.is_dir():
        continue
    for src, dst in MAP.items():
        if i.filename.startswith(src + "/"):
            if seen.get(dst, 0) >= CAP:
                break
            seen[dst] = seen.get(dst, 0) + 1
            todo.append((i, os.path.join(dest, dst, os.path.basename(i.filename))))
            break
    else:
        if i.filename.startswith("test/SIDD/") and i.filename.endswith(".mat"):
            todo.append((i, os.path.join(dest, "test", os.path.basename(i.filename))))
print(f"{len(todo)} members to extract", flush=True)
t0 = time.time()
for n, (info, out) in enumerate(todo):
    if n % 5000 == 0 and n:
        print(f"  [{n}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == info.file_size:
        continue                      # resumable: a 25 GB unzip should not restart from zero
    with z.open(info) as fh, open(out, "wb") as w:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            w.write(b)
print(f"extracted {len(todo)} in {time.time()-t0:.0f}s", flush=True)
PY

for d in original denoised val_original val_denoised; do
  say "  $d: $(ls "$DEST/$d" 2>/dev/null | wc -l) files"
done
NO=$(ls "$DEST/original" 2>/dev/null | wc -l); NC=$(ls "$DEST/denoised" 2>/dev/null | wc -l)
# sidd.py raises on a mismatch mid-run; catching it here turns a dead training job into a
# staging error with an obvious cause.
[ "$NO" -eq "$NC" ] && [ "$NO" -gt 0 ] || { say "FATAL pair count mismatch: $NO vs $NC"; exit 1; }
say "  test blocks: $(ls "$DEST/test" 2>/dev/null | wc -l)"
say "=== SIDD staged: $NO pairs ==="

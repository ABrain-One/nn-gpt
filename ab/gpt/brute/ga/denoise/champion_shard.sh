#!/usr/bin/env bash
# CHAMPION RETRAIN: the 8 best NPU-resident architectures under a PROPER training budget.
#
# WHY THE GAP IS A TRAINING GAP, NOT AN ARCHITECTURE GAP. Our best model reaches 37.042 dB
# against LiteDenoiseNet's published 37.660 -- a 0.618 dB deficit. The two recipes are not
# comparable:
#   * They distil from a 41.6M-parameter teacher at alpha=0.9, which their own ablation
#     credits with closing a 1.63 dB student-teacher gap down to 0.05 dB. We distil from
#     nothing.
#   * Our loader samples ONE 256x256 crop per image per epoch, so a 50-epoch run sees
#     658 x 50 = 32,900 samples -- about 0.61% of the available tile space. At K=16 the same
#     50 epochs see 526,400. The corpus is deliberately under-trained for search throughput,
#     and every stored row inherits that.
# So before concluding the architectures are 0.6 dB worse, the architectures have to be given
# a training budget comparable to the one the published figure used.
#
# DENOISE_TRAIN_CROPS RAISES TRAINING SIGNAL ONLY. The loader's own note: train crops are
# "safe to raise -- more gradient signal, metric untouched", unlike DENOISE_VAL_CROPS which
# changes what PSNR means. So these numbers remain comparable to the published full-resolution
# protocol, and are NOT comparable to our own K=1 corpus rows -- which is why this runs in its
# own namespace and does not write the shared corpus.
#
# WHAT WOULD FALSIFY THE PREMISE. If K=16 at 50 epochs leaves the champions at ~37.0, the
# deficit is architectural and no amount of training budget closes it; the honest paper then
# reports 0.6 dB short and says why. That is a real possible outcome of this run.
set -uo pipefail
IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-4}"
EP="${EPOCHS:-50}"
CROPS="${CROPS:-16}"
T=/a/mm
GA=$T/nn-gpt/ab/gpt/brute/ga/denoise
OUT=${OUT_DIR:-$T/sweep_out/champion}
WORK=/scratch/repo
LOG="$OUT/shard_${IDX}.log"
mkdir -p "$OUT/ckpt" "$OUT/curves"
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "champion shard $IDX/$NSHARDS  epochs=$EP  train_crops=$CROPS"
mkdir -p "$WORK"
# EXCLUDE THE HEAVY TREES, NOT JUST `train`.
# Three attempts at this run died "Evicted -- scratch exceeds the limit" at 40, 60 and 120Gi,
# the last filling in 11 minutes with an empty log -- i.e. death DURING the rsync. The cause
# was `data/midd`, another session's MIDD dataset living inside the repo and large enough to
# defeat any scratch limit. `--exclude='train'` never covered it. We exclude it and the other
# bulk trees rather than raising the limit again; `train` and `validation` are symlinked from
# shared storage below, so nothing needed is lost.
rsync -a --exclude='train' --exclude='validation' --exclude='data' --exclude='out' \
      --exclude='results' --exclude='sidd' --exclude='logs' --exclude='.git' \
      --exclude='_temp_model_*' --exclude='__pycache__' \
      "$T/nn-dataset/" "$WORK/" >>"$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && [ $rc -ne 23 ] && [ $rc -ne 24 ] && { log "FATAL rsync $rc"; exit 1; }

# SYMLINK THE DATA, DO NOT LET THE LOADER RE-DOWNLOAD IT.
# The rsync above excludes `train`, so the denoise loader would fetch the training set
# into $WORK/train on every pod. At 658 full-resolution pairs plus the zip that filled the
# 60Gi EmptyDir and the first two attempts at this run were EVICTED, not OOMKilled --
# "Usage of EmptyDir volume scratch exceeds the limit". The data is already on shared
# storage; pointing at it costs nothing and removes the failure mode entirely.
ln -sfn "$T/nn-dataset/train" "$WORK/train"
ln -sfn "$T/nn-dataset/validation" "$WORK/validation"
log "data symlinked: $(ls "$WORK/train/original" | wc -l) training pairs, no download"

export PYTHONPATH="$WORK"
export DENOISE_TRAIN_CROPS="$CROPS"       # the lever
export DENOISE_VAL_CROPS=1                # metric untouched, stays comparable

# WEIGHTS MUST LEAVE THE POD. save_if_best writes to Const.ckpt_dir, NOT to the save_path
# passed to check_nn -- which is how the first completed run produced crop PSNR for four
# models and zero checkpoints, 33 GPU-hours for a number we cannot evaluate on the published
# metric. LEMUR_CKPT_DIR exists for exactly this; its own comment says weights under a
# pod-local root "die with the pod".
export LEMUR_CKPT_DIR="$OUT/ckpt"
# And keep EVERY epoch, not just the crop-metric best. Util.save_if_best documents that the
# 256-crop selector can inverse-rank against full resolution (two seeds at 37.908/37.920 on
# crops were 37.927/37.562 at full res -- order reversed). The epoch this run should be judged
# on is the one that is best under the PUBLISHED metric, which is only knowable after the
# full-resolution pass, so the discarded epochs have to still exist.
export LEMUR_SAVE_EVERY_EPOCH=1
mkdir -p "$OUT/ckpt"
cd "$WORK"

python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json, os, sys
sys.path.insert(0, "$WORK")
import ab.nn.api as api
sub = json.load(open("$OUT/subset.json"))
models = sub["models"]
mine = models[$IDX::$NSHARDS]
print("shard $IDX trains %d of %d: %s" % (len(mine), len(models), [m[:28] for m in mine]), flush=True)
done_path = "$OUT/result_shard_${IDX}.jsonl"
done = set()
if os.path.exists(done_path):
    for ln in open(done_path):
        try:
            done.add(json.loads(ln)["model"])
        except Exception:
            pass
import sqlite3
con = sqlite3.connect("file:$T/nn-dataset/db/ab.nn.db?mode=ro", uri=True)
code = {n: c for n, c in con.execute("select name, code from nn")}
canon = {n.replace("_", "-"): n for n in code if "_" in n}
for m in mine:
    if m in done:
        print("  skip %s (already recorded)" % m[:34], flush=True)
        continue
    src = code.get(m) or code.get(canon.get(m, ""))
    if not src:
        print("  MISSING SOURCE %s" % m[:34], flush=True)
        continue
    # EXACTLY the corpus recipe, read back from the prm table of a stored DnSeedGA row:
    # batch 16, lr 1e-4, transform "denoise". Nothing else is set, because adding momentum
    # or dropout would make this a different recipe and the comparison would not isolate
    # the training-signal change we are testing. transform="denoise" is what all 4,000
    # denoise rows use; the name I first guessed does not take the same arguments.
    prm = dict(batch=16, lr=1e-4, transform="denoise", epoch=$EP)
    try:
        nm, acc, _, _ = api.check_nn(src, "img-denoising", "denoise", "psnr", prm,
                                     save_to_db=False, prefix="Champ",
                                     save_path="$OUT/curves/%s" % m,
                                     save_pth_weights=True)
        rec = {"model": m, "psnr_crop": (acc or 0) * 48, "epochs": $EP, "crops": $CROPS}
    except Exception as e:
        rec = {"model": m, "error": "%s: %s" % (type(e).__name__, str(e)[:160])}
    with open(done_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("  %s -> %s" % (m[:34], json.dumps(rec)[:120]), flush=True)
print("SHARD $IDX DONE", flush=True)
PYEOF
log "shard $IDX finished"

#!/usr/bin/env bash
# DIAGNOSTIC: find what fills /scratch during a K=16 epoch.
# Three attempts at the champion run died Evicted at 40, 60 and 120Gi. The repo is 400MB and
# the dataset is symlinked, so something inside training is writing tens of GB. This runs ONE
# epoch with a disk watcher sampling the largest directories every 20s, then stops.
set -uo pipefail
T=/a/mm
GA=$T/nn-gpt/ab/gpt/brute/ga/denoise
OUT=$T/sweep_out/champion
WORK=/scratch/repo
LOG="$OUT/diag.log"
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='.git' --exclude='_temp_model_*' --exclude='__pycache__' \
      "$T/nn-dataset/" "$WORK/" >>"$LOG" 2>&1
ln -sfn "$T/nn-dataset/train" "$WORK/train"
ln -sfn "$T/nn-dataset/validation" "$WORK/validation"
log "after rsync: $(du -sh /scratch 2>/dev/null | cut -f1)"

# background disk watcher -- names the culprit instead of leaving us to guess again
( while true; do
    tot=$(du -sx /scratch 2>/dev/null | cut -f1)
    log "WATCH /scratch=${tot}KB  top: $(du -sx /scratch/* 2>/dev/null | sort -rn | head -3 | awk '{printf "%s=%sKB ", $2, $1}')"
    sleep 20
  done ) &
W=$!
trap "kill $W 2>/dev/null" EXIT

export PYTHONPATH="$WORK"
export DENOISE_TRAIN_CROPS=16
export DENOISE_VAL_CROPS=1
cd "$WORK"
python3 - <<'PYEOF' 2>&1 | tail -25 | tee -a "$LOG"
import json, sqlite3, sys
sys.path.insert(0, "/scratch/repo")
import ab.nn.api as api
m = json.load(open("/a/mm/sweep_out/champion/subset.json"))["models"][0]
con = sqlite3.connect("file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro", uri=True)
code = dict(con.execute("select name, code from nn"))[m]
print("ONE EPOCH, K=16, weights OFF:", m[:40], flush=True)
prm = dict(batch=16, lr=1e-4, momentum=0.9, dropout=0.0,
           transform="norm_256_flip", epoch=1)
nm, acc, _, _ = api.check_nn(code, "img-denoising", "denoise", "psnr", prm,
                             save_to_db=False, prefix="Diag", save_pth_weights=False)
print("RESULT acc=%s -> %.3f dB" % (acc, (acc or 0) * 48), flush=True)
PYEOF
log "final: $(du -sh /scratch 2>/dev/null | cut -f1)"
log "DIAG DONE"

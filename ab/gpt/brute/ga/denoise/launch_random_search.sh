#!/usr/bin/env bash
# PHASE-1 BASELINE: random search in the SAME space, under the SAME gates, at matched budget.
#
# This is the comparison a NAS paper is rejected for omitting. Without it "our pipeline finds
# better models" is unsupported, because the space and the seed pool might be doing all the work
# and the guided search none of it. Random search is notoriously hard to beat, which is exactly
# why reviewers ask.
#
# WHAT IS HELD IDENTICAL to launch_run5.sh -- everything except the guidance:
#   * the seed pool, byte for byte (same --seeds and --seed-files)
#   * BRUTE_WEIGHTS, so mutation operators are drawn with the same probabilities
#   * --skip-ops --skip-prob 0.3
#   * --diversity-gate --diversity-graded --graded-max 0.95   <- the novelty gate STAYS ON
#   * --edge-filter --mac-cap 9053                            <- the edge gate STAYS ON
#   * --cost-metric/--cost-mode/--cost-weight, so reported fitness is on the same scale
#   * --epoch 5, the same screening depth
#
# WHY THE GATES STAY ON, which is not optional. Two reasons.
#   1. Fairness. The GA was gated, so an ungated baseline would be a different experiment: it
#      would be allowed to spend its budget on near-duplicates and edge-illegal models that the
#      GA was never allowed to propose.
#   2. Corpus hygiene. Random search has no novelty pressure and will propose far more near-
#      duplicates than the GA. Ungated, it would flood the distinct pool -- the arch_uid2 count
#      is the dataset's headline number and it must not be inflated by a baseline run.
#   A consequence worth reporting rather than hiding: because both arms are gated, "equal budget"
#   is equal ATTEMPTS, and the two arms will differ in how many attempts survive to become
#   distinct admitted models. That ratio is itself a result -- record attempts AND admissions.
#
# WHAT IS REMOVED -- the guidance, and only the guidance:
#   --gens 1        one generation IS i.i.d. random sampling: ga_core.Population.initialize()
#                   draws each individual independently from the search space, and no selection
#                   has happened yet. 30 generations of 50 is what makes the GA a GA.
#   --elitism 0     no winner is carried forward
#   --tournament 1  a tournament of size 1 is a uniform random pick, i.e. zero fitness pressure
#                   (moot at --gens 1, set explicitly so the intent is on the record)
#   no --crossover, no --diverse-pairing: both are generational mechanisms with nothing to act on
#   in a single generation.
#
# BUDGET. Production was --pop-size 50 --gens 30 --epoch 5 = 1500 candidate evaluations. This
# script runs 1500 total across SHARDS pods, so the two arms see the same number of 5-epoch
# trainings. Sharding is over independent populations, which is sound here precisely because
# there is no selection: 8 independent random samples of 188 pool to one random sample of 1504.
# That would be invalid for the GA, where generations depend on each other.
#
# PREFIX. --prefix DenoiseRS is mandatory. run_seed_ga.py hardcoded prefix="DenoiseGA" until now;
# reusing it would name baseline models identically to GA models and silently corrupt every
# producer-split analysis we have (on-device tables, arm comparison, dataset manifest).
#
# RNG. run_seed_ga.py sets no fixed seed, so each pod's populations come from OS entropy and the
# shards differ. The population checksums are logged so that assumption is checkable rather than
# assumed -- if two shards ever report identical checksums the pooled sample is not 1500.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
SHARDS="${SHARDS:-8}"
TOTAL="${TOTAL:-1500}"
EPOCH="${EPOCH:-5}"
MAC_CAP="${MAC_CAP:-9053}"

POP=$(( (TOTAL + SHARDS - 1) / SHARDS ))

export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0

SRC=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NN=$SRC/ab/nn/nn
OUT="${OUT:-/a/mm/sweep_out/randsearch}"
WORK="${SCRATCH:-/scratch}/rs_${IDX}"
LOG="$OUT/shard${IDX}.log"
mkdir -p "$OUT"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== RANDOM SEARCH shard=$IDX/$SHARDS pop=$POP epoch=$EPOCH $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

# private root + private DB copy, same as every other trainer here: the pod must never write the
# shared database, and check_nn(save_to_db=True) writes to whatever ab_root_path points at.
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
# The private DB lives on the SHARED mount, one directory per shard, and $WORK/db is a symlink
# to it -- the same arrangement launch_ga_variant.sh uses. It was originally on /scratch and
# copied out at the end, which meant a pod dying at generation N lost all N generations: the
# emptyDir goes with the pod. Per-shard paths mean no two writers ever touch the same file.
mkdir -p "$OUT/shard_db/${IDX}"
if [ ! -f "$OUT/shard_db/${IDX}/ab.nn.db" ]; then
  python3 -c "
import sqlite3,sys
a=sqlite3.connect(sys.argv[1]); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()
" "$SRC/db/ab.nn.db" "$OUT/shard_db/${IDX}/ab.nn.db" >>"$LOG" 2>&1 || {
    log "FATAL db copy failed"; exit 1; }
fi
rm -rf "$WORK/db"
ln -sfn "$OUT/shard_db/${IDX}" "$WORK/db"

cd "$WORK"; export PYTHONPATH="$WORK"
PY=python

"$PY" "$D/run_seed_ga.py" \
  --prefix DenoiseRS \
  --seeds DenoiseUnet3F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
  --seed-files "$D"/new_seeds/DenoiseChanShuffle.py "$D"/new_seeds/DenoiseDilatedFlat.py \
               "$D"/new_seeds/DenoiseELUCtx.py "$D"/new_seeds/DenoiseInstNorm.py \
               "$D"/new_seeds/DenoiseLapPyr.py "$D"/new_seeds/DenoiseLinearMix.py \
               "$D"/new_seeds/DenoiseLPPool.py "$D"/new_seeds/DenoiseMBConv.py \
               "$D"/new_seeds/DenoiseParallelMR.py "$D"/new_seeds/DenoisePoolDeconv.py \
               "$D"/new_seeds/DenoiseReflectPad.py "$D"/new_seeds/DenoiseSelfAttn.py \
               "$D"/new_seeds/DenoiseStatNorm.py "$D"/new_seeds/DenoiseSubPixel.py \
               "$D"/promoted_pool_r3/73506d1871e8f17d.py \
               "$D"/promoted_pool_r3/3fb2f2e558e6ef1e.py \
               "$D"/promoted_pool_r3/beae6349fce5da0c.py \
               "$D"/promoted_pool_r3/dad0218709203ada.py \
               "$NN"/DenoiseNAFNet.py \
               "$NN"/DenoiseRestormerLite.py \
  --pop-size "$POP" --gens 1 --epoch "$EPOCH" \
  --elitism 0 --tournament 1 \
  --skip-ops --skip-prob 0.3 \
  --diversity-gate --diversity-graded --graded-max 0.95 \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 8 \
  --checkpoint "$OUT/rs_${IDX}_checkpoint.pkl" \
  --cache      "$OUT/rs_${IDX}_cache.json" \
  --models-dir "$OUT/rs_${IDX}_models" \
  --results    "$OUT/rs_${IDX}_results.json" \
  --save-to-db >>"$LOG" 2>&1

rc=$?
# No copy-out step: the private DB was written straight to $OUT/shard_db/$IDX all along, so the
# arm's rows are already on shared storage and can be merged deliberately later, never implicitly.
n=$(python3 -c "
import json,os
p='$OUT/rs_${IDX}_results.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)
" 2>/dev/null || echo 0)
log "=== shard $IDX done rc=$rc, $n trial(s) recorded $(date -Is) ==="

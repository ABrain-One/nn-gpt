#!/usr/bin/env bash
# PHASE-1 BASELINE, TAKE 2: random WALK -- isolates selection pressure, and actually spends the budget.
#
# WHY THE FIRST ATTEMPT (launch_random_search.sh, --gens 1) IS NOT USABLE AS A BUDGET-MATCHED
# BASELINE. It asked for 1504 i.i.d. samples and delivered 88 scored trials, 11 per shard. The logs
# say why, over and over:
#     mutate DenoiseNAFNet: only seen-before variants after 8 tries -> reusing one
#     mutate DenoiseSelfAttn: no distinct variant in 8 tries -> parent unchanged
# With one generation every candidate is exactly ONE mutation away from a seed, and that
# neighbourhood holds only ~11 distinct variants that clear the diversity gate. A population of 188
# collapsed to 11. Worse, shard 0's winner was `DenoiseRestormerLite (from seed)` at 34.278 dB -- an
# UNMUTATED SEED. Random search never beat its own starting point, because it could not move.
#
# That is a real finding and it is kept (it explains that the GA's advantage is iterated
# COMPOSITION, not better sampling of a large space), but it cannot support "GA beats random at
# equal compute" because the compute was never spent.
#
# WHAT THIS RUN CHANGES -- one thing only, the selection rule:
#     --gens 30 --pop-size 50     same generational structure as production, so mutations COMPOUND
#     --tournament 1              a tournament of size 1 is a uniform random pick => zero fitness
#                                 pressure. This is the whole experiment.
#     --elitism 0                 no winner is carried forward
# Everything else is byte-identical to launch_run5.sh: seed pool, BRUTE_WEIGHTS, --skip-ops
# --skip-prob 0.3, --diversity-gate --diversity-graded --graded-max 0.95, --edge-filter
# --mac-cap 9053, the cost function, --epoch 5. So the contrast is selection pressure and nothing
# else -- which is precisely the GA's contribution.
#
# 50 x 30 = 1500 evaluations, matching production. Because generations now DEPEND on each other,
# this must NOT be sharded into independent populations the way --gens 1 was; each shard runs its
# own full 30-generation walk and the shards are replicates, not slices of one budget.
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
# CROSSOVER IS OFF HERE, deliberately. launch_run5.sh passes --crossover block and
# --diverse-pairing --mate-pool 6; both are selection-driven recombination, so leaving them on
# would smuggle guidance back into the control. The contrast is therefore mutation-with-selection
# (production) against mutation-without-selection (this run), which is the cleanest reading of
# "what does the GA's selection pressure buy".
#
# PREFIX. --prefix DenoiseRW is mandatory. run_seed_ga.py hardcoded prefix="DenoiseGA" until now;
# reusing it would name baseline models identically to GA models and silently corrupt every
# producer-split analysis we have (on-device tables, arm comparison, dataset manifest).
#
# RNG. run_seed_ga.py sets no fixed seed, so each pod's populations come from OS entropy and the
# shards differ. The population checksums are logged so that assumption is checkable rather than
# assumed -- if two shards ever report identical checksums the pooled sample is not 1500.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
SHARDS="${SHARDS:-8}"   # independent REPLICATE walks, not slices of one budget
POP="${POP:-50}"
GENS="${GENS:-30}"
EPOCH="${EPOCH:-5}"
MAC_CAP="${MAC_CAP:-9053}"


export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0

SRC=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NN=$SRC/ab/nn/nn
OUT="${OUT:-/a/mm/sweep_out/randwalk}"
WORK="${SCRATCH:-/scratch}/rw_${IDX}"
LOG="$OUT/shard${IDX}.log"
mkdir -p "$OUT"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== RANDOM WALK shard=$IDX/$SHARDS pop=$POP gens=$GENS epoch=$EPOCH $(date -Is) ==="
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
mkdir -p "$WORK/db"
python3 -c "
import sqlite3,sys
a=sqlite3.connect(sys.argv[1]); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()
" "$SRC/db/ab.nn.db" "$WORK/db/ab.nn.db" >>"$LOG" 2>&1 || { log "FATAL db copy failed"; exit 1; }

cd "$WORK"; export PYTHONPATH="$WORK"
PY=python

"$PY" "$D/run_seed_ga.py" \
  --prefix DenoiseRW \
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
  --pop-size "$POP" --gens "$GENS" --epoch "$EPOCH" \
  --elitism 0 --tournament 1 \
  --skip-ops --skip-prob 0.3 \
  --diversity-gate --diversity-graded --graded-max 0.95 \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 8 \
  --checkpoint "$OUT/rw_${IDX}_checkpoint.pkl" \
  --cache      "$OUT/rw_${IDX}_cache.json" \
  --models-dir "$OUT/rw_${IDX}_models" \
  --results    "$OUT/rw_${IDX}_results.json" \
  --save-to-db >>"$LOG" 2>&1

rc=$?
# export the private DB's new rows so the arm can be merged deliberately later, never implicitly
mkdir -p "$OUT/shard_db"
cp "$WORK/db/ab.nn.db" "$OUT/shard_db/shard${IDX}.db" 2>/dev/null || \
  log "WARN could not copy shard DB out"
n=$(python3 -c "
import json,os
p='$OUT/rw_${IDX}_results.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)
" 2>/dev/null || echo 0)
log "=== shard $IDX done rc=$rc, $n trial(s) recorded $(date -Is) ==="

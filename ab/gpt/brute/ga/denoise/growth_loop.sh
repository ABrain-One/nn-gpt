#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# SELF-SUSTAINING seed-growth loop (full-train-gated promotion) — the automated
# growth loop the manual runs kept skipping (Run 4 was never promoted).
#
# For each run N (starting from the CURRENTLY-RUNNING run), in order:
#   1. WAIT for run N to finish (its supervisor keeps it alive; we don't relaunch R5).
#   2. FULL-TRAIN its top-K offspring @50ep           (fulltrain_topk.py, resumable)
#   3. PROMOTE the psnr50-survivors -> promoted_pool_rN (promote.py: WL-distinct + floor
#      + deployable + graded near-dup dedup vs the pool they'll join).
#   4. LAUNCH run N+1 with the pool GROWN by promoted_pool_rN, teardown-resilient.
# Idempotent: skips any stage whose output already exists. Bounded by MAXRUN.
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NN=/a/mm/nn-dataset/ab/nn/nn
PY=python
LOG="$D/growth_loop.log"
MAC_CAP=9053
BAR=34.0          # psnr50 floor to earn a seed slot (competitive-good)
K=10              # top-K offspring per run to full-train
MAXRUN=10         # run R5..R10 this campaign (extendable)
log(){ echo "[growth $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- pool assembly (DRY): echo the seed args for run N ----
# base = 5 registry + 14 new_seeds + 4 promoted_r3 + NAFNet + Restormer (= Run 5's pool);
# run N>=6 additionally includes promoted_pool_r5 .. promoted_pool_r{N-1}.
NEW_SEEDS=(ChanShuffle DilatedFlat ELUCtx InstNorm LapPyr LinearMix LPPool MBConv \
           ParallelMR PoolDeconv ReflectPad SelfAttn StatNorm SubPixel)
# Run 4/5 used EXACTLY these 4 promoted-r3 seeds; 8901067b + e11d622b were evicted
# rejects, DELIBERATELY excluded (do NOT glob promoted_pool_r3/*.py).
PROMOTED_R3=(73506d1871e8f17d 3fb2f2e558e6ef1e beae6349fce5da0c dad0218709203ada)
pool_seed_args(){ # $1 = run number
  local n="$1" f
  printf -- '--seeds DenoiseUnet3F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet '
  printf -- '--seed-files '
  for f in "${NEW_SEEDS[@]}"; do printf '%s ' "$D/new_seeds/Denoise$f.py"; done
  for f in "${PROMOTED_R3[@]}"; do printf '%s ' "$D/promoted_pool_r3/$f.py"; done
  printf '%s %s ' "$NN/DenoiseNAFNet.py" "$NN/DenoiseRestormerLite.py"
  local m
  for ((m=5; m<n; m++)); do
    for f in "$D"/promoted_pool_r$m/*.py; do [ -e "$f" ] && printf '%s ' "$f"; done
  done
}

ga_args(){ # common GA hyperparams (identical to Run 4/5)
  printf -- '--pop-size 50 --gens 30 --epoch 5 --elitism 5 --tournament 3 --crossover block '
  printf -- '--skip-ops --skip-prob 0.3 --diversity-gate --diversity-graded --graded-max 0.95 '
  printf -- '--diverse-pairing --mate-pool 6 \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  '
  printf -- '--edge-filter --mac-cap %s --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 --num-workers 8 ' "$MAC_CAP"
}

wait_for_run(){ # $1 = run number: block until seed_ga_R{N} writes its DONE marker
  local n="$1"; local rlog="$D/run$n.log"
  log "waiting for run R$n to finish ..."
  while true; do
    grep -qa "=== DONE in" "$rlog" 2>/dev/null && { log "R$n DONE"; return 0; }
    sleep 60
  done
}

launch_run(){ # $1 = run number: teardown-resilient GA launch with the grown pool
  local n="$1"; local rlog="$D/run$n.log"
  if grep -qa "=== DONE in" "$rlog" 2>/dev/null; then log "R$n already done — skip launch"; return 0; fi
  local args; args="$(pool_seed_args "$n") $(ga_args) \
    --checkpoint $D/seed_ga_R${n}_checkpoint.pkl --cache $D/seed_ga_R${n}_cache.json \
    --models-dir $D/seed_ga_R${n}_models --results $D/seed_ga_R${n}_results.json --save-to-db"
  local attempt
  for attempt in $(seq 1 200); do
    if pgrep -f "seed_ga_R${n}_checkpoint.pkl" >/dev/null 2>&1; then sleep 60; continue; fi
    log "launch R$n attempt $attempt"
    # shellcheck disable=SC2086
    "$PY" "$D/run_seed_ga.py" $args >> "$rlog" 2>&1
    grep -qa "=== DONE in" "$rlog" 2>/dev/null && { log "R$n complete"; return 0; }
    log "R$n exited (rc=$?) — resume in 30s"; sleep 30
  done
}

promote_run(){ # $1 = run number: full-train top-K then promote -> promoted_pool_r{N}
  local n="$1"; local emit="$D/promoted_pool_r$n"; local res="$D/fulltrain_R$n.json"
  if [ -d "$emit" ] && ls "$emit"/*.py >/dev/null 2>&1; then log "promoted_pool_r$n exists — skip"; return 0; fi
  log "full-train top-$K of R$n @50ep ..."
  "$PY" "$D/fulltrain_topk.py" --arms "R$n" --k "$K" --epoch 50 --num-workers 8 --results "$res" >> "$LOG" 2>&1
  log "fulltrain R$n exit $? — promoting (psnr50 bar $BAR) ..."
  # shellcheck disable=SC2046
  "$PY" "$D/promote.py" --results "$res" --models-dirs "$D/seed_ga_R${n}_models" \
    --pool $(pool_seed_args "$n" | sed 's/--seeds//;s/--seed-files//') \
    --pareto --bar "$BAR" --strict-mac --mac-cap "$MAC_CAP" --graded --graded-max 0.95 \
    --emit-dir "$emit" --out "$D/promote_r$n.json" >> "$LOG" 2>&1
  log "promote R$n -> $(ls "$emit"/*.py 2>/dev/null | wc -l) new seed(s) in promoted_pool_r$n"
}

log "=== growth loop start (R5..R$MAXRUN, K=$K, bar=$BAR) ==="
for ((N=5; N<=MAXRUN; N++)); do
  [ "$N" -gt 5 ] && launch_run "$N"     # R5 is already running under run5_supervisor
  wait_for_run "$N"
  promote_run "$N"
done
log "=== growth loop done through R$MAXRUN ==="

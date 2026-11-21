#!/bin/bash
# Ablation requested by Dr. Ignatov (20 Nov 2026): the same experiment with ONE update type only.
#   PPO only : 4 cycles x (PPO -> merge -> PPO -> merge)     = 8 PPO phases
#   GRPO only: 4 cycles x (GRPO -> merge -> GRPO -> merge)   = 8 GRPO phases
# Everything else is identical to the final PPO+GRPO run (run_final.sh v3, qwen_cycle.py v5): same base model, prompts,
# scaled reward, settings, seeds 0 1 2, budget (8 phases x 16 steps) and a frozen evaluation after EVERY phase.
# Order: ppo seed0, grpo seed0, ppo seed1, grpo seed1, ppo seed2, grpo seed2 (balanced if stopped early).
# The script WAITS until run_final.sh has released the GPU lock, and starts only if the final run is complete
# (seed 2 evaluated after cycle 4). No automatic retries; resumable (finished phases, merges, evals are skipped).
# Usage: nohup ./run_ablation.sh > run_ablation.log 2>&1 &        (optional: RUNS="ppo:0 grpo:0" ./run_ablation.sh)
echo "===== ABLATION waiting for the GPU lock $(date) ====="
exec 9>/tmp/umair_qwen_cycles.lock; flock 9
cd ~/nn-gpt-main
[ -f out_qwen/final/seed2/eval_c4_grpo/DONE ] || { echo "===== STOP: final run not complete (no seed2 eval_c4_grpo/DONE) - ablation not started $(date) ====="; exit 1; }
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_SAMPLER=0
EXPECTED_MD5=8e118c2c10824660ea2c45598f0f48fb
[ "$(md5sum qwen_cycle.py | cut -d' ' -f1)" = "$EXPECTED_MD5" ] || { echo "STOP: qwen_cycle.py is not v5"; exit 1; }
ROOT=$HOME/nn-gpt-main/out_qwen/ablation; mkdir -p $ROOT
SET="ENT_COEF=0 CLIP_HIGH=0.2 MASK_TRUNC=1 BEST_WINDOW=4 STOP_ENT_FACTOR=3 STOP_ENT_ABS=1.2 REWARD_MODE=scaled R_OFFSET=0.5 R_SCALE=3 R_CENTER=0.6"
cleanup () { pkill -u $USER -f "VLLM::EngineCore" 2>/dev/null; sleep 5; }
need_disk () {
  free=$(df --output=avail -BG $ROOT | tail -1 | tr -dc 0-9)
  [ "$free" -ge 25 ] || { echo "===== STOP: only ${free} GB free disk (need 25) $(date) ====="; exit 1; }
}
phase () {   # name algo base   (one attempt; uses $O and $S)
  if [ -f $O/$1/DONE ]; then echo "skip $1 (done)"; return 0; fi
  cleanup; rm -rf $O/$1
  echo "===== START $A seed$S $1 algo=$2 base=$3 $(date) ====="
  env $SET SEED=$S ALGO=$2 BASE=$3 RUN_DIR=$O/$1 .venv/bin/python qwen_cycle.py > $O/log_$1.log 2>&1
  s=$?; echo "===== END $A seed$S $1 exit=$s $(date) ====="
  [ $s -eq 0 ] && [ -f $O/$1/DONE ] || { echo "===== STOP: $A seed$S $1 failed, see $O/log_$1.log ====="; return 1; }
  echo "===== $A seed$S $1: skipped updates = $(grep -c 'update skipped' $O/log_$1.log), repeated updates = $(grep -c 'OOM in update' $O/log_$1.log), entropy stop = $(grep -c 'STOP: entropy' $O/log_$1.log) ====="
  echo "===== $A seed$S $1: $(grep -a 'reward limits in this phase' $O/log_$1.log | sed 's/.*phase: //') ====="
}
merge () {   # base adapter out
  if [ -f $3/config.json ]; then echo "skip merge $3 (done)"; return 0; fi
  need_disk; cleanup
  .venv/bin/python merge_qwen.py $1 $2 $3 cuda > $3.merge.log 2>&1 || { echo "GPU merge failed - CPU merge"; rm -rf $3; .venv/bin/python merge_qwen.py $1 $2 $3 cpu >> $3.merge.log 2>&1; }
  [ -f $3/config.json ] || return 1
  echo "===== MERGED $3 $(date) ====="
}
feval () {   # tag model
  E=$O/eval_$1; if [ -f $E/DONE ]; then echo "skip eval $1 (done)"; return 0; fi
  mkdir -p $E; cleanup
  echo "===== START eval $A seed$S $1 $(date) ====="
  MODEL=$2 MAX_NEW=4096 .venv/bin/python gen_qwen.py ~/frozen_prompts_v3.json $E/run0.jsonl ~/frozen_prompts_v3.json $E/run1.jsonl ~/frozen_prompts_v3.json $E/run2.jsonl > $E/gen.log 2>&1 || return 1
  cleanup
  ( cd ~/nn-gpt && source .venv/bin/activate && python check_strict_safe.py $E/run0.jsonl $E/run1.jsonl $E/run2.jsonl > $E/strict.log 2>&1 \
    && python train_valid.py $E/run0_strict.jsonl $E/run1_strict.jsonl $E/run2_strict.jsonl > $E/train.log 2>&1 ) || return 1
  date > $E/DONE; echo "===== EVAL $A seed$S $1 DONE: $(grep -a 'strict OK' $E/strict.log | grep -o 'OK [0-9]*/[0-9]*' | tr '\n' ' ') $(date) ====="
}

echo "===== ABLATION START $(date) | qwen_cycle.py md5 $EXPECTED_MD5 | runs ${RUNS:-ppo:0 grpo:0 ppo:1 grpo:1 ppo:2 grpo:2} ====="
for R in ${RUNS:-ppo:0 grpo:0 ppo:1 grpo:1 ppo:2 grpo:2}; do
  A=${R%%:*}; S=${R##*:}
  O=$ROOT/$A/seed$S; mkdir -p $O
  [ -f $O/eval_c4_b/DONE ] && { echo "skip $A seed $S (done)"; continue; }
  CUR=Qwen/Qwen3.5-9B
  for c in 1 2 3 4; do
    phase c${c}_a $A $CUR                                       || exit 1
    merge $CUR $O/c${c}_a/adapter $O/merged_c${c}_a             || exit 1
    [ "$CUR" != "Qwen/Qwen3.5-9B" ] && rm -rf $CUR
    feval c${c}_a $O/merged_c${c}_a                             || exit 1
    phase c${c}_b $A $O/merged_c${c}_a                          || exit 1
    merge $O/merged_c${c}_a $O/c${c}_b/adapter $O/merged_c${c}  || exit 1
    feval c${c}_b $O/merged_c${c}                               || exit 1
    rm -rf $O/merged_c${c}_a
    CUR=$O/merged_c${c}
  done
  rm -rf $CUR
  echo "===== RUN $A seed $S DONE $(date) ====="
done
echo "===== ABLATION DONE $(date) ====="

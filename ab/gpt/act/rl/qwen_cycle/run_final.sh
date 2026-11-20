#!/bin/bash
# Final experiment: 3 seeds x 4 cycles (PPO -> merge -> GRPO -> merge) from Qwen3.5-9B, with the final method:
#   accuracy-scaled reward (valid: 0.5 + 3 x (accuracy - 0.6); invalid penalties unchanged), no entropy bonus,
#   stage-best checkpoint (4-step window), entropy stops (3x step 1 or 1.2), memory-bounded update that waits for free
#   GPU memory and repeats an update instead of skipping it, and (v5) every generated network built and trained in a
#   separate worker process with limits of 1800 s per training run and 12 GiB RAM (qwen_cycle.py v5).
# Frozen evaluation after EVERY phase (100 prompts x 3 sampling runs, strict check, valid networks trained once).
# SEED sets the LoRA and value-head initialisation and the sampling seeds of every step.
# No automatic retries: a crash stops the run (resumable: finished phases, merges and evaluations are skipped on restart).
# Usage: nohup ./run_final.sh > run_final.log 2>&1 &        (optional: SEEDS="1 2" ./run_final.sh)
exec 9>/tmp/umair_qwen_cycles.lock; flock -n 9 || { echo "QWEN CYCLES ALREADY RUNNING"; exit 1; }
cd ~/nn-gpt-main
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_SAMPLER=0
EXPECTED_MD5=8e118c2c10824660ea2c45598f0f48fb
[ "$(md5sum qwen_cycle.py | cut -d' ' -f1)" = "$EXPECTED_MD5" ] || { echo "STOP: qwen_cycle.py is not v5"; exit 1; }
ROOT=$HOME/nn-gpt-main/out_qwen/final; mkdir -p $ROOT
SET="ENT_COEF=0 CLIP_HIGH=0.2 MASK_TRUNC=1 BEST_WINDOW=4 STOP_ENT_FACTOR=3 STOP_ENT_ABS=1.2 REWARD_MODE=scaled R_OFFSET=0.5 R_SCALE=3 R_CENTER=0.6"
cleanup () { pkill -u $USER -f "VLLM::EngineCore" 2>/dev/null; sleep 5; }
need_disk () {
  free=$(df --output=avail -BG $ROOT | tail -1 | tr -dc 0-9)
  [ "$free" -ge 25 ] || { echo "===== STOP: only ${free} GB free disk (need 25) $(date) ====="; exit 1; }
}
phase () {   # name algo base   (one attempt; uses $O and $S)
  if [ -f $O/$1/DONE ]; then echo "skip $1 (done)"; return 0; fi
  cleanup; rm -rf $O/$1
  echo "===== START seed$S $1 algo=$2 base=$3 $(date) ====="
  env $SET SEED=$S ALGO=$2 BASE=$3 RUN_DIR=$O/$1 .venv/bin/python qwen_cycle.py > $O/log_$1.log 2>&1
  s=$?; echo "===== END seed$S $1 exit=$s $(date) ====="
  [ $s -eq 0 ] && [ -f $O/$1/DONE ] || { echo "===== STOP: seed$S $1 failed, see $O/log_$1.log ====="; return 1; }
  echo "===== seed$S $1: skipped updates = $(grep -c 'update skipped' $O/log_$1.log), repeated updates = $(grep -c 'OOM in update' $O/log_$1.log), entropy stop = $(grep -c 'STOP: entropy' $O/log_$1.log) ====="
  echo "===== seed$S $1: $(grep -a 'reward limits in this phase' $O/log_$1.log | sed 's/.*phase: //') ====="
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
  echo "===== START eval seed$S $1 $(date) ====="
  MODEL=$2 MAX_NEW=4096 .venv/bin/python gen_qwen.py ~/frozen_prompts_v3.json $E/run0.jsonl ~/frozen_prompts_v3.json $E/run1.jsonl ~/frozen_prompts_v3.json $E/run2.jsonl > $E/gen.log 2>&1 || return 1
  cleanup
  ( cd ~/nn-gpt && source .venv/bin/activate && python check_strict_safe.py $E/run0.jsonl $E/run1.jsonl $E/run2.jsonl > $E/strict.log 2>&1 \
    && python train_valid.py $E/run0_strict.jsonl $E/run1_strict.jsonl $E/run2_strict.jsonl > $E/train.log 2>&1 ) || return 1
  date > $E/DONE; echo "===== EVAL seed$S $1 DONE: $(grep -a 'strict OK' $E/strict.log | grep -o 'OK [0-9]*/[0-9]*' | tr '\n' ' ') $(date) ====="
}

echo "===== FINAL RUN START $(date) | qwen_cycle.py md5 $EXPECTED_MD5 | seeds ${SEEDS:-0 1 2} ====="
for S in ${SEEDS:-0 1 2}; do
  O=$ROOT/seed$S; mkdir -p $O
  [ -f $O/eval_c4_grpo/DONE ] && { echo "skip seed $S (done)"; continue; }
  CUR=Qwen/Qwen3.5-9B
  for c in 1 2 3 4; do
    phase c${c}_ppo ppo $CUR                                   || exit 1
    merge $CUR $O/c${c}_ppo/adapter $O/merged_c${c}_ppo        || exit 1
    [ "$CUR" != "Qwen/Qwen3.5-9B" ] && rm -rf $CUR             # previous cycle's model is no longer needed (adapters are kept)
    feval c${c}_ppo $O/merged_c${c}_ppo                        || exit 1
    phase c${c}_grpo grpo $O/merged_c${c}_ppo                  || exit 1
    merge $O/merged_c${c}_ppo $O/c${c}_grpo/adapter $O/merged_c${c}   || exit 1
    feval c${c}_grpo $O/merged_c${c}                           || exit 1
    rm -rf $O/merged_c${c}_ppo
    CUR=$O/merged_c${c}
  done
  rm -rf $CUR                                                  # final model of the seed (rebuildable from the 8 adapters)
  echo "===== SEED $S DONE $(date) ====="
done
echo "===== FINAL RUN DONE $(date) ====="

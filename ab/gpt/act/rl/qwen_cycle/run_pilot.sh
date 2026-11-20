#!/bin/bash
# Pilot: one cycle (PPO -> merge -> GRPO -> merge) from Qwen3.5-9B with the fixed settings and the accuracy-scaled
# reward (valid network: 0.5 + 3 x (accuracy - 0.6); invalid penalties unchanged), memory-bounded update (v4).
# Frozen evaluation after each phase. No automatic retries: any crash stops the pilot so that it can be inspected.
# Decision rule, fixed before the run (compared with the base model and with cycle 1 of the first run):
#   accept the scaled reward for the final runs if, after the pilot cycle, the strict-valid rate is >= 60% and the
#   mean accuracy on the common prompts is not significantly below the base model.
exec 9>/tmp/umair_qwen_cycles.lock; flock -n 9 || { echo "QWEN CYCLES ALREADY RUNNING"; exit 1; }
cd ~/nn-gpt-main
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_SAMPLER=0
O=$HOME/nn-gpt-main/out_qwen/pilot; mkdir -p $O
SET="ENT_COEF=0 CLIP_HIGH=0.2 MASK_TRUNC=1 BEST_WINDOW=4 STOP_ENT_FACTOR=3 STOP_ENT_ABS=1.2 REWARD_MODE=scaled R_OFFSET=0.5 R_SCALE=3 R_CENTER=0.6 SEED=0"
cleanup () { pkill -u $USER -f "VLLM::EngineCore" 2>/dev/null; sleep 5; }
need_disk () {
  free=$(df --output=avail -BG $O | tail -1 | tr -dc 0-9)
  [ "$free" -ge 25 ] || { echo "===== STOP: only ${free} GB free disk (need 25) $(date) ====="; exit 1; }
}
phase () {   # name algo base env   (one attempt only)
  if [ -f $O/$1/DONE ]; then echo "skip $1 (done)"; return 0; fi
  cleanup; rm -rf $O/$1
  echo "===== START $1 algo=$2 base=$3 env=[$4] $(date) ====="
  env $4 ALGO=$2 BASE=$3 RUN_DIR=$O/$1 .venv/bin/python qwen_cycle.py > $O/log_$1.log 2>&1
  s=$?; echo "===== END $1 exit=$s $(date) ====="
  [ $s -eq 0 ] && [ -f $O/$1/DONE ] || { echo "===== STOP: $1 failed, see $O/log_$1.log ====="; return 1; }
  n=$(grep -c "OOM in update" $O/log_$1.log); echo "===== $1: skipped updates (OOM) = $n ====="
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
  echo "===== START frozen eval $1 model=$2 $(date) ====="
  MODEL=$2 MAX_NEW=4096 .venv/bin/python gen_qwen.py ~/frozen_prompts_v3.json $E/run0.jsonl ~/frozen_prompts_v3.json $E/run1.jsonl ~/frozen_prompts_v3.json $E/run2.jsonl > $E/gen.log 2>&1 || return 1
  cleanup
  ( cd ~/nn-gpt && source .venv/bin/activate && python check_strict_safe.py $E/run0.jsonl $E/run1.jsonl $E/run2.jsonl > $E/strict.log 2>&1 \
    && python train_valid.py $E/run0_strict.jsonl $E/run1_strict.jsonl $E/run2_strict.jsonl > $E/train.log 2>&1 ) || return 1
  date > $E/DONE; echo "===== EVAL $1 DONE $(date) ====="
}

echo "===== PILOT START $(date) | qwen_cycle.py md5 $(md5sum qwen_cycle.py | cut -c1-12) ====="
phase p1_A_ppo ppo Qwen/Qwen3.5-9B "$SET"                          || exit 1
merge Qwen/Qwen3.5-9B $O/p1_A_ppo/adapter $O/merged_p1_A            || exit 1
feval p1A $O/merged_p1_A                                             || exit 1
phase p1_B_grpo grpo $O/merged_p1_A "$SET"                          || exit 1
merge $O/merged_p1_A $O/p1_B_grpo/adapter $O/merged_p1               || exit 1
rm -rf $O/merged_p1_A
feval p1 $O/merged_p1                                                || exit 1
echo "===== PILOT DONE $(date) ====="

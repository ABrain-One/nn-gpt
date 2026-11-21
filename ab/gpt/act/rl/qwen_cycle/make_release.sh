#!/bin/bash
# Collect everything needed for the release / pull request of the Qwen self-improvement experiments
# (code, generated networks, statistics, all logs). READ-ONLY on the experiment: it only copies finished files
# (phases and evaluations with a DONE marker) and never touches running processes, adapters or merged models.
# Usage: ~/make_release.sh            -> ~/qwen_release.tar.gz
set -u
R=$HOME/qwen_release; rm -rf $R; mkdir -p $R/code $R/runs
cd $HOME/nn-gpt-main
# ---- code (exact versions used) ----
for f in qwen_cycle.py gen_qwen.py merge_qwen.py run_final.sh run_ablation.sh run_pilot.sh test_memory.py test_v5_gpu.py; do
  [ -f $f ] && cp $f $R/code/
done
for f in check_strict_safe.py train_valid.py analyze_collapse.py ab/gpt/TunePPO.py; do
  [ -f $HOME/nn-gpt/$f ] && cp $HOME/nn-gpt/$f $R/code/
done
cp $HOME/frozen_prompts_v3.json $R/code/ 2>/dev/null
md5sum $R/code/* > $R/code/MD5SUMS
# ---- final run (all seeds, finished parts only) ----
cp run_final.log $R/runs/ 2>/dev/null
for S in out_qwen/final/seed*; do
  s=$(basename $S); mkdir -p $R/runs/final/$s
  for P in $S/c*_ppo $S/c*_grpo; do
    [ -f $P/DONE ] || continue
    p=$(basename $P); mkdir -p $R/runs/final/$s/$p
    cp $P/steps.jsonl $P/generations.jsonl $P/best.json $P/config.json $P/DONE $R/runs/final/$s/$p/ 2>/dev/null
    cp $S/log_$p.log $R/runs/final/$s/ 2>/dev/null
  done
  for E in $S/eval_*; do
    [ -f $E/DONE ] || continue
    e=$(basename $E); mkdir -p $R/runs/final/$s/$e
    cp $E/run?.jsonl $E/run?_strict.jsonl $E/run?_acc.jsonl $E/gen.log $E/strict.log $E/train.log $E/DONE $R/runs/final/$s/$e/ 2>/dev/null
  done
  cp $S/*.merge.log $R/runs/final/$s/ 2>/dev/null
done
# ---- stopped first start (v4c) and base model evaluation ----
mkdir -p $R/runs/stopped_v4c $R/runs/base
cp out_qwen/stopped_v4c/run_final_v4c.log $R/runs/stopped_v4c/ 2>/dev/null
cp out_qwen/stopped_v4c/seed0/log_*.log $R/runs/stopped_v4c/ 2>/dev/null
cp $HOME/qwen_baseline/qwen_v3_run?*.jsonl $R/runs/base/ 2>/dev/null
cp $HOME/qwen_baseline/*.log $R/runs/base/ 2>/dev/null
# ---- summary ----
echo "== release contents"; du -sh $R/code $R/runs/*; find $R -type f | wc -l
cd $HOME && tar czf qwen_release.tar.gz qwen_release && ls -la --block-size=M qwen_release.tar.gz

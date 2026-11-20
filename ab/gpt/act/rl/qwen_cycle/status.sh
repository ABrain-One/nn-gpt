#!/bin/bash
# Status of the final run and the ablation (read-only).
cd ~/nn-gpt-main
echo "== finished phases (final run, seeds 1-2) =="; grep -E "EVAL|: skipped|: train_timeout|STOP|SEED" run_final.log | grep -v seed0
if [ -f run_ablation.log ]; then
  echo "== ablation =="; grep -E "EVAL|: skipped|: train_timeout|STOP|RUN |ABLATION" run_ablation.log | tail -12
fi
f=$(ls -t out_qwen/final/seed*/log_*.log out_qwen/ablation/*/seed*/log_*.log 2>/dev/null | head -1)
echo "== current training log: $f =="; grep -a "^\[CYCLE\] step" $f | tail -2 | cut -c1-200
echo "== reward events in this phase: $(grep -a -c '\[REWARD\]' $f)"
e=$(ls -td out_qwen/final/seed*/eval_* out_qwen/ablation/*/seed*/eval_* 2>/dev/null | head -1)
if [ -n "$e" ] && [ ! -f $e/DONE ]; then
  echo "== eval running: $e (train.log last written $(date -r $e/train.log +%H:%M 2>/dev/null || echo '-'))"
fi
echo "== processes =="; pgrep -af "run_final|run_ablation|qwen_cycle|train_valid|check_strict|gen_qwen" | cut -c1-90 | head -6
echo "== disk =="; df -BG ~ | tail -1
date

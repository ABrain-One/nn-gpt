#!/usr/bin/env bash
LOG=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise/run4.log
dead=0
while true; do
  if grep -qa "Best fitness in Generation 5:" "$LOG" 2>/dev/null; then
    echo "GEN5-DONE"
    grep -aE "Best fitness in Generation [1-5]:|New best overall" "$LOG" | tail -8
    exit 0
  fi
  if pgrep -f "seed_ga_R4_checkpoint[.]pkl" >/dev/null 2>&1 || pgrep -f "run4_supervisor[.]sh" >/dev/null 2>&1; then
    dead=0
  else
    dead=$((dead+1))
    if [ $dead -ge 6 ]; then
      echo "RUN4-STOPPED: no GA and no supervisor for 3min — needs relaunch"
      grep -aE "Generation [0-9]+/30" "$LOG" | tail -1
      exit 1
    fi
  fi
  sleep 30
done

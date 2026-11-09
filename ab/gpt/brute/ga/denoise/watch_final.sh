#!/usr/bin/env bash
LOG=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise/run4.log
dead=0
while true; do
  if grep -qa "Best fitness in Generation 30:\|=== DONE in" "$LOG" 2>/dev/null; then
    echo "RUN4-COMPLETE"; grep -aE "Best fitness in Generation 30:|=== DONE in" "$LOG" | tail -3; exit 0
  fi
  if pgrep -f "seed_ga_R4_checkpoint[.]pkl" >/dev/null 2>&1 || pgrep -f "run4_supervisor[.]sh" >/dev/null 2>&1; then dead=0; else
    dead=$((dead+1)); [ $dead -ge 6 ] && { echo "RUN4-STOPPED needs relaunch"; grep -aE "Generation [0-9]+/30" "$LOG"|tail -1; exit 1; }
  fi
  sleep 30
done

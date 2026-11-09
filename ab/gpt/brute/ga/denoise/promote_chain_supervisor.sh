#!/usr/bin/env bash
# Crash-resilient wrapper for the Run-3 promote chain. Safe to start while the chain is
# already running: it waits, and only relaunches if the chain dies without the DONE marker.
# Resume is lossless-per-model: fulltrain_results.json caches finished checksums.
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/promote_chain.log"
for attempt in $(seq 1 100); do
  if grep -qa "PROMOTE CHAIN DONE" "$LOG" 2>/dev/null; then
    echo "[psup] chain complete (attempt $attempt)" >> "$LOG"; break
  fi
  if pgrep -f "run_promote_chain.sh" >/dev/null 2>&1 || pgrep -f "fulltrain_topk.py" >/dev/null 2>&1; then
    sleep 120; continue
  fi
  echo "[psup] === chain not running, (re)launch attempt $attempt $(date) ===" >> "$LOG"
  bash "$D/run_promote_chain.sh" >> "$LOG" 2>&1
done

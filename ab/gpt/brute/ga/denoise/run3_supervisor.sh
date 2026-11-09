#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for PRODUCTION RUN 3 (graded gate + diverse pairing).
# Guard uses the unique checkpoint path (pgrep -f sees full cmdline; ps -o cmd truncates).
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/run3.log"
if pgrep -f "seed_ga_R3_checkpoint.pkl" >/dev/null 2>&1; then
  echo "[sup3] already running — abort double-launch"; exit 0
fi
for attempt in $(seq 1 200); do
  echo "[sup3] === attempt $attempt $(date) ===" | tee -a "$LOG"
  bash "$D/launch_run3.sh" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== DONE in" "$LOG"; then echo "[sup3] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[sup3] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for PRODUCTION RUN 2 (diversity-gated).
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/production_v2_run.log"
if pgrep -f "run_seed_ga.py.*seed_ga_P2" >/dev/null 2>&1; then
  echo "[sup2] already running — abort double-launch"; exit 0
fi
for attempt in $(seq 1 200); do
  echo "[sup2] === attempt $attempt $(date) ===" | tee -a "$LOG"
  bash "$D/launch_production_v2.sh" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== DONE in" "$LOG"; then echo "[sup2] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[sup2] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

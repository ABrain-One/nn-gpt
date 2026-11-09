#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for the WL-hash medium validation run.
# Resumable (per-gen checkpoint + checksum cache), so re-running continues.
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/wlval_run.log"
# pgrep -f matches the FULL cmdline (ps -o cmd truncates on our 17-file-path command);
# the unique checkpoint path can't appear in the supervisor's own `bash launch_wlval.sh`.
if pgrep -f "seed_ga_WLVAL_checkpoint.pkl" >/dev/null 2>&1; then
  echo "[wlval] already running — abort double-launch"; exit 0
fi
for attempt in $(seq 1 100); do
  echo "[wlval] === attempt $attempt $(date) ===" | tee -a "$LOG"
  bash "$D/launch_wlval.sh" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== DONE in" "$LOG"; then echo "[wlval] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[wlval] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

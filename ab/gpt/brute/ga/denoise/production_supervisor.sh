#!/usr/bin/env bash
# Self-healing supervisor for the denoise PRODUCTION run.
#
# It re-runs launch_production.sh — which RESUMES from seed_ga_P1_checkpoint.pkl (per-generation)
# + seed_ga_P1_cache.json (per-model) — every time the run dies, until it prints "=== DONE in".
# So transient crashes (NaN eval hiccups, OOM, worker teardown) auto-recover WITHOUT losing gens.
#
# Survives: crashes while the compute node is up.
# Does NOT survive: full node/allocation teardown (nothing on the node does). AFTER a teardown,
# just re-run THIS script and it picks up from the last checkpoint:
#     cd /a/mm/nn-dataset
#     nohup bash /a/mm/nn-gpt/ab/gpt/brute/ga/denoise/production_supervisor.sh \
#           >> /a/mm/nn-gpt/ab/gpt/brute/ga/denoise/production_supervisor.log 2>&1 &
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/production_run.log"
MAX=400

# guard: never run two production processes at once
if pgrep -f "run_seed_ga.py.*seed_ga_P1" >/dev/null 2>&1; then
  echo "[supervisor] a production run_seed_ga is already running — attaching as monitor only."
fi

for ((i=1; i<=MAX; i++)); do
  if grep -qa "=== DONE in" "$LOG" 2>/dev/null; then
    echo "[supervisor] run COMPLETE ($(date)) — stopping."; exit 0
  fi
  # don't double-launch if one is somehow still alive
  if pgrep -f "run_seed_ga.py.*seed_ga_P1" >/dev/null 2>&1; then
    sleep 30; continue
  fi
  echo "[supervisor] === launch/resume attempt $i $(date) ==="
  bash "$D/launch_production.sh" >> "$LOG" 2>&1
  echo "[supervisor] launcher exited rc=$? ($(date))"
  grep -qa "=== DONE in" "$LOG" 2>/dev/null && { echo "[supervisor] COMPLETE ($(date))"; exit 0; }
  echo "[supervisor] resuming from checkpoint in 30s ..."
  sleep 30
done
echo "[supervisor] hit MAX=$MAX restarts — stopping; investigate $LOG"

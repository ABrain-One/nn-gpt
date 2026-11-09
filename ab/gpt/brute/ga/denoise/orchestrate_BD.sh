#!/usr/bin/env bash
# Unattended chain: wait for arm B to finish -> auto-launch arm D.
# Self-healing: if either arm's process dies WITHOUT its "=== DONE" marker
# (e.g. transient kill), re-run its launcher, which resumes from the per-gen
# checkpoint + fitness cache (zero recompute). NOTE: survives a plain session
# end (setsid-detached) but NOT a full compute-allocation teardown — if the
# allocation is reclaimed overnight, re-run this script next session to resume.
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
BLOG="$D/seed_ga_B0_run.log"
DLOG="$D/seed_ga_D0_run.log"
STATUS="$D/chain_BD_status.txt"

echo "chain start $(date)" > "$STATUS"

# 1) ensure arm B completes
while ! grep -q "=== DONE" "$BLOG" 2>/dev/null; do
  if ! pgrep -f "run_seed_ga.py.*seed_ga_B0" >/dev/null 2>&1; then
    echo "$(date) arm B not done + not running -> resuming launch_B0.sh" >> "$STATUS"
    nohup bash "$D/launch_B0.sh" >> "$BLOG" 2>&1 &
    sleep 120   # let it boot before re-checking (avoid double-launch)
  fi
  sleep 60
done
echo "$(date) arm B DONE -> launching arm D" >> "$STATUS"

# 2) ensure arm D completes
while ! grep -q "=== DONE" "$DLOG" 2>/dev/null; do
  if ! pgrep -f "run_seed_ga.py.*seed_ga_D0" >/dev/null 2>&1; then
    echo "$(date) arm D not done + not running -> launching/resuming launch_D0.sh" >> "$STATUS"
    nohup bash "$D/launch_D0.sh" >> "$DLOG" 2>&1 &
    sleep 120
  fi
  sleep 60
done
echo "$(date) CHAIN COMPLETE: arm B and arm D both done" >> "$STATUS"

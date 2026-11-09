#!/bin/bash
while true; do
  grep -qa "=== DONE in" production_v2_run.log 2>/dev/null && { echo "RUN2_DONE $(date)"; break; }
  pgrep -f "run_seed_ga.py.*seed_ga_P2" >/dev/null || { grep -qa "=== DONE in" production_v2_run.log 2>/dev/null && echo "RUN2_DONE" || echo "RUN2_PROC_GONE_NO_MARKER $(date)"; break; }
  sleep 60
done

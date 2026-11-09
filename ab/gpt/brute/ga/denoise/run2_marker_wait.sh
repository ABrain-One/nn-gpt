#!/bin/bash
# Robust: exit ONLY when Run 2 prints its real DONE marker. The supervisor
# auto-restarts the GA on any preemption, so proc-gone is NOT an end condition.
while true; do
  grep -qa "=== DONE in" production_v2_run.log 2>/dev/null && { echo "RUN2_DONE $(date)"; break; }
  sleep 60
done

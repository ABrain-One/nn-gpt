#!/bin/bash
# exits only when BOTH tasks print their real completion marker
while true; do
  ft=0; ga=0
  grep -qa "=== RESULT" fulltrain_P1.log 2>/dev/null && ft=1
  grep -qa "=== DONE in" production_v2_run.log 2>/dev/null && ga=1
  if [ "$ft" = 1 ] && [ "$ga" = 1 ]; then echo "BOTH_DONE $(date)"; break; fi
  sleep 120
done

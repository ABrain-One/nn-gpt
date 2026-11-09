#!/usr/bin/env bash
# Milestone watcher: block until run R<RUN> reaches generation >= <TARGET> (or completes),
# then print r5_report.py and exit (so the tracked-task completion notifies the caller).
# Usage: watch_gen.sh <RUN> <TARGET_GEN>
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
RUN="${1:?run number}"; TARGET="${2:?target gen}"
CKPT="$D/seed_ga_R${RUN}_checkpoint.pkl"
RLOG="$D/run${RUN}.log"
read_gen(){ "$PY" -c "import pickle;print(pickle.load(open('$CKPT','rb'))['generation'])" 2>/dev/null || echo 0; }
for i in $(seq 1 960); do
  if grep -qa "=== DONE in" "$RLOG" 2>/dev/null; then echo "[R${RUN} COMPLETE]"; "$PY" "$D/r5_report.py"; exit 0; fi
  g="$(read_gen)"
  if [ "${g:-0}" -ge "$TARGET" ] 2>/dev/null; then echo "[R${RUN} MILESTONE gen>=$TARGET]"; "$PY" "$D/r5_report.py"; exit 0; fi
  sleep 30
done
echo "[R${RUN} watcher timed out]"; "$PY" "$D/r5_report.py"

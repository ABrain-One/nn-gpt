#!/usr/bin/env bash
# Auto-refresh the accuracy-predictor tracker every REFRESH_EVERY generations of the
# production run, appending a timestamped row to predictor_tracking_history.csv.
# CPU-only + nice, so it never competes with the GPU training. Stops when the run finishes.
#
# The tracker is STATELESS (reads the live cache + models-dir), so if this loop dies in a
# teardown it's harmless — just re-launch it; nothing is lost. Survives session expiry (nohup).
#   nohup bash track_autorefresh.sh >> track_autorefresh.log 2>&1 &
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/production_run.log"
REFRESH_EVERY=${REFRESH_EVERY:-3}
export PYTHONPATH=/a/mm/nn-dataset
PY=python

cur_gen() { grep -aoE "Generation [0-9]+/30" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1; }
run_tracker() {
  if pgrep -f track_predictor.py >/dev/null 2>&1; then return; fi   # no overlap
  echo "[track] refresh at gen $1 ($(date))"
  nice -n 15 "$PY" "$D/track_predictor.py" >> "$D/track_autorefresh.log" 2>&1
}

last=-999
for i in $(seq 1 2000); do
  if grep -qa "=== DONE in" "$LOG" 2>/dev/null; then
    echo "[track] run complete — final refresh + stop ($(date))"
    run_tracker "final"; exit 0
  fi
  g=$(cur_gen); g=${g:-0}
  if [ "$g" -ge $((last + REFRESH_EVERY)) ]; then
    run_tracker "$g"; last=$g
  fi
  sleep 180
done

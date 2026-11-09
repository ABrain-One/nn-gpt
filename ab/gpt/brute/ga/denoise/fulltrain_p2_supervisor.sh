#!/usr/bin/env bash
# Crash/teardown-resilient full-train of Run 2's distinct FRONTIER candidates @50ep.
# These are the genuinely-new seed-pool candidates the diversity gate produced:
#   281445839534b390  DenoiseUnet9FXDenoiseUnet9F+m  (NEW family, ~8981 MMac corner)
#   d737ecea229ba3b5  DenoiseUnet3F+m+m+m            (deeper 3F, ~5629 MMac)
#   6392d44d8f2bfedb  LiteDenoiseNet+m+m+s+m+m        (top distinct Lite, ~8574 MMac)
# Resumable by-checksum (seed_ga_P2_fulltrain.json). Feeds promote.py --pareto.
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
LOG="$D/fulltrain_P2.log"
if pgrep -f "fulltrain_topk.py.*--arms P2" >/dev/null 2>&1; then
  echo "[ftp2] already running — abort double-launch"; exit 0
fi
for attempt in $(seq 1 100); do
  echo "[ftp2] === attempt $attempt $(date) ===" | tee -a "$LOG"
  "$PY" "$D/fulltrain_topk.py" --arms P2 --k 0 --epoch 50 \
    --batch 16 --lr 1e-4 --num-workers 8 \
    --include-checksums 281445839534b390 d737ecea229ba3b5 6392d44d8f2bfedb \
    --results "$D/seed_ga_P2_fulltrain.json" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== RESULT" "$LOG"; then echo "[ftp2] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[ftp2] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

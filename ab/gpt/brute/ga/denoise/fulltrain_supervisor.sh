#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for the production top-K full-train.
# fulltrain_topk.py is resumable by checksum (seed_ga_P1_fulltrain.json), so re-running
# continues; this just auto-restarts on crash until it prints "=== RESULT".
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
LOG="$D/fulltrain_P1.log"
if pgrep -f "fulltrain_topk.py.*--arms P1" >/dev/null 2>&1; then
  echo "[fts] already running — abort double-launch"; exit 0
fi
for attempt in $(seq 1 100); do
  echo "[fts] === attempt $attempt $(date) ===" | tee -a "$LOG"
  # TRIMMED (user request): top-4 Lite by eff-dB (already trained, resume-skipped)
  # PLUS the 5 distinct non-Lite winners (1F+s, 3F, 1FX6F+s crossover, 3F+m, 6F).
  # The ~7 remaining untrained Lite near-clones are dropped — behavioural duplicates,
  # ~4h GPU for marginal data. Already-trained Lite stay in the JSON as free data.
  "$PY" "$D/fulltrain_topk.py" --arms P1 --k 4 --epoch 50 \
    --batch 16 --lr 1e-4 --num-workers 8 \
    --include-checksums 276e98587908ce90 900c067ae0dc9c11 e2963188de6ba376 8c8aa2e638fd311c 0b1f4ba0f34beedd \
    --results "$D/seed_ga_P1_fulltrain.json" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== RESULT" "$LOG"; then echo "[fts] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[fts] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

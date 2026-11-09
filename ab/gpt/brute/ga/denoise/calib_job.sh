#!/usr/bin/env bash
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
GEN=/a/mm/sweep_out/llm_gen2
ROOT=/scratch/root
mkdir -p "$ROOT/out" /a/mm/sweep_out/llm_div
for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
cd "$ROOT" || exit 1
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
echo "=== diversity calibration $(date -Is) ==="
PYTHONPATH="$ROOT" python -u "$GA/calibrate_diversity.py" \
    --run-dirs "$GEN/cold/run" "$GEN/seeded/run" \
    --nn-dir "$SRC/ab/nn/nn" --out /a/mm/sweep_out/llm_div
echo "=== done rc=$? $(date -Is) ==="

#!/usr/bin/env bash
# WL-hash + graded-cosine diversity census over the 516 LLM candidates.
# Needs a GPU: candidates default to device='cuda' and call self.to(device) in __init__,
# so arch_embed._load cannot instantiate them CPU-only.
# Pod-local LEMUR root for the same reason the screen shards need one -- arch_embed._load
# writes each model through mutator.lemur_io into out/nn/tmp/ before tracing it.
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
echo "=== diversity census $(date -Is) ==="
PYTHONPATH="$ROOT" python -u "$GA/llm_diversity.py" \
    --run-dirs "$GEN/cold/run" "$GEN/seeded/run" \
    --out /a/mm/sweep_out/llm_div
echo "=== done rc=$? $(date -Is) ==="

#!/usr/bin/env bash
# Ablation ARM C, repeat 0 — arm A + REAL RECOMBINATION.
# IDENTICAL to launch_A0.sh (same 5-seed pool, pop12/6gen/ep5, same component-only
# mutation weights, same edge objective eff=psnr-0.3*ln(MACs/budget) budget=median seed)
# EXCEPT --crossover block (block_crossover.py: child = parent A skeleton + parent B
# conv-block internals). This isolates what recombination adds over mutation-only (arm A).
# save_to_db OFF (professor-gate). Resumable: per-gen checkpoint + fitness cache.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.0,skip:0.0"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
  --pop-size 12 --gens 6 --epoch 5 \
  --crossover block \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 2 \
  --checkpoint "$D/seed_ga_C0_checkpoint.pkl" \
  --cache      "$D/seed_ga_C0_cache.json" \
  --models-dir "$D/seed_ga_C0_models" \
  --results    "$D/seed_ga_C0_results.json"

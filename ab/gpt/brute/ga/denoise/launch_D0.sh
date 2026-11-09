#!/usr/bin/env bash
# Ablation ARM D, repeat 0 — FULL: arm B's new topology operators (scale + skip)
# COMBINED WITH arm C's real recombination (block crossover). Same 5-seed pool,
# pop12/6gen/ep5, same edge objective eff=psnr-0.3*ln(MACs/budget) budget=median seed.
# This is the "everything on" arm: BRUTE_WEIGHTS scale+skip ON (like B) AND
# --crossover block (like C). Interpretable only against B and C (D = B + C).
# save_to_db OFF (professor-gate). Resumable: per-gen checkpoint + fitness cache.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
  --pop-size 12 --gens 6 --epoch 5 \
  --crossover block \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 6 \
  --checkpoint "$D/seed_ga_D0_checkpoint.pkl" \
  --cache      "$D/seed_ga_D0_cache.json" \
  --models-dir "$D/seed_ga_D0_models" \
  --results    "$D/seed_ga_D0_results.json"

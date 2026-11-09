#!/usr/bin/env bash
# Ablation ARM E, repeat 0 — the SKIP arm. GA produces models using the new
# skip_operators (global + intra-block residual; lateral/fusion WIP). Same 5-seed
# pool, pop12/6gen/ep5, same edge objective eff=psnr-0.3*ln(MACs/budget) budget=
# median seed. Isolates what SKIP-CONNECTION insertion adds: --skip-ops routes
# ~half the mutations through skip_operators; crossover none (measure skips alone).
# Motivation: skip richness predicts seed strength (skip_inventory.py); auto-adding
# the global residual already gave 2F +3.41 dB. save_to_db OFF (professor-gate).
# Resumable: per-gen checkpoint + fitness cache.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
# component ops still available; skip handled by --skip-ops (skip_operators module),
# not the planner's global-only 'skip' weight -> keep planner skip weight 0.
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.0,skip:0.0"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
  --pop-size 12 --gens 6 --epoch 5 \
  --crossover none --skip-ops --skip-prob 0.5 \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 2 \
  --checkpoint "$D/seed_ga_E0_checkpoint.pkl" \
  --cache      "$D/seed_ga_E0_cache.json" \
  --models-dir "$D/seed_ga_E0_models" \
  --results    "$D/seed_ga_E0_results.json"

#!/usr/bin/env bash
# Ablation ARM A, repeat 0 — mutation-only, PRE-EXISTING component ops
# (dimension/activation/layer_type/kernel_size/stride); OUR new topology
# operators scale+skip are OFF (they define arm B). crossover=none (atomic).
# Pool = the 5 swappable seeds {1F,6F,7F,9F,10F} shared across ALL arms.
# ep5 proxy, edge-aware objective eff=psnr-0.3*ln(MACs/budget), budget=median seed.
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
  --crossover none \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 2 \
  --checkpoint "$D/seed_ga_A0_checkpoint.pkl" \
  --cache      "$D/seed_ga_A0_cache.json" \
  --models-dir "$D/seed_ga_A0_models" \
  --results    "$D/seed_ga_A0_results.json"

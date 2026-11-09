#!/usr/bin/env bash
# Ablation ARM B, repeat 0 — arm A + OUR NEW TOPOLOGY OPERATORS (scale + skip).
# IDENTICAL to launch_A0.sh (same 5-seed pool, pop12/6gen/ep5, same edge objective
# eff=psnr-0.3*ln(MACs/budget) budget=median seed, crossover none) EXCEPT the
# BRUTE_WEIGHTS now turn ON scale + skip (0 in arm A). This isolates what our own
# operators add over the mutation-only baseline, and is the control D (=B+C) needs.
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
  --crossover none \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 2 \
  --checkpoint "$D/seed_ga_B0_checkpoint.pkl" \
  --cache      "$D/seed_ga_B0_cache.json" \
  --models-dir "$D/seed_ga_B0_models" \
  --results    "$D/seed_ga_B0_results.json"

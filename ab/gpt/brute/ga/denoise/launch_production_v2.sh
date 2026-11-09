#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# PRODUCTION RUN 2 = Run 1 (launch_production.sh) + the functional-CKA DIVERSITY GATE.
# Clean A/B: identical seeds/pop/gens/operators/filters as Run 1; the ONLY change is
# --diversity-gate (reject a generated child whose residual-CKA behaviour is a near-
# duplicate, resCKA >= --cka-max, of an already-accepted model). Fixes Run 1's collapse
# onto the Lite family (291/300 Lite). Namespace seed_ga_P2_* so Run 1 stays intact.
# Smoke-validated: cka_max 0.80 -> 5 families vs Run 1's 1, no starvation.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
MAC_CAP=9053
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
          DenoiseUnet3F \
  --pop-size 50 --gens 30 --epoch 5 \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  --diversity-gate --cka-max 0.80 --diversity-n 16 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 8 \
  --checkpoint "$D/seed_ga_P2_checkpoint.pkl" \
  --cache      "$D/seed_ga_P2_cache.json" \
  --models-dir "$D/seed_ga_P2_models" \
  --results    "$D/seed_ga_P2_results.json" \
  --save-to-db

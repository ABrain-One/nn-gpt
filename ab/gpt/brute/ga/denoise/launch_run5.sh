#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# PRODUCTION RUN 5 — macro-family BROADENING iteration.
# Run 4 hit mutation-exhaustion ("no distinct variant in N tries") at gen ~25-30:
# the 23-seed pool's operator-reachable neighbourhood is tapped out. Run 5 adds TWO
# new edge-legal MACRO-families to lift the distinctness ceiling (dataset-contribution
# goal) — the pool was otherwise all conv U-Nets:
#     DenoiseNAFNet        pure NAFNet (LayerNorm2d/SimpleGate/SCA, no ReLU)  1.13M / 3874 MMac
#     DenoiseRestormerLite MDTA channel-attn + GDFN transformer               0.70M / 7186 MMac*
#   (*conv-only MACs; MDTA attention is linear-in-HW, not in the proxy — both UNDER 9053 cap.)
# All other config identical to Run 4 (pop50 x 30 gens, ep5 proxy, graded diversity gate 0.95,
# diverse pairing, predictor funnel, edge filter + MAC cap 9053, skip-ops, save-to-db).
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NN=/a/mm/nn-dataset/ab/nn/nn
PY=python
MAC_CAP=9053
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet3F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
  --seed-files "$D"/new_seeds/DenoiseChanShuffle.py "$D"/new_seeds/DenoiseDilatedFlat.py \
               "$D"/new_seeds/DenoiseELUCtx.py "$D"/new_seeds/DenoiseInstNorm.py \
               "$D"/new_seeds/DenoiseLapPyr.py "$D"/new_seeds/DenoiseLinearMix.py \
               "$D"/new_seeds/DenoiseLPPool.py "$D"/new_seeds/DenoiseMBConv.py \
               "$D"/new_seeds/DenoiseParallelMR.py "$D"/new_seeds/DenoisePoolDeconv.py \
               "$D"/new_seeds/DenoiseReflectPad.py "$D"/new_seeds/DenoiseSelfAttn.py \
               "$D"/new_seeds/DenoiseStatNorm.py "$D"/new_seeds/DenoiseSubPixel.py \
               "$D"/promoted_pool_r3/73506d1871e8f17d.py \
               "$D"/promoted_pool_r3/3fb2f2e558e6ef1e.py \
               "$D"/promoted_pool_r3/beae6349fce5da0c.py \
               "$D"/promoted_pool_r3/dad0218709203ada.py \
               "$NN"/DenoiseNAFNet.py \
               "$NN"/DenoiseRestormerLite.py \
  --pop-size 50 --gens 30 --epoch 5 \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  --diversity-gate --diversity-graded --graded-max 0.95 \
  --diverse-pairing --mate-pool 6 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 8 \
  --checkpoint "$D/seed_ga_R5_checkpoint.pkl" \
  --cache      "$D/seed_ga_R5_cache.json" \
  --models-dir "$D/seed_ga_R5_models" \
  --results    "$D/seed_ga_R5_results.json" \
  --save-to-db

#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# PRODUCTION RUN 4 — seed-pool GROWTH iteration (see metric_ablation/DECISION_LEDGER.md).
# Identical measured configuration to Run 3, with ONE change: the seed pool is GROWN
# from 19 -> 23 by adding Run-3's 4 PROMOTED Pareto-frontier models as new seeds:
#     73506d18  DenoisePoolDeconvXDenoiseSelfAttn~715b67+m  36.50 dB / 6812 MMac
#     3fb2f2e5  DenoiseELUCtxXDenoiseUnet9F+s~c77e5b+m       36.01 dB / 3331 MMac
#     beae6349  DenoisePoolDeconvXDenoiseSelfAttn~bdc3cf     34.73 dB / 2917 MMac
#     dad02187  DenoiseLapPyrXDenoiseLPPool                  34.31 dB /  916 MMac
# (Evicted rejects 8901067b / e11d622b are DELIBERATELY excluded.)
#
# Carried over unchanged from Run 3:
#   * GRADED diversity gate: exact WL hash + continuous-WL cosine < 0.95 (calibrated;
#     trivial variation >=0.985, distinct <=0.921 -> 0.95 sits in the empty band).
#   * DIVERSE PAIRING: 2nd crossover parent = most-distant of a 6-mate pool.
#   * Fixed block crossover; BN NaN-guard injected into EVERY seed at load.
#   * pop50 x 30 gens, epoch-5 proxy, edge filter + MAC cap 9053, predictor funnel, save-to-db.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
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
  --checkpoint "$D/seed_ga_R4_checkpoint.pkl" \
  --cache      "$D/seed_ga_R4_cache.json" \
  --models-dir "$D/seed_ga_R4_models" \
  --results    "$D/seed_ga_R4_results.json" \
  --save-to-db

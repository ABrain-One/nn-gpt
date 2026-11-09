#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# PRODUCTION RUN 3 — the full measured configuration (see metric_ablation/DECISION_LEDGER.md):
#   * SCORECARD-SELECTED 19-seed 4-family pool (run3_seed_scorecard.json): cuts = 4F/7F/10F
#     (over 9053 MMac cap), ResBN/GroupNorm/LayerNorm (near-clones >=0.97), 1F (sub-floor+covered).
#     Identical-recipe PSNR@5ep for ALL seeds; capacity of full pool: 76 distinct children/gen.
#   * GRADED diversity gate (exact WL hash + continuous-WL cosine < 0.95): rejects
#     near-dups, not just literal clones. Run-2's over-count (3.9x) fixed.
#   * DIVERSE PAIRING: second crossover parent = most-distant of a 6-mate pool
#     (parent-pair cosine 0.797 -> 0.665 measured).
#   * Fixed block crossover (dead-code prune — no gen-29 genome-bloat deadlock).
# pop50 x 30 gens, epoch-5 proxy, edge filter + MAC cap, predictor funnel, save-to-db.
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
  --checkpoint "$D/seed_ga_R3_checkpoint.pkl" \
  --cache      "$D/seed_ga_R3_cache.json" \
  --models-dir "$D/seed_ga_R3_models" \
  --results    "$D/seed_ga_R3_results.json" \
  --save-to-db

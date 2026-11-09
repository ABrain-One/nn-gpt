#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# SMOKE TEST of the full PRODUCTION stack — NOT a real run. Tiny (pop6/2gen/ep1) so
# it finishes fast; its only job is to prove the production config runs END-TO-END on
# GPU with the two capabilities the ablation never exercised together: the accuracy-
# predictor funnel (--predictor) AND the edge deployability filter (--edge-filter),
# alongside block crossover + skip-ops. Throwaway namespace seed_ga_SMOKE_* so it
# CANNOT touch any ablation/production cache. save_to_db OFF. Delete seed_ga_SMOKE_*
# afterwards. Nothing here is committed.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet9F \
  --pop-size 6 --gens 2 --epoch 1 \
  --elitism 1 --tournament 2 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 2 -- DISABLED, see header`  \
  --edge-filter --mac-cap 9053 \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 2 \
  --checkpoint "$D/seed_ga_SMOKE_checkpoint.pkl" \
  --cache      "$D/seed_ga_SMOKE_cache.json" \
  --models-dir "$D/seed_ga_SMOKE_models" \
  --results    "$D/seed_ga_SMOKE_results.json"

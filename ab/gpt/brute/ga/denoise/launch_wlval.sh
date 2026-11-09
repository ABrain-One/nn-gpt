#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# MEDIUM VALIDATION RUN (Run-3 pre-flight): WL-graph-hash diversity gate + grown pool.
# Question it answers: does the validated WL-hash gate + the grown 26-model pool produce
# CROSS-FAMILY diversity, or collapse back onto the Lite basin (which would mean we need
# generator/topology work before a full production Run 3)?
# pop30 x 10 gens (short enough to dodge the gen-29 crossover genome-bloat deadlock),
# epoch 3 proxy. NO --save-to-db (validation only). Namespace seed_ga_WLVAL_*.
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
          DenoiseUnet3F \
  --seed-files "$D"/promoted_pool/*.py \
  --pop-size 30 --gens 10 --epoch 3 --elitism 3 --tournament 3 \
  --crossover block --skip-ops --skip-prob 0.3 \
  --diversity-gate \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap 9053 \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 6 \
  --checkpoint "$D/seed_ga_WLVAL_checkpoint.pkl" \
  --cache "$D/seed_ga_WLVAL_cache.json" \
  --models-dir "$D/seed_ga_WLVAL_models" \
  --results "$D/seed_ga_WLVAL_results.json"

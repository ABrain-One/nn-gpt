#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# DISTINCT-SEED GA, STEP 2: the FULL distinct pool = 24 families (7 DB + 17 distinct new).
# Same config as launch_distinct10.sh, broader seeding for maximum basin coverage toward the
# 2000-distinct-model target. Seeds from the distinct manifest (over-cap + near-clones already
# excluded); load_seeds defends with --seed-mac-cap. Separate namespace (DFULL) so it does not
# collide with the step-1 checkpoint/cache.
# NOTE: git push/commit stays USER-MANUAL. Needs a GPU.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
source "$D/seeds_distinct.sh"
MAC_CAP=9053

"$PY" "$D/run_seed_ga.py" \
  --seeds $DB_SEEDS \
  --seed-files $NEW_DISTINCT_FILES \
  --seed-mac-cap "$MAC_CAP" \
  --pop-size 50 --gens 30 --epoch 5 \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 12 \
  --checkpoint "$D/seed_ga_DFULL_checkpoint.pkl" \
  --cache      "$D/seed_ga_DFULL_cache.json" \
  --models-dir "$D/seed_ga_DFULL_models" \
  --results    "$D/seed_ga_DFULL_results.json" \
  --save-to-db

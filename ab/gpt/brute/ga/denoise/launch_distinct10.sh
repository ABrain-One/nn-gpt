#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# DISTINCT-SEED GA, STEP 1: best-10 distinct families (7 measured DB seeds + MFDNet/Thunder/
# SplitterNet). Same ablation-winner config as launch_production.sh; seeds come from the
# distinct manifest so no over-cap / near-clone seed can enter. load_seeds also defends with
# --seed-mac-cap (default 9053). "Just to try" before the full 24-family run.
# NOTE: git push/commit stays USER-MANUAL. Needs a GPU (save_to_db writes after training).
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
  --seeds $BEST10_DB \
  --seed-files $BEST10_FILES \
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
  --checkpoint "$D/seed_ga_D10_checkpoint.pkl" \
  --cache      "$D/seed_ga_D10_cache.json" \
  --models-dir "$D/seed_ga_D10_models" \
  --results    "$D/seed_ga_D10_results.json" \
  --save-to-db

#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# PRODUCTION RUN (spec: /a/mm/PRODUCTION_RUN.md).
# =============================================================================
# CLEARED TO RUN (2026-07-16): both gates passed —
#   (1) Gate 1 (launch_fulltrain.sh) confirmed crossover>mutation: D0 7FX9F+m 36.31 dB
#       beats arm A (35.38) and the manual 7F baseline (35.56) at 50ep.
#   (2) Prof Dymtro GREENLIT production.
# --save-to-db is now ON (writes to shared LEMUR DB). NaN-guard auto-injected into every
# seed's learn() (run_seed_ga._inject_nan_guard) to keep generation yield >90%.
# NOTE: git push/commit remains USER-MANUAL (never automated) — unrelated to this gate.
# =============================================================================
#
# Config = the ablation WINNER (arm D = block crossover + scale/skip operators),
# scaled up, with the two professor-required capabilities ON that the ablation ran
# WITHOUT: the accuracy-predictor funnel (stage 1) and the edge deployability filter.
#   - Generator:  --crossover block + BRUTE_WEIGHTS scale/skip (arm D)
#   - Pop/gens:   50 / 30, elitism 5 (10%), tournament 3
#   - Objective:  eff = psnr - 0.3*ln(MACs/budget); budget = FIXED mobile MAC cap
#   - Funnel:     --predictor (train top 50% of each gen, skip rest) -> ep5
#   - Deploy:     --edge-filter (+ --mac-cap) rejects non-TFLite / over-cap children
# This is RUN 1 (pool = 5 swappable seeds). Multi-run pool-growth is a later step.
# Resumable: per-gen checkpoint + fitness cache (namespace seed_ga_P1_*).
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python

# Fixed mobile MAC target (MMac). Ablation used median-seed 9052.61 as a relative
# budget; production ties the budget to a real device cap. Tune to the deployment
# target; here we keep the same reference so results stay comparable to the ablation.
MAC_CAP=9053

# Seed pool = 9 (was 5 in the ablation; production grows/diversifies the pool).
#   core (deployable + crossover-compatible):        1F 6F 7F 9F 10F
#   block-donors (undeployable skeleton, but their BLOCK enriches deployable
#     skeletons via crossover — the edge filter auto-rejects their bad-skeleton
#     children at generation time, cheaply):          4F 8F
#   mutation-only diversity (deployable, no crossover block -> seed NEW deployable
#     basins; Lite is the 37.66 dB KD champ = strongest distinct skeleton): 3F LiteDenoiseNet
#   EXCLUDED 5F: undeployable skeleton AND no crossover block -> 100% of its lineage
#     is filtered = guaranteed-zero contribution (dropped on purpose, not an omission).
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
          DenoiseUnet3F \
  --pop-size 50 --gens 30 --epoch 5 \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 12 \
  --checkpoint "$D/seed_ga_P1_checkpoint.pkl" \
  --cache      "$D/seed_ga_P1_cache.json" \
  --models-dir "$D/seed_ga_P1_models" \
  --results    "$D/seed_ga_P1_results.json" \
  --save-to-db   # professor greenlight obtained 2026-07-16 -> persist to shared LEMUR DB

#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# DISTINCT-SEED GA on the MEASURED best 10, one model per family (BEST10_V2_DB).
#
# Differs from launch_distinct10.sh in TWO things -- the seed pool and --diversity-gate.
#
#   seed pool     v1's BEST10 was chosen before the families had measured PSNR; it seeds 6
#                 variants of one U-Net lineage and includes DenoiseUnet3F/6F/1F, which the
#                 canonical sweep later put at the BOTTOM of the corpus (33.18 / 33.76 /
#                 34.37 dB). v2 takes the best measured representative of each distinct
#                 family instead.
#   --diversity-gate  NEW in v2, and it is not a free comparison knob: it rejects children
#                 whose WL exact graph hash collides with an existing member, so v2 explores
#                 a structurally wider population per generation than v1 did. Any v1-vs-v2
#                 difference in yield or best-dB has two causes, not one. (An earlier
#                 revision of this header claimed the seed pool was the only change; it was
#                 wrong.)
#
# Everything else is identical so the runs stay as comparable as those two changes allow.
#
# NOT ENABLED, deliberately:
#   --prof-predictor  the remote accuracy predictor measured rho=-0.140 / MAE 0.553 dB at
#                     n=18 -- 4.7x worse than predicting the corpus mean. Its rho=+0.84 at
#                     n=6 did not survive the larger sample.
#   --predictor       the built-in surrogate is kept (as in v1) since it only re-ranks
#                     within a generation and falls back to measurement.
#
# --epoch 5 is inherited from v1 for cost, but read results knowing that ep5->ep50 Spearman
# on this corpus is +0.263, not the 0.90 assumed when that budget was chosen: rank order at
# 5 epochs largely does not survive to 50. It is a screen for "not broken", not for "best".
#
# git push/commit stays USER-MANUAL.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
source "$D/seeds_distinct.sh"
MAC_CAP=9053

POP="${POP:-50}"
GENS="${GENS:-30}"
EP="${EP:-5}"

echo "=== GA distinct10 v2  pop=$POP gens=$GENS epoch=$EP  $(date -Is) on $(hostname) ==="
echo "seeds: $BEST10_V2_DB"

"$PY" "$D/run_seed_ga.py" \
  --seeds $BEST10_V2_DB \
  --seed-mac-cap "$MAC_CAP" \
  --pop-size "$POP" --gens "$GENS" --epoch "$EP" \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --diversity-gate \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 12 \
  --checkpoint "$D/seed_ga_D10v2_checkpoint.pkl" \
  --cache      "$D/seed_ga_D10v2_cache.json" \
  --models-dir "$D/seed_ga_D10v2_models" \
  --results    "$D/seed_ga_D10v2_results.json" \
  --save-to-db

#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# NORM-DIVERSITY GA. Same config as launch_distinct10_v2.sh except the seed pool and
# BRUTE_NORM_SAFE, so any change in the norm distribution is attributable to those two.
#
# THE PROBLEM THIS TESTS. d10v2 produced 88 norm-free of 89 generated sources. The cause was
# not a missing operator -- `layer_type` exists and ran at weight 0.15 -- but that it can only
# SWAP a norm that is already present, and 9 of its 10 seeds had no norm module at all. It
# also had only edge-hostile targets (BatchNorm2d -> GroupNorm/LayerNorm/InstanceNorm2d, all
# CPU-fallback on the TFLite GPU delegate), so on the one seed where it could fire it made
# things worse.
#
# TWO CHANGES:
#   1. Seeds now span five norm regimes instead of one, so `layer_type` has something to act
#      on: NORM-FREE 6, BatchNorm 2, GroupNorm 1, LayerNorm 1, InstanceNorm 1.
#   2. BRUTE_NORM_SAFE=1 restricts norm targets to {BatchNorm2d, Identity} -- the mobile-safe
#      set -- and config.py now carries Identity as both source and target so the axis can
#      move in BOTH directions (previously it was a one-way ratchet toward norm-free).
#
# NOT using --edge-strict on purpose: it would reject children carrying the SOFT norms, i.e.
# most descendants of the LayerNorm/GroupNorm/InstanceNorm seeds, and we want to OBSERVE what
# the norm distribution does rather than force it. Deployability is labelled afterwards.
#
# FRESH checkpoint/cache/models paths -- reusing d10v2's would resume that population at
# generation 8 with the old seeds and prove nothing.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.25,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_NORM_SAFE=1
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NN=/a/mm/nn-dataset/ab/nn/nn
MAC_CAP=9053
POP="${POP:-50}"; GENS="${GENS:-30}"; EP="${EP:-5}"

# in-DB seeds (resolved by name) -- norm-free top performers + the one GroupNorm with stats
SEEDS_DB="DenoiseSubPixel DenoiseLPPool DenoiseELUCtx DenoiseThunder DenoiseSelfAttn DenoiseGroupNorm"
# source seeds: never trained under our recipe, so not resolvable by name.
# DenoiseUNet2 is the BEST model measured (35.368 last10) and norm-free; DenoiseUNet is the
# same lineage with BatchNorm; ResBN/LayerNorm/InstNorm supply the remaining norm regimes.
SEEDS_FILES="$NN/DenoiseUNet2.py $NN/DenoiseUNet.py $NN/DenoiseResBN.py $NN/DenoiseLayerNorm.py $NN/DenoiseInstNorm.py"

echo "=== GA NORM-v3 pop=$POP gens=$GENS ep=$EP NORM_SAFE=$BRUTE_NORM_SAFE $(date -Is) ==="
echo "db seeds   : $SEEDS_DB"
echo "file seeds : $SEEDS_FILES"

python -u "$D/run_seed_ga.py" \
  --seeds $SEEDS_DB \
  --seed-files $SEEDS_FILES \
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
  --checkpoint "$D/seed_ga_NORMv3_checkpoint.pkl" \
  --cache      "$D/seed_ga_NORMv3_cache.json" \
  --models-dir "$D/seed_ga_NORMv3_models" \
  --results    "$D/seed_ga_NORMv3_results.json" \
  --save-to-db

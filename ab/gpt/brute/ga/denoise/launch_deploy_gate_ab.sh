#!/usr/bin/env bash
# A/B: does the SOURCE-ONLY deployability gate produce a more deployable corpus?
#
# THE CLAIM UNDER TEST. deploy_gate.py predicts full CPU eviction from the AST at AUC 0.986
# out-of-fold (0.921 on a held-out generator). That makes it a validated PREDICTOR. It does not
# yet make it a METHOD: nobody has shown that gating a search on it yields a corpus that is
# measurably more deployable at the same quality. This run is that experiment.
#
# DESIGN. Two arms, identical in every respect -- same seeds, population, generations, operators,
# crossover, diversity gate, MAC cap, epochs, random seed -- differing ONLY in whether the
# export gate is active:
#   ctrl   no deployability gate at all
#   pred   --deploy-predict-thr 0.5   (the measured operating point)
# Both arms SCORE and KEEP every candidate in the search, so the explorable space is identical
# and the arms cannot diverge through selection pressure. The gate decides corpus admission
# only. That is what makes the comparison attributable to the gate rather than to a different
# search trajectory.
#
# WHAT IS MEASURED AFTERWARDS. Both arms' exported models go to AI Hub; the comparison is
# eviction rate and the PSNR distribution. The gate wins only if eviction falls WITHOUT quality
# falling -- a gate that improves deployability by discarding good models is not useful.
#
# NOT the existing --deploy-gate, which pays a ~3.4 s TFLite conversion per candidate and gates
# on graph size (43% recall). Running that as a third arm would be a fairer cost comparison and
# is deliberately out of scope here: it would triple the runtime of the experiment.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
ARM="${ARM:?set ARM=ctrl or ARM=pred}"
MAC_CAP=9053

case "$ARM" in
  ctrl) GATE=() ;;
  pred) GATE=(--deploy-predict-thr 0.5) ;;
  *) echo "ARM must be ctrl or pred"; exit 2 ;;
esac

python3 "$D/run_seed_ga.py" \
  --seeds DenoiseUnet1F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
          DenoiseUnet3F \
  --pop-size 40 --gens 12 --epoch 5 \
  --elitism 4 --tournament 3 \
  --crossover block \
  --skip-ops --skip-prob 0.3 \
  --diversity-gate --cka-max 0.80 --diversity-n 16 \
  --edge-filter --mac-cap "$MAC_CAP" \
  --cost-metric macs --cost-mode log --cost-weight 0.3 --cost-budget 0 \
  --num-workers 8 \
  "${GATE[@]}" \
  --checkpoint "/a/mm/sweep_out/dgate_${ARM}/checkpoint.pkl" \
  --cache      "/a/mm/sweep_out/dgate_${ARM}/cache.json" \
  --models-dir "/a/mm/sweep_out/dgate_${ARM}/models" \
  --results    "/a/mm/sweep_out/dgate_${ARM}/results.json" \
  --save-to-db

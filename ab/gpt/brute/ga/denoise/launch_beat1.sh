#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# BEAT-1 — beat-SplitterNet search on GPU1 (GPU-B), MAI dataset (head-start; same ai-benchmark
# ecosystem as the paper's MAI2021 baseline row). Uses the FAMILY-AUDITED lean seed pool:
#   - 3 NEW efficiency-frontier families: SplitterNet, MFDNet, Thunder (the point of this run)
#   - one representative per distinct existing family: 9F, Lite, MBConv, SelfAttn, ChanShuffle,
#     SubPixel, DilatedFlat, LapPyr
#   - DROPPED: over-cap U-Nets (7F/10F/4F/3F), the redundant Lite/norm near-clones, and
#     NAFNet/Restormer (torch.fx trace bug — re-add after fix).
# Pareto edge objective (PSNR - 0.3*log MACs), MAC cap 9053, diversity gate on. Provable MIDD
# head-to-head runs later once MIDD is downloaded + a --dataset flag is wired.
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
export BRUTE_DEBUG=0
export CUDA_VISIBLE_DEVICES=1          # reserve GPU-B
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NS="$D/new_seeds"
PY=python
MAC_CAP=9053
"$PY" "$D/run_seed_ga.py" \
  --seeds DenoiseUnet9F LiteDenoiseNet \
  --seed-files "$NS"/DenoiseSplitterNet.py "$NS"/DenoiseMFDNet.py "$NS"/DenoiseThunder.py \
               "$NS"/DenoiseMBConv.py "$NS"/DenoiseSelfAttn.py "$NS"/DenoiseChanShuffle.py \
               "$NS"/DenoiseSubPixel.py "$NS"/DenoiseDilatedFlat.py "$NS"/DenoiseLapPyr.py \
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
  --checkpoint "$D/seed_ga_BEAT1_checkpoint.pkl" \
  --cache      "$D/seed_ga_BEAT1_cache.json" \
  --models-dir "$D/seed_ga_BEAT1_models" \
  --results    "$D/seed_ga_BEAT1_results.json" \
  --save-to-db

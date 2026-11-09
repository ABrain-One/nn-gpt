#!/usr/bin/env bash
# Run-3 promote chain: full-train 18 selected children @50ep (resumable by checksum
# cache in fulltrain_results.json), then promote.py (Pareto + graded dedup vs the 19-seed
# pool) emitting the grown pool to promoted_pool_r3/.
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/fulltrain_topk.py" --arms R3 --k 0 --epoch 50 --num-workers 8 \
  --include-checksums ee08e4973d68754a 42056d8b19c06b7c 73506d1871e8f17d 2601f64bc630d6f7 d3c30e8cb532e4fe 4d4bbf3bebbc87d4 043a30e4f02c083c 8901067b4e1ec298 beae6349fce5da0c 3fb2f2e558e6ef1e e11d622b58e7cbe7 240d63d38230b41c f87d377e7306380b dad0218709203ada 734b25f10d2791f2 520ffea350dcde86 7d560825360773a1 a4e38d1fac5ee945
rc=$?
echo "=== FULLTRAIN EXIT $rc ==="
"$PY" "$D/promote.py" \
  --results "$D/fulltrain_results.json" \
  --models-dirs "$D/seed_ga_R3_models" \
  --pool DenoiseUnet3F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet \
         "$D"/new_seeds/DenoiseChanShuffle.py "$D"/new_seeds/DenoiseDilatedFlat.py \
         "$D"/new_seeds/DenoiseELUCtx.py "$D"/new_seeds/DenoiseInstNorm.py \
         "$D"/new_seeds/DenoiseLapPyr.py "$D"/new_seeds/DenoiseLinearMix.py \
         "$D"/new_seeds/DenoiseLPPool.py "$D"/new_seeds/DenoiseMBConv.py \
         "$D"/new_seeds/DenoiseParallelMR.py "$D"/new_seeds/DenoisePoolDeconv.py \
         "$D"/new_seeds/DenoiseReflectPad.py "$D"/new_seeds/DenoiseSelfAttn.py \
         "$D"/new_seeds/DenoiseStatNorm.py "$D"/new_seeds/DenoiseSubPixel.py \
  --pareto --bar 33.0 --strict-mac --mac-cap 9053 \
  --graded --graded-max 0.95 \
  --emit-dir "$D/promoted_pool_r3" --out "$D/promote_r3.json"
echo "=== PROMOTE CHAIN DONE ==="

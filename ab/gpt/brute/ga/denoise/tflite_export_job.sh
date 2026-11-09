#!/usr/bin/env bash
# Export a MAC-SPANNING set of denoisers to TFLite for on-device testing.
#
# THREE models, not one, and that is the point. "Is 9053 MMac a problem on mobile?" cannot be
# answered by a single measurement -- you need the SCALING CURVE. These span 7x:
#   DenoiseThunder   1274 MMac  -- inside the published mobile envelope (19.4 kMAC/px)
#   DenoiseSubPixel  4247 MMac  -- our best model by PSNR (35.246 dB), 2.7x over that envelope
#   DenoiseUnet9F    9053 MMac  -- EXACTLY the cap under test, 5.8x over
# If latency is roughly linear in MACs across these, 9F's cost is predictable and the cap is
# simply too high; if it blows up superlinearly, memory traffic dominates and the MAC proxy
# is the wrong currency altogether. Either result is useful.
#
# WEIGHTS: only DenoiseUnet9F has a trained checkpoint (out/ckpt/.../best_model.pth). The
# other two export at random init. That is fine for LATENCY -- inference time does not depend
# on weight values -- but their OUTPUT will be noise, so do not judge image quality from them.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
OUT=/a/mm/sweep_out/tflite_export
ROOT=/scratch/root
mkdir -p "$OUT" "$ROOT/out"
for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
cd "$ROOT" || exit 1
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
export PYTHONPATH="$ROOT:/a/mm/nn-gpt"

for HW in 256 512; do
  echo
  echo "######## exporting at ${HW}x${HW} $(date -Is) ########"
  python -u "$GA/to_tflite.py" \
      --seeds DenoiseThunder DenoiseSubPixel DenoiseUnet9F \
      --ckpt-dir "$SRC/out/ckpt" \
      --h "$HW" --w "$HW" --warmup 5 --iters 30 \
      --out-dir "$OUT/${HW}" \
      --results "$OUT/report_${HW}.json"
done
echo
echo "=== artefacts ==="
find "$OUT" -name '*.tflite' -printf '%p  %kKB\n' 2>/dev/null | sort

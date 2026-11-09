#!/usr/bin/env bash
# Mobile export in TWO INDEPENDENT PASSES WITH DISJOINT DEPENDENCY STACKS.
#
# Pass 1  torch -> ONNX     container torch 2.7.1, clean PYTHONPATH, no pip install.
# Pass 2  ONNX  -> TFLite   tensorflow + onnx2tf in their own --target dir. NO TORCH.
#
# WHY NOT ai-edge-torch (the obvious answer). It pins torch 2.12.1. Installing it beside the
# container's 2.7.1 loaded two torches into one process and died on
#   KeyError: 'getpwuid(): uid not found: 1067'
#   AssertionError: Artifact of type=precompile already registered
# taking pass 1 down with it. The follow-up fix -- rm -rf the pinned torch stack out of the
# target dir -- then removed litert_torch's own dependencies (typing_extensions, sympy,
# networkx, jinja2, filelock, fsspec) and left it unimportable, which is what the last run
# reported. Both failures have ONE cause: the converter and the exporter sharing a process.
# Converting from the ONNX FILE removes the shared process, so there is no version to
# negotiate and nothing to delete afterwards.
#
# The old target dir is not reused -- it is the damaged one.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
PIPDIR=/a/mm/pipx/o2tf
OUT=/a/mm/sweep_out/mobile_export
mkdir -p "$OUT" /scratch/o2tf
# torch and pip both resolve HOME for caches; uid 1067 has no passwd entry in this image.
export HOME=/tmp XDG_CACHE_HOME=/tmp/.cache
mkdir -p /tmp/.cache

echo "######## PASS 1: torch -> ONNX (container torch, clean path) ########"
( cd "$SRC" && env -u PYTHONPATH PYTHONPATH="$SRC:/a/mm/nn-gpt" \
    python -u "$GA/export_mobile.py" --out "$OUT" --ckpt-dir "$SRC/out/ckpt" --sizes 256 512 )
echo "PASS 1 rc=$?"

echo
echo "######## PASS 2: ONNX -> TFLite (tensorflow + onnx2tf, no torch) ########"
# tf_keras is checked separately: onnx2tf's "tf_converter" backend -- the one that emits a
# SavedModel, and the only route to a working DenoiseUnet9F -- raises
# OptionalTensorFlowDependencyError without it, and silently falls back to the direct
# FlatBuffer builder that produced the broken file. tensorflow alone is not enough.
if [ ! -d "$PIPDIR/onnx2tf" ] || [ ! -d "$PIPDIR/tf_keras" ]; then
  echo "installing converter stack into $PIPDIR (fresh; the damaged aiedge dir is not reused)"
  timeout 2400 pip install --no-cache-dir --target="$PIPDIR" \
      onnx2tf tensorflow tf_keras onnx onnxruntime onnx-graphsurgeon sng4onnx \
      simple_onnx_processing_tools psutil ai_edge_litert 2>&1 | tail -6
fi
if [ ! -d "$PIPDIR/onnx2tf" ]; then
  echo "converter stack unavailable; ONNX artefacts from pass 1 stand on their own"
else
  # cwd is /tmp, not $SRC: nothing of ours should be importable in this pass. PYTHONPATH is
  # REPLACED, not extended -- if the container's torch were reachable and something pulled it
  # in, we would be back in the two-torch failure this split exists to prevent.
  ( cd /tmp && env -u PYTHONPATH PYTHONPATH="$PIPDIR" \
      python -u "$GA/onnx_to_tflite.py" --in-dir "$OUT" --work-dir /scratch/o2tf )
  echo "PASS 2 rc=$?"
fi

echo
echo "=== artefacts ==="
ls -la "$OUT" | awk 'NR>1 {printf "  %-42s %8.1f KB\n", $9, $5/1024}'

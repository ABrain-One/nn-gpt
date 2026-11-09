#!/usr/bin/env bash
# Re-export the selected models to TFLite at the two PUBLISHED runtime resolutions.
#
# WHY TWO RESOLUTIONS. The two reference papers benchmark at different sizes and there is no way to
# convert between them -- NPU latency is not linear in pixel count, so a measured 720x480 number
# cannot be scaled to Full HD or back:
#     720 x 480    MIDD (CVPR 2024) Table 2 and MAI 2021    -> scored with C = 2^64
#    1088 x 1920   MAI 2026 official runtime protocol        -> scored with C ~= 7.56e18
# Exporting both costs one extra conversion per model and removes the need to caveat either table.
#
# NO GPU, AND THAT IS THE POINT. fp32 TFLite latency is a property of the graph, not of the trained
# weights, so re-exporting needs neither checkpoints nor a GPU -- which matters because the cluster
# is at 16/16 GPUs (abl-train 8, ft50 5, confirm-train 2, another session 1). This job requests
# nvidia.com/gpu: 0 and therefore consumes no GPU quota and starts immediately. CUDA_VISIBLE_DEVICES
# is still exported empty: `gpus: 0` is a SCHEDULING statement, not isolation -- a probe on this
# cluster saw all four cards from inside a gpus:0 pod -- and mutator/lemur_io.py:169 sets
# device=cuda unconditionally when torch.cuda.is_available(). Emptying the variable is the only
# thing that actually keeps this off a card another job is using.
#
# SOURCES ARE THE STAGED COPIES from codes.jsonl, i.e. the same text the 256x256 artifacts were
# built from, so the 720x480 and Full HD graphs are the same graphs that produced the 256x256
# latencies. Re-deriving them from the DB would risk profiling a different model than the one the
# earlier table describes.
set -uo pipefail

export CUDA_VISIBLE_DEVICES=""
# PY is the interpreter that actually has a converter. The ai-linux image ships neither
# litert_torch nor ai_edge_torch, so the caller builds a venv (see k8s/reexport-cpu.yaml, copied
# from the lp-tfpool-*.json bootstrap that produced the 2,511-model pool) and passes its python in.
# Defaulting to python3 keeps the script runnable anywhere the packages happen to be present.
PY="${PY:-python3}"
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
RX="${RX:-/a/mm/sweep_out/reexport}"
SRC="$RX/src"
LOG="$RX/export.log"
mkdir -p "$RX"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== TFLITE RE-EXPORT at published resolutions ==="
say "sources: $(ls "$SRC"/*.py 2>/dev/null | wc -l) staged .py"
[ -s "$RX/subset.json" ] || { say "FATAL no subset.json -- run reexport_select.py first"; exit 1; }
"$PY" -c "import torch" 2>/dev/null || say "WARNING torch/onnx import check failed"

rc=0
for RES in "480 720" "1088 1920"; do
  set -- $RES
  H=$1; W=$2
  OUT="$RX/tflite_${H}x${W}"
  mkdir -p "$OUT"
  say "--- converting at ${H}x${W} -> $OUT"
  # reexport_convert.py, NOT to_tflite.py: the latter imports litert_torch unconditionally, which
  # is absent from this image, and failed all 16 models before tracing a single graph.
  # reexport_convert.py keeps tflite_pool_convert.py's proven ai_edge_torch fallback (that script
  # built the 2,511-model pool in this same image) and adds separate h/w, which the square-only
  # `--size` could not express.
  "$PY" "$GA/reexport_convert.py" --src "$SRC" --out "$OUT" --h "$H" --w "$W" \
      --report "$RX/report_${H}x${W}.json" >>"$LOG" 2>&1 || rc=1
  n=$(ls "$OUT"/*.tflite 2>/dev/null | wc -l)
  say "    ${H}x${W}: $n artifacts"
  # FAIL LOUD ON A SILENT ZERO. An earlier sweep in this project "succeeded" while converting
  # nothing, because the loop body failed and the last command in the script still exited 0.
  [ "$n" -gt 0 ] || { say "FATAL zero artifacts at ${H}x${W}"; exit 1; }
done

say "--- per-resolution summary ---"
"$PY" - "$RX" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
rx = sys.argv[1]
for res in ("480x720", "1088x1920"):
    f = os.path.join(rx, f"report_{res}.json")
    if not os.path.exists(f):
        print(f"{res}: no report"); continue
    try:
        rows = json.load(open(f))
    except Exception as e:
        print(f"{res}: unreadable report ({e})"); continue
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    kb = [r["size_kb"] for r in ok if r.get("size_kb")]
    print(f"{res}: {len(ok)} ok, {len(bad)} failed"
          + (f" | {min(kb)}-{max(kb)} KB" if kb else ""))
    for r in bad[:5]:
        print(f"   FAIL {r.get('model','?')[:44]}: {str(r.get('error'))[:90]}")
PY

say "=== export done (rc=$rc) ==="
say "next: reexport_submit.py profiles both resolutions on the AI Hub farm (login node, needs the API token)"
exit $rc

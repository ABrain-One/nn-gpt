#!/usr/bin/env bash
# NPU deployability gate as a pipeline stage: candidate sources in, deployable subset out.
#
# WHERE THIS GOES IN THE FUNNEL. Immediately after preflight and BEFORE any GPU is spent:
#
#   generate -> preflight (valid, novel, MMac<=9053) -> [ THIS GATE ] -> 1-epoch screen -> 50-epoch train
#
# Preflight already establishes that a candidate builds, runs a forward pass, is architecturally
# novel and fits the MAC budget. What it cannot establish is whether the thing will run on a phone:
# 33 of 128 profiled models (26%) exceed 10 ms at 256x256 against a 1.72 ms median, and every one of
# them passes the MAC budget. Placing the gate before the 1-epoch screen means zero GPU is spent on
# a candidate that can never ship.
#
# WHY IT RUNS ON THE LOGIN NODE. Two capabilities the pods do not have: the TFLite converter (the
# ai-linux image ships neither litert_torch nor ai_edge_torch, hence the shared venv at
# thesis/venv_tflite) and the AI Hub API token (which lives in the user's config; cluster pods have
# no network credentials). Both halves are CPU-only and cost no GPU quota, so running them outside
# the cluster costs nothing.
#
# COST. Conversion is ~3.4 s per candidate; profile jobs run in parallel on the farm. 250 candidates
# is roughly 15 minutes wall clock and zero GPU, against ~50 GPU-minutes per undeployable candidate
# that would otherwise be trained.
#
# USAGE
#   npu_gate_stage.sh <src_dir_of_candidate_py> [work_dir] [max_ms]
# writes  <work>/verdict.json  with a "pass" list the caller should filter its candidates by.
set -uo pipefail

SRC="${1:?usage: npu_gate_stage.sh SRC_DIR [WORK_DIR] [MAX_MS]}"
WORK="${2:-/shared/ssd/home/b-b-ahmed/thesis/sweep_out/npu_gate}"
MAX_MS="${3:-10}"
GA=/shared/ssd/home/b-b-ahmed/thesis/nn-gpt/ab/gpt/brute/ga/denoise
VENV=/shared/ssd/home/b-b-ahmed/thesis/venv_tflite/bin/python
LOG="$WORK/stage.log"
mkdir -p "$WORK"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

n_src=$(ls "$SRC"/*.py 2>/dev/null | wc -l)
say "=== NPU GATE: $n_src candidates from $SRC (threshold ${MAX_MS} ms) ==="
[ "$n_src" -gt 0 ] || { say "FATAL no *.py in $SRC"; exit 1; }
[ -x "$VENV" ] || { say "FATAL no converter venv at $VENV"; exit 1; }
"$VENV" -c "import litert_torch" 2>/dev/null || { say "FATAL venv has no litert_torch"; exit 1; }

say "--- 1/3 convert to TFLite @ 256x256 (CPU, no GPU) ---"
python3 "$GA/npu_gate.py" --convert --src "$SRC" --work "$WORK" >>"$LOG" 2>&1
n_tfl=$(ls "$WORK"/tflite/*.tflite 2>/dev/null | wc -l)
say "    converted $n_tfl/$n_src"
# Zero conversions means the toolchain is broken, not that every candidate is bad. Those are very
# different situations and only the first should stop the pipeline.
[ "$n_tfl" -gt 0 ] || { say "FATAL nothing converted -- toolchain problem, not a verdict"; exit 1; }

say "--- 2/3 profile on the AI Hub farm (needs the API token) ---"
for i in $(seq 1 90); do   # 90 min: the farm ran ~1 h deep behind a 400-job sweep
  python3 "$GA/npu_gate.py" --profile --work "$WORK" >>"$LOG" 2>&1
  rc=$?
  [ "$rc" -ne 2 ] && { say "    profiling complete"; break; }
  say "    pass $i: jobs still pending"
  sleep 60
done

say "--- 3/3 verdict ---"
python3 "$GA/npu_gate.py" --verdict --work "$WORK" --max-ms "$MAX_MS" 2>&1 | tee -a "$LOG"

PASS=$(python3 -c "
import json;d=json.load(open('$WORK/verdict.json'));print(d['n_pass'])" 2>/dev/null || echo 0)
say "=== gate done: $PASS of $n_src candidates are deployable ==="
say "the caller should train ONLY the names in $WORK/verdict.json['pass']"
# An empty pass list is a legitimate verdict (a whole batch can be undeployable) and must not look
# like a crash, so this exits 0. Callers should read n_pass, not the exit code.
exit 0

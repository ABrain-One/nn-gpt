#!/usr/bin/env bash
# 2-GPU LLM denoise generation: index 0 = cold arm, index 1 = seeded arm.
# Split by ARM rather than sharding one arm, so each pod's output is self-contained and
# the two arms cannot contend for the same --out directory.
# Weights are already cached in /a/mm/hf_cache (14.5 GB) so there is no download.
set -uo pipefail
IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
case "$IDX" in
  0) ARM=cold;   N="${N_COLD:-150}" ;;
  1) ARM=seeded; N="${N_SEEDED:-400}" ;;
  *) echo "bad index $IDX"; exit 1 ;;
esac
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
OUTD=/a/mm/sweep_out/llm_gen2/$ARM
mkdir -p "$OUTD"
export HF_HOME=/a/mm/hf_cache HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=/a/mm/nn-gpt
cd "$GA" || exit 1
echo "=== LLM GEN2 arm=$ARM n=$N $(date -Is) on $(hostname) ==="
python -u llm_denoise.py --mode "$ARM" --n "$N" --supporting 3 \
       --llm "${LLM:-Qwen/Qwen2.5-Coder-7B-Instruct}" --out "$OUTD/run" 2>&1
echo "=== done arm=$ARM rc=$? $(date -Is) ==="

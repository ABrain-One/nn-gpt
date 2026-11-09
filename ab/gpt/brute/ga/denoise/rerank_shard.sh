#!/usr/bin/env bash
# Re-rank one shard of the corpus on the PUBLISHED protocol: full-resolution RGB PSNR over the
# 50 MAI validation frames.
#
# WHY THIS RUN EXISTS. The corpus ranks models by Y-channel PSNR on a single 256x256 centre crop.
# Measured on 21 models, that ranking is ANTI-CORRELATED with full-frame performance at the top:
# every model above 36 dB on the crop loses 3.3-4.0 dB at full resolution (DenoiseUnet5 -3.97,
# LiteDenoiseNet -3.26, the LLM 50-epoch champions ~-3.6), while every model below 34.8 GAINS
# 2.0-3.6 dB. The corpus's nominal best edge-legal model, DenoiseUnet5 at 37.436, lands at 33.469
# -- third-worst of everything measured. So the internal ranking is not merely noisy, it is
# inverted where it matters, and the models that actually top the published protocol sit mid-table
# on the crop. This run replaces the ranking rather than sampling it.
#
# It also settles a statistical weakness in that finding: the top-end evidence currently rests on
# about three independent architectures, because four of the six high-crop cases are LLM champions
# sharing a lineage. n=1,582 removes that objection either way.
#
# EVALUATION ONLY -- no training, no DB writes, no repo writes. Each shard appends to its own
# results file, so shards cannot interleave, and fullres_eval.py skips models already present in
# that file, which makes every shard resumable after a preemption without redoing work.
#
# STRIDED SHARDING (models[s::8], done by the manifest builder) rather than contiguous blocks:
# the model list is sorted by name, so contiguous blocks would put all of one family in one shard
# and make its runtime unpredictable. Striding gives every shard the same family mix.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
RX=/a/mm/sweep_out/rerank
# EXPLICIT SHARD WINS OVER THE COMPLETION INDEX, and getting this backwards cost seven shards of
# GPU time. rerank_launch.sh launches eight SEPARATE single-completion jobs (so each can slot into
# whatever memory the namespace frees), passing the shard number as SHARD=n. But mkjob.py emits an
# Indexed job, so every one of those pods also gets JOB_COMPLETION_INDEX=0. With the index taking
# precedence, all eight pods read subset_shard0.json and appended to fullres_shard0.jsonl: 710 rows
# covering 198 models, and nothing at all from shards 1-7. Prefer SHARD; fall back to the index so
# a single `--completions 8` indexed job still works if we ever launch it that way.
SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
SUB="$RX/subset_shard${SHARD}.json"
OUT="$RX/fullres_shard${SHARD}.jsonl"

echo "=== rerank shard $SHARD | $(date -Is) ==="
[ -s "$SUB" ] || { echo "FATAL: $SUB missing"; exit 1; }
python3 -c "import json;print('  models in shard:',len(json.load(open('$SUB'))['models']))"
[ -f "$OUT" ] && echo "  resuming: $(wc -l < "$OUT") already done"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1

# The validation frames are the published set; fullres_eval.py reads them from --val and pairs
# them positionally after asserting equal counts, so a mismatched directory fails loudly.
cd /a/mm/nn-dataset || exit 1
PYTHONPATH=/a/mm/nn-dataset python -u "$GA/fullres_eval.py" \
    --subset "$SUB" --val /a/mm/nn-dataset/validation --out "$OUT"
rc=$?
echo "=== shard $SHARD done rc=$rc  rows=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )  $(date -Is) ==="
exit $rc

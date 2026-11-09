#!/usr/bin/env bash
# Full-resolution RGB eval for the 1,341 models added by the four dnseed2 GA arms.
#
# WHY THESE NEED IT. They were selected by the GA on 256x256 Y-channel crop PSNR, and that metric
# carries essentially no rank information at the top of the range: Spearman -0.025 among the top
# 100 of the 1,578 models already re-ranked. So the arms' own ordering says little about which of
# them is actually good on the published protocol. This measures all of them on it.
#
# SHARD SELECTION MIRRORS rerank_shard.sh, INCLUDING THE BUG IT COST US. An explicit SHARD must
# win over JOB_COMPLETION_INDEX: mkjob.py emits an Indexed Job, so with the index taking priority
# every pod read shard 0 and eight shards of GPU time produced one shard of results.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
RX=/a/mm/sweep_out/rerank2
SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
SUB="$RX/subset_shard${SHARD}.json"
OUT="$RX/fullres_shard${SHARD}.jsonl"

echo "=== rerank2 shard $SHARD | $(date -Is) ==="
[ -s "$SUB" ] || { echo "FATAL $SUB missing"; exit 1; }
python3 -c "import json;print('  models in shard:',len(json.load(open('$SUB'))['models']))"
[ -f "$OUT" ] && echo "  resuming: $(wc -l < "$OUT") already done"
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1

cd /a/mm/nn-dataset || exit 1
PYTHONPATH=/a/mm/nn-dataset python -u "$GA/fullres_eval.py" \
    --subset "$SUB" --val /a/mm/nn-dataset/validation --out "$OUT"
rc=$?
echo "=== shard $SHARD done rc=$rc rows=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 ) $(date -Is) ==="
exit $rc

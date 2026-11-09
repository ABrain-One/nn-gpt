#!/usr/bin/env bash
# Launch the full-res re-rank shards AS CAPACITY FREES, rather than all at once.
#
# WHY A WAITER AND NOT `kubectl apply` x8. The namespace is at 558 of 560 Gi of limits.memory --
# three GA arms of ours plus ~17 midd-* pods from another session. Applying eight jobs now would
# create eight Jobs whose pods the controller cannot create; they would sit emitting FailedCreate
# forever and, worse, they would grab quota the moment it appeared, ahead of the other session's
# work and ahead of our own GA arms if one restarts. So this polls and launches one shard at a
# time, only when there is genuinely room for it.
#
# GPU IS NOT THE BINDING CONSTRAINT AND THAT IS WORTH KNOWING: quota shows 12 of 16 GPUs used but
# only 2 Gi of 560 Gi memory. Every scheduling failure in this session has been memory, so that is
# what this checks.
#
# FOOTPRINT PER SHARD IS DELIBERATELY SMALL. Evaluation holds one 3200x2432 frame plus the model;
# 16 Gi is ample and lets a shard slot into a gap that a 48 Gi training pod could not. Eight of
# these are 128 Gi total, versus 384 Gi if they were sized like the GA arms.
#
#   bash rerank_launch.sh              # default: up to 8 shards, poll every 5 min
#   SHARDS=4 POLL=120 bash rerank_launch.sh
set -uo pipefail
GA_H=/shared/ssd/home/b-b-ahmed/thesis/nn-gpt/ab/gpt/brute/ga/denoise
RX_H=/shared/ssd/home/b-b-ahmed/thesis/sweep_out/rerank
SHARDS="${SHARDS:-8}"
POLL="${POLL:-300}"
MEM_GI="${MEM_GI:-16}"          # per shard
HEADROOM_GI="${HEADROOM_GI:-8}" # leave this much free so we never take the last slice
LOG="${LOG:-$RX_H/launch.log}"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_gi() {
  # "limits.memory  558Gi  560Gi" -> free integer Gi. Falls back to 0 on any parse surprise,
  # which makes the loop wait rather than launch blind.
  kubectl describe quota 2>/dev/null | awk '
    /limits\.memory/ { gsub(/Gi/,"",$2); gsub(/Gi/,"",$3); if ($3 > $2) print $3-$2; else print 0; exit }' \
    | head -1 | grep -E '^[0-9]+$' || echo 0
}

say "rerank launcher up: $SHARDS shards, ${MEM_GI}Gi each, poll ${POLL}s, headroom ${HEADROOM_GI}Gi"
launched=0
while [ "$launched" -lt "$SHARDS" ]; do
  name="rerank-$launched"
  # Already running or finished from an earlier invocation? Skip rather than duplicate.
  if kubectl get job "$name" >/dev/null 2>&1; then
    say "  $name already exists -- skipping"
    launched=$((launched+1)); continue
  fi
  f=$(free_gi)
  need=$((MEM_GI + HEADROOM_GI))
  if [ "$f" -lt "$need" ]; then
    say "  waiting: ${f}Gi free, need ${need}Gi for $name"
    sleep "$POLL"; continue
  fi
  y=$(mktemp /tmp/rerank-XXXXXX.json)
  python3 "$GA_H/mkjob.py" --name "$name" --completions 1 --gpus 1 --cpu 6 \
      --mem "${MEM_GI}Gi" --scratch 10Gi \
      --command "SHARD=$launched bash /a/mm/nn-gpt/ab/gpt/brute/ga/denoise/rerank_shard.sh" \
      --out "$y" >/dev/null
  if kubectl apply -f "$y" >>"$LOG" 2>&1; then
    say "  launched $name (${f}Gi was free)"
    launched=$((launched+1))
    sleep 30      # let the pod actually claim its quota before measuring again
  else
    say "  apply failed for $name; retrying after ${POLL}s"
    sleep "$POLL"
  fi
  rm -f "$y"
done
say "all $SHARDS shards launched or pre-existing"
say "when they finish, merge with:  cat $RX_H/fullres_shard*.jsonl | wc -l   then rank on psnr_rgb_fullres"

#!/usr/bin/env bash
# Trip the ablation's MIDD_DONE gate when the cross-dataset eval finishes. Login node.
#
# The gate was originally a marker the USER touches, chosen when the MIDD/SIDD run was going
# to be driven from another terminal. It is now xd-train, so the marker can be tripped by the
# thing it actually gates on. Without this, ft50 releases its 8 GPUs at ~14:00 and the
# ablation waits for a human -- burning the whole afternoon idle.
#
# ONLY TRIPS ON REAL RESULTS. A job can end with zero output (quota rejection leaves no pod;
# a crashed runner leaves an empty result file). Requiring completed runs on disk means a
# broken cross-dataset run does NOT hand the GPUs to the ablation and quietly lose the
# transfer result -- it stops and waits for a human, which is the correct failure mode.
set -uo pipefail
NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
MARK=$H/sweep_out/MIDD_DONE
LOG=$H/sweep_out/crossdata/gate.log
mkdir -p "$(dirname "$LOG")"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for xd-train (120 runs / 6 shards)"
while :; do
  ok=$(kubectl get job xd-train -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  run=$(kubectl get pods -l job-name=xd-train -n $NS --no-headers 2>/dev/null | grep -c Running)
  n=$(cat "$H/sweep_out/crossdata"/result_shard*.jsonl 2>/dev/null | wc -l)
  nok=$(cat "$H/sweep_out/crossdata"/result_shard*.jsonl 2>/dev/null | grep -c '"status": "OK"')
  if [ "${ok:-0}" -ge 6 ] 2>/dev/null; then say "xd-train COMPLETE ($nok/$n OK)"; break; fi
  if [ "${run:-0}" -eq 0 ] && [ "$n" -gt 0 ]; then
    say "xd-train has no running pods ($nok/$n OK) -- treating as finished"; break
  fi
  sleep 300
done

nok=$(cat "$H/sweep_out/crossdata"/result_shard*.jsonl 2>/dev/null | grep -c '"status": "OK"')
if [ "${nok:-0}" -lt 60 ]; then
  say "REFUSING to trip the gate: only $nok/120 runs succeeded."
  say "  The ablation stays blocked so the GPUs are still here to retry on."
  exit 1
fi
touch "$MARK"
say "tripped $MARK ($nok/120 OK) -- ablation may now start once ft50 frees 8 GPUs"

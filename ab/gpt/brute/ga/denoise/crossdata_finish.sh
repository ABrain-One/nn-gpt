#!/usr/bin/env bash
# When xd-train lands, build the full per-epoch stats and the transfer report. Login node.
#
# WHY A SEPARATE FINISHER. crossdata_train.sh is executing in six pods and must not be edited
# (bash reads a script by byte offset and re-seeks between commands, so an edit mid-run makes
# the interpreter resume mid-line). It already writes one <epoch>.json per epoch with 27
# fields -- the raw data is complete -- but it does NOT write the merged all_epochs.json or
# the max/avg_last10/std/min summary that beatbar produces. Those are all recoverable from
# the raw files afterwards, which is exactly how ft50 was handled.
#
# THE REPORT IS RUN TWICE, AND THE SECOND RUN IS THE REAL ONE. The 40-model subset spans
# 9.65..36.98 dB on purpose, and it includes architectures that trained cleanly to nothing.
# Those extreme points make rank correlation look excellent for a trivial reason: every method
# agrees that a 10 dB model is worse than a 34 dB one. The claim that matters is whether
# AWGN ranking predicts real-noise ranking AMONG MODELS THAT ACTUALLY WORK, so the second
# pass restricts to >=32 dB. If the correlation survives that and does not survive it, those
# are different papers -- report both.
set -uo pipefail

NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
GA=$H/nn-gpt/ab/gpt/brute/ga/denoise
XD=$H/sweep_out/crossdata
LOG=$XD/finish.log
mkdir -p "$XD"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for xd-train (120 runs / 6 shards)"
while :; do
  ok=$(kubectl get job xd-train -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  run=$(kubectl get pods -l job-name=xd-train -n $NS --no-headers 2>/dev/null | grep -c Running)
  n=$(cat "$XD"/result_shard*.jsonl 2>/dev/null | wc -l)
  [ "${ok:-0}" -ge 6 ] 2>/dev/null && { say "xd-train COMPLETE ($n/120)"; break; }
  if [ "${run:-0}" -eq 0 ] && [ "$n" -gt 0 ]; then
    say "xd-train has no running pods ($n/120) -- proceeding with what exists"; break
  fi
  sleep 300
done

say "--- aggregating per-epoch curves into all_epochs.json ---"
python3 "$GA/aggregate_curves.py" --dir "$XD/curves" --out "$XD/summary.json" 2>&1 | tail -4 | tee -a "$LOG"

say "--- transfer report: ALL models (includes non-functional architectures) ---"
python3 "$GA/crossdata_report.py" --dir "$XD" 2>&1 | tee -a "$LOG"

say "--- transfer report: WORKING models only (>=32 dB on the in-experiment AWGN column) ---"
python3 "$GA/crossdata_report.py" --dir "$XD" --min-dB 32.0 2>&1 | tee -a "$LOG"

say "=== cross-dataset analysis complete ==="
say "READ THE >=32 dB TABLE. The all-models correlation is inflated by architectures that"
say "trained to nothing -- agreeing that a 10 dB model is bad is not evidence of transfer."

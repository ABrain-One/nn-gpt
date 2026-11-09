#!/usr/bin/env bash
# When confirm-train lands, aggregate the per-epoch curves and run the regret analysis.
# Login node. Mirrors crossdata_finish.sh so the two runs are archived the same way.
#
# WHY A SEPARATE FINISHER. confirm_train.sh is executing in pods and must not be edited -- bash
# reads a script by byte offset and re-seeks between commands, so an edit mid-run makes the
# interpreter resume mid-line. It writes one <epoch>.json per epoch (the raw data is complete)
# but not the merged all_epochs.json, exactly as ft50 and crossdata did; that merge is
# recoverable afterwards and is done here.
#
# THE REPORT IS ALSO RUN AT PARTIAL COVERAGE, ON PURPOSE. confirm_report.py degrades to a
# coverage table when fewer than 4 models have repeats everywhere, so running it early is free
# and shows whether the run is on track rather than waiting 10 h to discover it is not.
set -uo pipefail

NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
GA=$H/nn-gpt/ab/gpt/brute/ga/denoise
CF=$H/sweep_out/confirm
LOG=$CF/finish.log
mkdir -p "$CF"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for confirm-train (72 runs / 2 shards, ~19.9 GPU-h)"
while :; do
  ok=$(kubectl get job confirm-train -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  run=$(kubectl get pods -l job-name=confirm-train -n $NS --no-headers 2>/dev/null | grep -c Running)
  n=$(cat "$CF"/result_shard*.jsonl 2>/dev/null | wc -l)
  [ "${ok:-0}" -ge 2 ] 2>/dev/null && { say "confirm-train COMPLETE ($n/72)"; break; }
  # No running pods AND some results means the job ended (succeeded, failed, or was evicted).
  # Proceeding with partial data beats blocking forever; the report states its own coverage.
  if [ "${run:-0}" -eq 0 ] && [ "$n" -gt 0 ]; then
    say "confirm-train has no running pods ($n/72) -- proceeding with what exists"; break
  fi
  sleep 300
done

say "--- aggregating per-epoch curves ---"
python3 "$GA/aggregate_curves.py" --dir "$CF/curves" --out "$CF/summary.json" 2>&1 | tail -4 | tee -a "$LOG"

say "--- regret analysis ---"
python3 "$GA/confirm_report.py" --dir "$CF" 2>&1 | tee -a "$LOG"

say "=== confirmation run complete ==="
say "READ regret@1 AGAINST regret@3. If they differ a lot, the finding is 'AWGN cannot pick"
say "THE winner but works as a shortlist filter' -- narrower than 'AWGN does not transfer',"
say "and the one the numbers actually support."

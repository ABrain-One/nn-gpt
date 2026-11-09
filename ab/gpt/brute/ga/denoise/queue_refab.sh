#!/usr/bin/env bash
# Wait for the farm submitter to finish, then launch the reference-arm A/B.
#
# The submitter is the heaviest I/O consumer on this host (~10.7% CPU, constant network + disk for
# ~1,250 uploads). Running the A/B against it is what starved the first attempt to 0.33 MB/s. Rather
# than guess at a delay, this polls for the process to exit and only then applies the jobs.
set -uo pipefail
H=/shared/ssd/home/b-b-ahmed/thesis
LOG=$H/sweep_out/refab/queue.log
mkdir -p "$(dirname "$LOG")"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for bulk_profile --submit to exit"
while :; do
  # match the script name but NOT this waiter, and not a grep of itself
  n=$(ps -eo cmd 2>/dev/null | grep 'bulk_profil[e].*--submit' | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 60
done
say "submitter finished; verifying GPU headroom on the namespace quota"
used=$(kubectl get resourcequota -n b-b-ahmed -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
hard=$(kubectl get resourcequota -n b-b-ahmed -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
say "quota ${used:-?}/${hard:-?}"
if [ "$(( ${hard:-16} - ${used:-16} ))" -lt 2 ]; then
  say "FEWER THAN 2 GPUs FREE -- not launching; rerun this script when capacity frees"
  exit 1
fi
for arm in ctl trt; do
  kubectl delete job "refab-$arm" -n b-b-ahmed --ignore-not-found >/dev/null 2>&1
  kubectl apply -f "$H/k8s/refab-$arm.yaml" 2>&1 | tee -a "$LOG"
done
say "both arms applied"

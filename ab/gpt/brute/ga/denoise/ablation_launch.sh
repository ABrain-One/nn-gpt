#!/usr/bin/env bash
# Launch the skip ablation on 8 shards as soon as 8 GPUs are free. Runs on the login node.
#
# NO LONGER GATED ON ft50. The previous launcher waited for ft50 to finish before it even
# looked at capacity, which would have held the ablation until ~Aug 7 14:00 on a cluster that
# may well have 8 idle cards this afternoon: xd-train releases 6 when it lands (~13:00) and
# the other session's two MIDD pods are expected to free up too. ft50 and the ablation are
# independent experiments and there is no scientific reason to order them, so capacity is the
# only real constraint and therefore the only gate. Worst case -- the other session keeps its
# two GPUs -- this simply waits, which is exactly what the old version did anyway.
#
# STILL GATED ON MIDD_DONE. The cross-dataset eval must land first: it is the finding nothing
# else in the pipeline produces, and touch_midd_done.sh refuses to trip the marker unless at
# least 60 of its 120 runs actually succeeded. A broken cross-dataset run therefore keeps the
# GPUs blocked for a retry rather than handing them to the ablation and quietly losing the
# transfer result.
#
# NSHARDS IS FIXED AT 8. ablation_train.sh assigns whole BASES to shards (i % NSHARDS over the
# sorted base list) and there are exactly 8 bases, so NSHARDS is part of the work partition,
# not a throughput dial. Changing it repartitions the grid and the per-shard resume sets stop
# covering what they used to -- that is how the TFLite sweep produced duplicate conversions
# and silently skipped 42 models. One base per GPU is also what keeps every within-base
# contrast on a single card; the effects here sit near a 0.14 dB noise floor, and splitting a
# base across the mixed 4090/3090 fleet would confound silicon with topology.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
K8S=$HOST/k8s
MARKER=$HOST/sweep_out/MIDD_DONE
LOG=$HOST/sweep_out/ablation/launcher.log
mkdir -p "$HOST/sweep_out/ablation"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== ablation launcher (gates: MIDD_DONE + 8 free GPUs; ft50 gate REMOVED) ==="
[ -f "$K8S/abl-train-8.yaml" ] || { say "FATAL missing $K8S/abl-train-8.yaml"; exit 1; }

n=0
while [ ! -f "$MARKER" ]; do
  n=$((n+1)); [ $((n % 12)) -eq 1 ] && say "waiting for MIDD_DONE ($((n*5)) min elapsed)"
  sleep 300
done
say "MIDD_DONE present"

# Require 8 free TWICE IN A ROW, 60 s apart. A single reading races pod teardown: xd-train's
# six pods can have released their quota while the kubelet still holds them, and a job admitted
# into that window comes back FailedCreate with NO POD -- a state byte-for-byte identical to
# "still scheduling" unless the job's events are read. That silence has already cost this
# project time twice.
stable=0
for i in $(seq 1 576); do
  used=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
  hard=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
  free=$(( ${hard:-0} - ${used:-0} ))
  if [ "${free:-0}" -ge 8 ]; then
    stable=$((stable+1))
    say "  headroom $free (used ${used:-?}/${hard:-?}) [stable $stable/2]"
    [ "$stable" -ge 2 ] && break
    sleep 60
    continue
  fi
  stable=0
  [ $((i % 10)) -eq 1 ] && say "  headroom ${free:-?} (used ${used:-?}/${hard:-?}); need 8"
  sleep 300
done
if [ "$stable" -lt 2 ]; then
  say "FATAL never saw 8 free GPUs twice in 48 h -- NOT launching"
  exit 1
fi

say "launching abl-train (79 variants -> 267 runs, 8 shards)"
kubectl delete job abl-train -n $NS --ignore-not-found >/dev/null 2>&1
sleep 5
kubectl apply -f "$K8S/abl-train-8.yaml" >>"$LOG" 2>&1

# FAIL LOUD ON A ZERO-POD LAUNCH. An earlier version printed a warning here and then ended on
# a successful `say`, so a quota-rejected ablation exited 0 and looked launched.
sleep 30
pods=$(kubectl get pods -l job-name=abl-train -n $NS --no-headers 2>/dev/null | wc -l)
if [ "$pods" -eq 0 ]; then
  say "FATAL no pods created for abl-train -- quota rejection:"
  kubectl describe job abl-train -n $NS 2>/dev/null | grep -m3 "forbidden\|FailedCreate" | tee -a "$LOG"
  exit 1
fi
say "abl-train up: $pods pods"
kubectl get pods -l job-name=abl-train -n $NS \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,S:.status.phase' --no-headers | tee -a "$LOG"
say "critical path 39 runs x ~47 min = ~30.6 h"

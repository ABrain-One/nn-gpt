#!/usr/bin/env bash
# Launch reexport-gpu when 5 GPUs are free. Login node.
#
# GATE IS CAPACITY ONLY. ft50 holds 5 cards and lands around 05:20; abl-train holds 8 until roughly
# 03:00 on Aug 8 and must not be disturbed. Waiting for 5 free therefore means "waiting for ft50",
# without hard-coding that assumption -- if capacity appears from somewhere else, this takes it.
#
# 5 FREE TWICE, 60 s APART. A single reading races pod teardown: a job admitted into the window
# where quota is released but the kubelet still holds the cards comes back FailedCreate with NO POD,
# a state indistinguishable from "still scheduling" unless the job's events are read. That silence
# has cost this project time twice already.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
K8S=$HOST/k8s
RX=$HOST/sweep_out/reexport
LOG=$RX/launcher.log
mkdir -p "$RX"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== reexport-gpu launcher (gate: 5 free GPUs, seen twice) ==="
[ -f "$K8S/reexport-gpu.yaml" ] || { say "FATAL missing $K8S/reexport-gpu.yaml"; exit 1; }
[ -s "$RX/subset.json" ] || { say "FATAL missing $RX/subset.json"; exit 1; }

stable=0
for i in $(seq 1 576); do
  used=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
  hard=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
  free=$(( ${hard:-0} - ${used:-0} ))
  if [ "${free:-0}" -ge 5 ]; then
    stable=$((stable+1))
    say "  headroom $free (used ${used:-?}/${hard:-?}) [stable $stable/2]"
    [ "$stable" -ge 2 ] && break
    sleep 60
    continue
  fi
  stable=0
  [ $((i % 12)) -eq 1 ] && say "  headroom ${free:-?} (used ${used:-?}/${hard:-?}); need 5"
  sleep 300
done
if [ "$stable" -lt 2 ]; then
  say "FATAL never saw 5 free GPUs twice in 48 h -- NOT launching"
  exit 1
fi

need=$(python3 -c "
import json,os
d=json.load(open('$RX/subset.json'))
S=['$HOST/sweep_out/reexport/ckpt','$HOST/sweep_out/ckpt','$HOST/nn-dataset/out/ckpt']
print(sum(1 for m in d['models'] if not any(os.path.exists(os.path.join(s,m['model'],'best_model.pth')) for s in S)))")
say "launching reexport-gpu: $need models to retrain, then full-res eval of all 16"

kubectl delete job reexport-gpu -n $NS --ignore-not-found >/dev/null 2>&1
sleep 5
kubectl apply -f "$K8S/reexport-gpu.yaml" >>"$LOG" 2>&1

# FAIL LOUD ON A ZERO-POD LAUNCH -- a quota-rejected job otherwise ends on a successful `say`
# and looks launched.
sleep 30
pods=$(kubectl get pods -l job-name=reexport-gpu -n $NS --no-headers 2>/dev/null | wc -l)
if [ "$pods" -eq 0 ]; then
  say "FATAL no pods created for reexport-gpu -- quota rejection:"
  kubectl describe job reexport-gpu -n $NS 2>/dev/null | grep -m3 "forbidden\|FailedCreate" | tee -a "$LOG"
  exit 1
fi
say "reexport-gpu up: $pods pods"
kubectl get pods -l job-name=reexport-gpu -n $NS \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,S:.status.phase' --no-headers | tee -a "$LOG"
say "~2 models per shard x ~50 min, then shard 0 evaluates all 16 at 3200x2432"

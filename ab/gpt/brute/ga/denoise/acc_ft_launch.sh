#!/usr/bin/env bash
# Launch the predictor fine-tune when 1 GPU frees up. Login node.
#
# GATE IS 1 FREE GPU. The quota is namespace-wide with no scopes (requests.nvidia.com/gpu 16/16),
# so the H100 counts against the same pool as the RTX fleet -- targeting dgx-h100 does not dodge it.
# Capacity appears when reexport-gpu finishes (~08:10) or when the ablation's three 24-run shards
# end (~13:15), whichever comes first; this takes whichever it is.
#
# SEEN TWICE, 60 s APART: a single reading races pod teardown, and a job admitted into that window
# comes back FailedCreate with NO POD -- indistinguishable from "still scheduling" unless the job's
# events are read.
set -uo pipefail
NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
LOG=$H/sweep_out/acc_ft/launcher.log
mkdir -p "$H/sweep_out/acc_ft"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== acc-ft launcher (gate: 1 free GPU) ==="
for f in train.jsonl val.jsonl test_rows.json; do
  [ -s "$H/sweep_out/acc_ft/$f" ] || { say "FATAL missing $f"; exit 1; }
done

# YIELD TO reexport-gpu, WHICH NEEDS FIVE. Both launchers poll the same quota, so a naive
# "grab 1 free GPU" here is a race that reexport-gpu always loses: ft50 releases exactly 5, and if
# this job takes one of them first, reexport-gpu's gate never sees 5 again and stalls until the
# ablation's short shards land hours later. Neither job would report an error -- it would just
# silently sit there, which is the failure mode this project keeps rediscovering.
# So this waits for EITHER reexport-gpu to be finished (its 5 come back) OR enough headroom that
# both fit. One GPU is all this needs; it simply must not be the one reexport-gpu was waiting for.
need_free() {
  local used hard
  used=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
  hard=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
  echo $(( ${hard:-0} - ${used:-0} ))
}
rx_done() {
  local s
  s=$(kubectl get job reexport-gpu -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  [ "${s:-0}" -ge 1 ] 2>/dev/null && return 0
  # never launched at all is also "not in our way"
  kubectl get job reexport-gpu -n $NS >/dev/null 2>&1 || return 0
  return 1
}

stable=0
for i in $(seq 1 576); do
  free=$(need_free)
  if rx_done; then
    ok=$([ "${free:-0}" -ge 1 ] && echo 1 || echo 0)
    why="reexport-gpu clear"
  else
    ok=$([ "${free:-0}" -ge 6 ] && echo 1 || echo 0)
    why="waiting for reexport-gpu's 5 + 1"
  fi
  if [ "$ok" = "1" ]; then
    stable=$((stable+1)); say "  headroom $free ($why) [stable $stable/2]"
    [ "$stable" -ge 2 ] && break
    sleep 60; continue
  fi
  stable=0
  [ $((i % 12)) -eq 1 ] && say "  headroom ${free:-?}; $why"
  sleep 300
done
[ "$stable" -ge 2 ] || { say "FATAL never saw a free GPU in 48 h"; exit 1; }

say "launching acc-ft (warm start from ABrain/Accuracy-Prediction, 892 denoising examples)"
kubectl delete job acc-ft -n $NS --ignore-not-found >/dev/null 2>&1
sleep 5
kubectl apply -f "$H/k8s/acc-ft.yaml" >>"$LOG" 2>&1
sleep 30
pods=$(kubectl get pods -l job-name=acc-ft -n $NS --no-headers 2>/dev/null | wc -l)
if [ "$pods" -eq 0 ]; then
  say "FATAL no pod created -- quota/scheduling rejection:"
  kubectl describe job acc-ft -n $NS 2>/dev/null | grep -m3 "forbidden\|FailedCreate\|nodeSelector" | tee -a "$LOG"
  exit 1
fi
say "acc-ft up: $pods pod(s)"
kubectl get pods -l job-name=acc-ft -n $NS -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,S:.status.phase' --no-headers | tee -a "$LOG"
say "it trains, then scores itself on the 100 frozen holdout models against epoch-3-alone"
say "READ ft_report.json: beats_baseline=false means DO NOT SERVE IT"

#!/usr/bin/env bash
# Launch the skip ablation on 8 GPUs once ft50 AND the MIDD/SIDD run have both released them.
# Runs on the login node. Replaces ablation_after_ft50.sh, which gated on ft50 alone.
#
# THREE GATES, ALL REQUIRED:
#   1. ft50 finished          -- it holds all 8 RTX cards.
#   2. sweep_out/MIDD_DONE    -- a marker the USER touches when the cross-dataset run is done.
#   3. 8 free GPUs in quota   -- H100 pods count against the same namespace quota, so free
#                                RTX cards are not the same thing as admissible ones.
#
# WHY A MARKER AND NOT A JOB NAME. The MIDD/SIDD run is driven from another terminal, so its
# K8s job name is not known here, and guessing a `midd*` pattern is actively dangerous: the
# midd-x20-* pods belong to a DIFFERENT session and must not be treated as ours. A marker file
# is unambiguous and survives the job being renamed or re-run.
#
#   when the MIDD/SIDD run finishes:  touch /shared/ssd/home/b-b-ahmed/thesis/sweep_out/MIDD_DONE
#
# NSHARDS STAYS AT 8. ablation_train.sh assigns whole BASES to shards (i % NSHARDS over the
# sorted base list), so NSHARDS is part of the work partition, not a throughput dial. Changing
# it later repartitions the grid and the per-shard resume sets stop covering what they used to
# -- that is exactly how the TFLite sweep produced duplicate conversions and silently skipped
# 42 models. It also matters for wall-clock: with 8 bases, 8 shards gives a max of 39 runs per
# shard (~30.6 h), while 6 shards forces two shards to carry two bases each, 75 runs (~58.8 h).
# Sequencing MIDD BEFORE the ablation rather than beside it is what buys back those 28 hours.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
K8S=$HOST/k8s
MARKER=$HOST/sweep_out/MIDD_DONE
LOG=$HOST/sweep_out/ablation/launcher.log
mkdir -p "$HOST/sweep_out/ablation"

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== ablation launcher (re-sequenced: ft50 -> MIDD/SIDD -> ablation) ==="
say "gate 1: ft50 (8 shards)"
while :; do
  ok=$(kubectl get job ft50 -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  fail=$(kubectl get job ft50 -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
  run=$(kubectl get pods -l job-name=ft50 -n $NS --no-headers 2>/dev/null | grep -c Running)
  done_n=$(cat "$HOST/sweep_out/fulltrain50"/result_shard*.jsonl 2>/dev/null | wc -l)
  [ "${ok:-0}" -ge 8 ] 2>/dev/null && { say "  ft50 COMPLETE ($done_n/300 models)"; break; }
  if [ "${run:-0}" -eq 0 ] && [ "$done_n" -gt 0 ]; then
    say "  ft50 has no running pods (succeeded=${ok:-0} failed=${fail:-0}, $done_n/300) -- proceeding"
    break
  fi
  sleep 300
done

say "gate 2: waiting for $MARKER (touch it when the MIDD/SIDD run finishes)"
n=0
while [ ! -f "$MARKER" ]; do
  n=$((n+1))
  [ $((n % 12)) -eq 1 ] && say "  still waiting for MIDD_DONE ($((n*5)) min elapsed)"
  sleep 300
done
say "  MIDD_DONE present ($(date -r "$MARKER" -Is 2>/dev/null))"

# Quota, not free silicon, is what admits a pod: a rejection produces FailedCreate and NO POD,
# a state indistinguishable from "still scheduling" unless the job's events are read.
say "gate 3: GPU headroom"
for i in $(seq 1 288); do
  used=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
  hard=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
  free=$(( ${hard:-0} - ${used:-0} ))
  [ "$free" -ge 8 ] && { say "  headroom $free (used ${used:-?}/${hard:-?}) -- proceeding"; break; }
  [ $((i % 12)) -eq 1 ] && say "  headroom $free (used ${used:-?}/${hard:-?}); need 8"
  sleep 300
done
if [ "${free:-0}" -lt 8 ]; then
  say "FATAL never saw 8 free GPUs in 24 h -- NOT launching (a quota-rejected job is invisible)"
  exit 1
fi

say "launching abl-train (8 shards, 99 variants x 3 repeats = 297 runs)"
kubectl delete job abl-train -n $NS --ignore-not-found >/dev/null 2>&1; sleep 5
cat > "$K8S/abl-train-8.yaml" <<'EOF'
apiVersion: batch/v1
kind: Job
metadata: {name: abl-train, namespace: b-b-ahmed}
spec:
  backoffLimit: 0
  completions: 8
  parallelism: 8
  completionMode: Indexed
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: ml
        image: abrainone/ai-linux:cu12.6.3-latest
        command: ["/bin/bash","-lc"]
        args:
        - "NSHARDS=8 EPOCHS=50 REPEATS=3 bash /a/mm/nn-gpt/ab/gpt/brute/ga/denoise/ablation_train.sh"
        resources: {limits: {cpu: 16, memory: 8Gi, nvidia.com/gpu: 1}, requests: {cpu: 8, memory: 8Gi, nvidia.com/gpu: 1}}
        securityContext: {runAsUser: 1067, runAsGroup: 1376, allowPrivilegeEscalation: false}
        volumeMounts:
        - {mountPath: /a/mm, name: v0}
        - {mountPath: /scratch, name: scratch}
        - {mountPath: /dev/shm, name: shm}
      volumes:
      - {name: v0, hostPath: {path: /shared/ssd/home/b-b-ahmed/thesis, type: Directory}}
      - {name: scratch, emptyDir: {sizeLimit: 20Gi}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 4Gi}}
EOF
kubectl apply -f "$K8S/abl-train-8.yaml" >>"$LOG" 2>&1

# FAIL LOUD ON A ZERO-POD LAUNCH. The previous version printed a WARN here and then ended on a
# successful `say`, so a quota-rejected ablation exited 0 and looked launched.
sleep 30
pods=$(kubectl get pods -l job-name=abl-train -n $NS --no-headers 2>/dev/null | wc -l)
if [ "$pods" -eq 0 ]; then
  say "FATAL no pods created for abl-train -- quota rejection:"
  kubectl describe job abl-train -n $NS 2>/dev/null | grep -m3 "forbidden\|FailedCreate" | tee -a "$LOG"
  exit 1
fi
say "abl-train up: $pods pods"
kubectl get pods -l job-name=abl-train -n $NS \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase' --no-headers | tee -a "$LOG"
say "ablation launched; max shard 39 runs x ~47 min = ~30.6 h"

#!/usr/bin/env bash
# Launch the skip ablation on 8 GPUs as soon as ft50 releases them. Runs on the login node.
#
# WAITS RATHER THAN QUEUES. Submitting now gets the job rejected by quota admission with
# FailedCreate and NO POD -- a state that looks identical to "waiting" unless you read the
# job's events (it silently held the LLM generation for 2.5 minutes earlier). So the launch
# is gated on ft50 actually finishing and on the quota showing real headroom.
#
# SHARDING IS FIXED AT 8 AND MUST NOT CHANGE. ablation_train.sh assigns whole BASES to
# shards (i % NSHARDS over the sorted base list); changing NSHARDS later repartitions the
# work, so the per-shard resume sets stop covering what they used to and runs get repeated.
# That is exactly how the TFLite sweep ended up with duplicate conversions and 42 models
# silently never attempted.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
K8S=$HOST/k8s
LOG=$HOST/sweep_out/ablation/launcher.log
mkdir -p "$HOST/sweep_out/ablation"

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for ft50 to finish (8 shards)"
while :; do
  ok=$(kubectl get job ft50 -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  fail=$(kubectl get job ft50 -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
  run=$(kubectl get pods -l job-name=ft50 -n $NS --no-headers 2>/dev/null | grep -c Running)
  done_n=$(cat "$HOST/sweep_out/fulltrain50"/result_shard*.jsonl 2>/dev/null | wc -l)
  [ "${ok:-0}" -ge 8 ] 2>/dev/null && { say "ft50 COMPLETE ($done_n/300 models)"; break; }
  if [ "${run:-0}" -eq 0 ] && [ "$done_n" -gt 0 ]; then
    say "ft50 has no running pods (succeeded=${ok:-0} failed=${fail:-0}, $done_n/300) -- proceeding"
    break
  fi
  sleep 300
done

# quota check: 8 pods x 8Gi. Measured peak for this workload is 2Gi, so 8Gi is 4x headroom.
for i in $(seq 1 60); do
  used=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null)
  hard=$(kubectl get resourcequota -n $NS -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
  free=$(( ${hard:-0} - ${used:-0} ))
  say "GPU headroom: $free (used ${used:-?}/${hard:-?})"
  [ "$free" -ge 8 ] && break
  say "  waiting for 8 free GPUs..."
  sleep 300
done

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

sleep 30
pods=$(kubectl get pods -l job-name=abl-train -n $NS --no-headers 2>/dev/null | wc -l)
if [ "$pods" -eq 0 ]; then
  say "WARN no pods created -- quota rejection?"
  kubectl describe job abl-train -n $NS 2>/dev/null | grep -m2 "forbidden\|FailedCreate" | tee -a "$LOG"
else
  say "abl-train up: $pods pods"
  kubectl get pods -l job-name=abl-train -n $NS \
    -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase' --no-headers | tee -a "$LOG"
fi
say "ablation launched; ~297 runs / 8 shards x ~47 min = ~29 h"

#!/usr/bin/env bash
# Resume the llmprod pipeline from the preflight stage. Generation is DONE (1752 candidates
# across a1/a5/a10/a12); only preflight + screen remain. Runs ON THE LOGIN NODE.
#
# WHY THIS EXISTS: the first preflight died at 50/1699 with
#   torch.OutOfMemoryError: ... GPU 0 has 79.18 GiB of which 14.69 MiB is free.
#   Process 28242 has 78.53 GiB memory in use.
# It was submitted with gpus=0 but PINNED to dgx-h100, and that node exposes its GPUs to
# containers that never requested one -- so the DiversityGate's torch.fx shape-prop tried to
# trace on a card another tenant had filled. The OOM path degrades gracefully
# ("untraceable for the graded check"), but the follow-on CUBLAS_STATUS_ALLOC_FAILED does not,
# and the job died.
#
# FIX IS TWO PARTS, both needed:
#   1. CUDA_VISIBLE_DEVICES="" -- makes the CPU-only claim in the preflight banner actually
#      true. Without it the "no GPU requested" resource spec is not enforcement, just intent.
#   2. no nodeSelector -- preflight does zero GPU work, so pinning it to a shared DGX only
#      exposed it to another tenant's memory. It runs anywhere.
# The screen still needs a GPU and still goes to the H100: all 8 RTX cards are held by ft50
# for roughly another 40 h, so there is nowhere else for it.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
POD=/a/mm
GA=$POD/nn-gpt/ab/gpt/brute/ga/denoise
SRC=$POD/nn-dataset
OUT=$POD/sweep_out/llmprod
OUT_H=$HOST/sweep_out/llmprod
MAINDB=$SRC/db/ab.nn.db
COSMAX=0.90
SHARDS=4

ROOTSETUP="R=/scratch/root; mkdir -p \$R/out; \
for l in ab README.md data train validation; do ln -sfn $SRC/\$l \$R/\$l; done"

wait_job() {
  local name="$1" want="$2" ok fail
  while :; do
    ok=$(kubectl get job "$name" -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
    fail=$(kubectl get job "$name" -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
    [ "${ok:-0}" -ge "$want" ] 2>/dev/null && return 0
    [ "${fail:-0}" -gt 0 ] 2>/dev/null && return 1
    sleep 60
  done
}

mkjob() {  # mkjob <name> <completions> <gpus> <mem> <command> [node]
  local name="$1" comp="$2" gpus="$3" mem="$4" cmd="$5" node="${6:-}"
  local placement=""
  if [ "$node" = "h100" ]; then
    placement=$'      nodeSelector: {kubernetes.io/hostname: dgx-h100}\n      tolerations:\n      - {key: dgx, operator: Equal, value: access, effect: NoSchedule}'
  fi
  kubectl delete job "$name" -n $NS --ignore-not-found >/dev/null 2>&1; sleep 3
  cat > "/tmp/$name.yaml" <<EOF
apiVersion: batch/v1
kind: Job
metadata: {name: $name, namespace: $NS}
spec:
  backoffLimit: 0
  completions: $comp
  parallelism: $comp
  completionMode: Indexed
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
$placement
      containers:
      - name: ml
        image: abrainone/ai-linux:cu12.6.3-latest
        command: ["/bin/bash","-lc"]
        args: ["$cmd"]
        resources:
          limits: {cpu: 8, memory: $mem$(if [ "$gpus" != 0 ]; then echo ", nvidia.com/gpu: $gpus"; fi)}
          requests: {cpu: 4, memory: $mem$(if [ "$gpus" != 0 ]; then echo ", nvidia.com/gpu: $gpus"; fi)}
        securityContext: {runAsUser: 1067, runAsGroup: 1376, allowPrivilegeEscalation: false}
        volumeMounts:
        - {mountPath: /a/mm, name: v0}
        - {mountPath: /scratch, name: scratch}
        - {mountPath: /dev/shm, name: shm}
      volumes:
      - {name: v0, hostPath: {path: $HOST, type: Directory}}
      - {name: scratch, emptyDir: {sizeLimit: 30Gi}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 8Gi}}
EOF
  kubectl apply -f "/tmp/$name.yaml" >/dev/null || return 1
  sleep 20
  if [ -z "$(kubectl get pods -l job-name=$name -n $NS --no-headers 2>/dev/null)" ]; then
    echo "  WARN no pod created for $name:"
    kubectl describe job "$name" -n $NS 2>/dev/null | grep -m2 "forbidden\|FailedCreate" | sed 's/^/    /'
  fi
}

ADAPTERS="1 5 10 12"
RUNDIRS=""; for K in $ADAPTERS; do RUNDIRS="$RUNDIRS $OUT/a$K/run"; done

echo "######## LLMPROD RESUME (preflight + screen) $(date -Is) ########"
NGEN=0
for K in $ADAPTERS; do
  n=$(ls "$OUT_H/a$K/run/cand" 2>/dev/null | wc -l); NGEN=$((NGEN+n))
done
echo "    $NGEN candidates already generated across 4 adapters"

# The diversity store from the dead run is discarded. It is a GREEDY, STATEFUL accumulator:
# the 20 architectures it accepted before the crash are in it, so reusing it would reject
# their near-duplicates on this pass while their own pass/fail records live in a pass_ids
# file that was never written. Starting clean keeps the store and the id list consistent.
rm -f "$OUT_H/diversity_store.pt"

echo "--- preflight over all 4 adapters (CPU-forced, unpinned) ---"
mkjob lp-prod-pre 1 0 16Gi \
  "$ROOTSETUP; cd \$R && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/preflight.py \
   --run-dirs$RUNDIRS --out $OUT/pre --arch-dedupe --arch-graded --cos-max $COSMAX \
   --arch-dedupe-db $MAINDB --arch-store $OUT/diversity_store.pt --strip-dead-code \
   2>&1 | tee $OUT/preflight.log"
PRE_OK=1; wait_job lp-prod-pre 1 || PRE_OK=0
NPASS=$(wc -l < "$OUT_H/pre/pass_ids.txt" 2>/dev/null || echo 0)
echo "    $NPASS of $NGEN survived preflight"
if [ "$PRE_OK" -eq 0 ]; then
  echo "FATAL preflight job did not succeed ($NPASS ids written); last log lines:"
  tail -5 "$OUT_H/preflight.log" 2>/dev/null | sed 's/^/    /'
  exit 1
fi
[ "$NPASS" -gt 0 ] || { echo "preflight completed but nothing was novel"; exit 3; }

echo "--- 1-epoch screen ($SHARDS shards, h100) ---"
mkjob lp-prod-scr $SHARDS 1 8Gi \
  "$ROOTSETUP; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/screen_and_train.py \
   --run-dirs$RUNDIRS --only-ids $OUT/pre/pass_ids.txt --out $OUT/eval --gate 30.0 \
   --screen-only --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS \
   2>&1 | tee $OUT/screen_\$JOB_COMPLETION_INDEX.log" h100
wait_job lp-prod-scr $SHARDS || echo "    WARN screen had failures"

python3 - "$OUT_H" <<'PYEOF'
import glob, json, statistics as st, sys
recs = []
for p in glob.glob(f"{sys.argv[1]}/eval/results_shard*.json") + glob.glob(f"{sys.argv[1]}/eval/results.json"):
    try: recs += json.load(open(p))
    except Exception: pass
sc = [r["ep1"] for r in recs if r.get("ep1") is not None]
byad = {}
for r in recs:
    if r.get("ep1") is None: continue
    byad.setdefault(str(r.get("id","")).split("_")[0], []).append(r["ep1"])
for k in sorted(byad):
    v = byad[k]
    print(f"  {k}: {len(v)} scored, >=32.5: {sum(1 for x in v if x>=32.5)}, max {max(v):.2f} dB")
if sc:
    print(f"SCREEN RESULT: {len(sc)} scored | median {st.median(sc):.2f} dB | max {max(sc):.2f} dB "
          f"| >=32.5: {sum(1 for s in sc if s>=32.5)} | >=31.5: {sum(1 for s in sc if s>=31.5)}")
else:
    print("SCREEN RESULT: nothing scored")
PYEOF
echo "######## LLMPROD RESUME COMPLETE $(date -Is) ########"

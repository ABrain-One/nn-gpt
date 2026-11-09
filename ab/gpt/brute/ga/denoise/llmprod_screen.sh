#!/usr/bin/env bash
# Coordinator for the production LLM-gen batch: wait for generation, preflight, 1-epoch screen.
# Runs ON THE LOGIN NODE (pods cannot create jobs), same pattern as run3_evolve.sh.
#
# WHY IT WAITS RATHER THAN RUNNING NOW: the namespace memory quota is the binding resource,
# not GPUs (measured 544Gi of 560Gi used, 11 of 16 GPUs). ft50's eight 48Gi pods hold most
# of it, and llmprod-gen holds 64Gi more. Submitting the screen immediately gets it killed
# by the quota admission check with FailedCreate and NO POD -- silent unless you go looking
# at the job's events. Generation releases its 64Gi on exit, which is the headroom the
# screen then uses.
#
# SCREEN IS 1 EPOCH. screen_and_train.py hardcodes {"epoch": 1} in its stage-1 prm and
# writes nothing to the DB (save_to_db=False). --screen-only skips the full-train stage.
# Epoch-1 ordering is a WEAK RANKER -- run3_evolve.sh records KPN scoring 34.60 dB at 50
# epochs from a bottom-tier epoch-1 score -- so treat the output as a filter, not a ranking.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
POD=/a/mm
GA=$POD/nn-gpt/ab/gpt/brute/ga/denoise
GA_H=$HOST/nn-gpt/ab/gpt/brute/ga/denoise
SRC=$POD/nn-dataset
OUT=$POD/sweep_out/llmprod
OUT_H=$HOST/sweep_out/llmprod
MAINDB=$SRC/db/ab.nn.db
COSMAX=0.90
SHARDS=4

ROOTSETUP="R=/scratch/root; mkdir -p \$R/out; \
for l in ab README.md data train validation; do ln -sfn $SRC/\$l \$R/\$l; done"

wait_job() {  # wait_job <name> <completions>
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
  # node="h100" pins to the DGX; anything else leaves scheduling to the RTX fleet. The
  # H100 needs BOTH the hostname selector and the dgx taint toleration -- with the
  # toleration missing the pod stays Pending forever against a NoSchedule taint.
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
  # a quota rejection produces a job with NO pod and no failure count -- catch it explicitly
  sleep 20
  if [ -z "$(kubectl get pods -l job-name=$name -n $NS --no-headers 2>/dev/null)" ]; then
    echo "  WARN no pod created for $name:"
    kubectl describe job "$name" -n $NS 2>/dev/null | grep -m2 "forbidden\|FailedCreate" | sed 's/^/    /'
  fi
}

ADAPTERS="1 5 10 12"
RUNDIRS=""; for K in $ADAPTERS; do RUNDIRS="$RUNDIRS $OUT/a$K/run"; done

echo "######## LLMPROD SCREEN $(date -Is) ########"
echo "--- waiting for 4 parallel generations (llmprod-a{1,5,10,12}) ---"
NGEN=0
for K in $ADAPTERS; do
  wait_job "llmprod-a$K" 1 || echo "    WARN llmprod-a$K did not succeed; using whatever it wrote"
  n=$(ls "$OUT_H/a$K/run/cand" 2>/dev/null | wc -l)
  echo "    adapter cycle$K: $n candidates"
  NGEN=$((NGEN+n))
done
echo "    generation done: $NGEN candidates across 4 adapters"
[ "$NGEN" -gt 0 ] || { echo "FATAL no candidates"; exit 1; }

# ONE preflight over ALL FOUR run dirs, not four separate ones. The DiversityGate is greedy
# and stateful, so a shared pass dedupes ACROSS adapters as well as within each -- which is
# the number that matters: if the four adapters mostly re-derive each other's architectures,
# four times the candidates does not mean four times the distinct models, and only a joint
# pass can measure that. Per-adapter yield is still recoverable from the id-prefix (p1/p5/
# p10/p12), so the head-to-head comparison survives.
# PREFLIGHT RUNS ON CPU AND IS NOT PINNED. It was previously submitted with gpus=0 but
# pinned to dgx-h100, which killed the 2026-08-05 run at candidate 50/1699:
#   torch.OutOfMemoryError: GPU 0 has 79.18 GiB of which 14.69 MiB is free.
#   Process 28242 has 78.53 GiB memory in use.
# Two things made that possible. (1) Omitting nvidia.com/gpu is a SCHEDULING statement, not
# an isolation one -- with no GPU requested the device plugin injects nothing, and on this
# node the container still saw all eight cards. (2) The graded diversity check builds each
# candidate through mutator/lemur_io.py:169, which is unconditionally
#   params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# so every candidate was constructed on a card another tenant had filled. preflight.py's own
# `device='cpu'` covers the build/forward validity check only -- a different loader.
# CUDA_VISIBLE_DEVICES= makes is_available() false, which is the only thing that actually
# forces the graded path onto the CPU. Cost of doing so is ~6 min over the whole batch
# (measured 0.62 s/candidate CPU vs 0.42 s/candidate GPU); there is no batch and no
# throughput here, just one 1x3x64x64 shape-prop per candidate.
echo "--- preflight over all 4 adapters (dedupe vs corpus AND across adapters) ---"
mkjob lp-prod-pre 1 0 16Gi \
  "$ROOTSETUP; cd \$R && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/preflight.py \
   --run-dirs$RUNDIRS --out $OUT/pre --arch-dedupe --arch-graded --cos-max $COSMAX \
   --arch-dedupe-db $MAINDB --arch-store $OUT/diversity_store.pt --strip-dead-code \
   2>&1 | tee $OUT/preflight.log"
PRE_OK=1; wait_job lp-prod-pre 1 || PRE_OK=0
NPASS=$(wc -l < "$OUT_H/pre/pass_ids.txt" 2>/dev/null || echo 0)
echo "    $NPASS of $NGEN survived preflight"

# FAIL LOUD, AND DISTINGUISH THE TWO ZEROS. This block used to be
#   [ "$NPASS" -gt 0 ] || { echo "nothing passed preflight"; exit 0; }
# and that `exit 0` is why the crash above went unnoticed for five hours: a preflight that
# died at candidate 50 and a preflight that legitimately found nothing novel both left the
# coordinator reporting SUCCESS with no pass_ids.txt. Any wrapper watching the exit code saw
# a clean run. A crashed job must exit non-zero; only a completed job that genuinely deduped
# everything away is allowed to be a quiet zero.
if [ "$PRE_OK" -eq 0 ]; then
  echo "FATAL preflight job did not succeed ($NPASS ids written); last log lines:"
  tail -5 "$OUT_H/preflight.log" 2>/dev/null | sed 's/^/    /'
  exit 1
fi
[ "$NPASS" -gt 0 ] || { echo "preflight completed but nothing was novel"; exit 3; }

echo "--- 1-epoch screen ($SHARDS shards) ---"
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
    pre = str(r.get("id","")).split("_")[0]
    byad.setdefault(pre, []).append(r["ep1"])
for k in sorted(byad):
    v = byad[k]
    print(f"  {k}: {len(v)} scored, >=32.5: {sum(1 for x in v if x>=32.5)}, max {max(v):.2f} dB")
if sc:
    print(f"SCREEN RESULT: {len(sc)} scored | median {st.median(sc):.2f} dB | max {max(sc):.2f} dB "
          f"| >=32.5: {sum(1 for s in sc if s>=32.5)} | >=31.5: {sum(1 for s in sc if s>=31.5)}")
else:
    print("SCREEN RESULT: nothing scored")
PYEOF
echo "######## LLMPROD SCREEN COMPLETE $(date -Is) ########"

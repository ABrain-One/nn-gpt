#!/usr/bin/env bash
# Stage 2 only: WAIT on the already-running lp-prod-pre job, then run the 1-epoch screen.
#
# WHY A SEPARATE FILE RATHER THAN A RESTART OF llmprod_resume.sh: that script was edited
# while it was executing. Bash reads a script by byte offset and re-seeks between commands,
# so inserting lines ahead of the interpreter's position makes it resume mid-line. The
# coordinator was killed for that reason. Re-running it would also re-enter mkjob, which
# DELETES AND RECREATES lp-prod-pre and would throw away the preflight progress that is
# currently live. This script only observes that job; it never recreates it.
set -uo pipefail

NS=b-b-ahmed
HOST=/shared/ssd/home/b-b-ahmed/thesis
POD=/a/mm
GA=$POD/nn-gpt/ab/gpt/brute/ga/denoise
SRC=$POD/nn-dataset
OUT=$POD/sweep_out/llmprod
OUT_H=$HOST/sweep_out/llmprod
SHARDS=4

ROOTSETUP="R=/scratch/root; mkdir -p \$R/out; \
for l in ab README.md data train validation; do ln -sfn $SRC/\$l \$R/\$l; done"

ADAPTERS="1 5 10 12"
RUNDIRS=""; for K in $ADAPTERS; do RUNDIRS="$RUNDIRS $OUT/a$K/run"; done

echo "######## LLMPROD STAGE 2 $(date -Is) ########"
echo "--- observing the live lp-prod-pre job (not recreating it) ---"
PRE_OK=0
while :; do
  ok=$(kubectl get job lp-prod-pre -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  fail=$(kubectl get job lp-prod-pre -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
  [ "${ok:-0}" -ge 1 ] 2>/dev/null && { PRE_OK=1; break; }
  [ "${fail:-0}" -gt 0 ] 2>/dev/null && { PRE_OK=0; break; }
  sleep 60
done

NPASS=$(wc -l < "$OUT_H/pre/pass_ids.txt" 2>/dev/null || echo 0)
echo "    preflight succeeded=$PRE_OK | $NPASS ids written"

# FAIL LOUD. The previous coordinator turned a crashed preflight into `exit 0`, so a job that
# died at candidate 50/1699 was indistinguishable from one that legitimately found nothing
# novel -- five hours passed before anyone noticed. A crash exits 1; a clean run that deduped
# everything away exits 3. Both are visible to a wrapper watching the exit code.
if [ "$PRE_OK" -eq 0 ]; then
  echo "FATAL preflight job did not succeed ($NPASS ids written); last log lines:"
  tail -8 "$OUT_H/preflight.log" 2>/dev/null | sed 's/^/    /'
  exit 1
fi
[ "$NPASS" -gt 0 ] || { echo "FATAL preflight completed but nothing was novel"; exit 3; }

# The screen TRAINS (one epoch), so it genuinely needs a GPU -- unlike preflight, which only
# shape-props a single 1x3x64x64 tensor per candidate. All 8 RTX cards are held by ft50 for
# roughly another 40 h, so the H100 is the only place it can go.
echo "--- 1-epoch screen ($SHARDS shards, h100) ---"
kubectl delete job lp-prod-scr -n $NS --ignore-not-found >/dev/null 2>&1; sleep 3
cat > /tmp/lp-prod-scr.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata: {name: lp-prod-scr, namespace: $NS}
spec:
  backoffLimit: 0
  completions: $SHARDS
  parallelism: $SHARDS
  completionMode: Indexed
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {kubernetes.io/hostname: dgx-h100}
      tolerations:
      - {key: dgx, operator: Equal, value: access, effect: NoSchedule}
      containers:
      - name: ml
        image: abrainone/ai-linux:cu12.6.3-latest
        command: ["/bin/bash","-lc"]
        args:
        # SINGLE-QUOTED YAML SCALAR ON PURPOSE. The first attempt used a double-quoted one and
        # kubectl rejected the whole manifest with
        #   error converting YAML to JSON: found unknown escape character
        # because the heredoc was over-escaped (three backslashes produced a literal
        # backslash-dollar) and YAML double-quoted scalars accept only a fixed escape set,
        # which backslash-dollar is not in. In a single-quoted scalar the backslash has no
        # meaning at all, so a shell command full of variable references and backslashes can
        # be embedded verbatim; the only character needing care is ' (doubled), and there is
        # none here. The two references below are escaped ONCE so this unquoted heredoc leaves
        # them for the container's shell instead of expanding them on the login node.
        #
        # NOTE FOR EDITORS: this heredoc is UNQUOTED, so it expands even inside comment lines.
        # Writing a bare dollar-VAR in a comment here aborts the script under `set -u`. That
        # is exactly how the second attempt died -- an explanatory comment, not the command.
        - '$ROOTSETUP; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/screen_and_train.py --run-dirs$RUNDIRS --only-ids $OUT/pre/pass_ids.txt --out $OUT/eval --gate 30.0 --screen-only --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS 2>&1 | tee $OUT/screen_\$JOB_COMPLETION_INDEX.log'
        resources:
          limits: {cpu: 8, memory: 8Gi, nvidia.com/gpu: 1}
          requests: {cpu: 4, memory: 8Gi, nvidia.com/gpu: 1}
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
kubectl apply -f /tmp/lp-prod-scr.yaml || { echo "FATAL could not apply screen job"; exit 1; }

# A namespace-quota rejection produces a job with NO pod and NO failure count. That state is
# byte-for-byte indistinguishable from "still scheduling" unless the job's events are read --
# it silently held the LLM generation for 2.5 min earlier in this project.
sleep 30
if [ -z "$(kubectl get pods -l job-name=lp-prod-scr -n $NS --no-headers 2>/dev/null)" ]; then
  echo "FATAL no pod created for lp-prod-scr:"
  kubectl describe job lp-prod-scr -n $NS 2>/dev/null | grep -m3 "forbidden\|FailedCreate" | sed 's/^/    /'
  exit 1
fi
kubectl get pods -l job-name=lp-prod-scr -n $NS \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase' --no-headers

SCR_OK=0
while :; do
  ok=$(kubectl get job lp-prod-scr -n $NS -o jsonpath='{.status.succeeded}' 2>/dev/null)
  fail=$(kubectl get job lp-prod-scr -n $NS -o jsonpath='{.status.failed}' 2>/dev/null)
  [ "${ok:-0}" -ge "$SHARDS" ] 2>/dev/null && { SCR_OK=1; break; }
  [ "${fail:-0}" -ge "$SHARDS" ] 2>/dev/null && break
  sleep 60
done
[ "$SCR_OK" -eq 1 ] || echo "WARN screen had failures; summarising whatever was written"

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
echo "######## LLMPROD STAGE 2 COMPLETE $(date -Is) ########"

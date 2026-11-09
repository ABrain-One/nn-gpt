#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for PRODUCTION RUN 5 (broadened 25-seed pool:
# Run-4's 23 seeds + DenoiseNAFNet + DenoiseRestormerLite). Slurm cgroup teardown
# kills the whole job on session end; this relaunches and the GA resumes losslessly
# from its checkpoint + checksum cache (warm-started from Run 4's cache).
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/run5.log"
if pgrep -f "seed_ga_R5_checkpoint.pkl" >/dev/null 2>&1; then
  echo "[sup5] already running — abort double-launch"; exit 0
fi
# Self-heal teardown-poisoned SEEDS (see run4_supervisor.sh rationale).
preclean_poisoned_seeds() {
  python3 - "$D/seed_ga_R5_cache.json" <<'PY' 2>>"$LOG"
import json,sys
p=sys.argv[1]
try: c=json.load(open(p))
except Exception: sys.exit(0)
v=[k for k,e in c.items() if isinstance(e,dict) and not e.get("parent")
   and not e.get("diversity_reject") and (e.get("psnr") or 0)==0 and not e.get("edge_reject")]
if v:
    for k in v: del c[k]
    json.dump(c,open(p,"w"))
    print(f"[sup5] pre-clean: cleared {len(v)} teardown-poisoned seed(s) for re-eval")
PY
}
for attempt in $(seq 1 200); do
  echo "[sup5] === attempt $attempt $(date) ===" | tee -a "$LOG"
  preclean_poisoned_seeds
  bash "$D/launch_run5.sh" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== DONE in" "$LOG"; then echo "[sup5] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[sup5] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

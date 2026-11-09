#!/usr/bin/env bash
# Crash/teardown-resilient wrapper for PRODUCTION RUN 4 (grown 23-seed pool).
# Slurm cgroup teardown kills the whole job on session end; this relaunches and the
# GA resumes losslessly from its checkpoint + checksum cache. Guard on the unique
# checkpoint path (pgrep -f sees the full cmdline; ps -o cmd truncates it).
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
LOG="$D/run4.log"
if pgrep -f "seed_ga_R4_checkpoint.pkl" >/dev/null 2>&1; then
  echo "[sup4] already running — abort double-launch"; exit 0
fi
# Self-heal teardown-poisoned SEEDS: a big/slow seed killed mid-train gets cached as
# psnr=0 and then skipped forever on resume. Before each attempt, drop any seed entry
# with psnr==0 that was NOT edge-rejected, so it re-evaluates. (Offspring untouched.)
preclean_poisoned_seeds() {
  python3 - "$D/seed_ga_R4_cache.json" <<'PY' 2>>"$LOG"
import json,sys
p=sys.argv[1]
try: c=json.load(open(p))
except Exception: sys.exit(0)
v=[k for k,e in c.items() if isinstance(e,dict) and not e.get("parent")
   and not e.get("diversity_reject") and (e.get("psnr") or 0)==0 and not e.get("edge_reject")]
if v:
    for k in v: del c[k]
    json.dump(c,open(p,"w"))
    print(f"[sup4] pre-clean: cleared {len(v)} teardown-poisoned seed(s) for re-eval")
PY
}
for attempt in $(seq 1 200); do
  echo "[sup4] === attempt $attempt $(date) ===" | tee -a "$LOG"
  preclean_poisoned_seeds
  bash "$D/launch_run4.sh" >> "$LOG" 2>&1
  rc=$?
  if grep -qa "=== DONE in" "$LOG"; then echo "[sup4] complete rc=$rc" | tee -a "$LOG"; break; fi
  echo "[sup4] exited rc=$rc — resuming in 30s" | tee -a "$LOG"; sleep 30
done

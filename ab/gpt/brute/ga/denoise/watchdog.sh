#!/usr/bin/env bash
# Watchdog for the current batch of jobs. EXITS (non-zero) the moment something is wrong, and
# exits 0 when everything has finished, so the caller is notified either way instead of having to
# poll by hand.
#
# WHY EXIT RATHER THAN LOOP-AND-LOG. A monitor that only writes to a file is a monitor nobody
# reads. Exiting on the first real problem turns it into an alert.
#
# WHAT COUNTS AS A PROBLEM:
#   * a pod in Error / CrashLoopBackOff / OOMKilled / Evicted
#   * a Job reporting failed > 0
#   * a STALL: a job whose progress counter has not moved in $STALL_MIN minutes while its pods
#     are still Running. This is the failure mode that has actually bitten us -- jobs that sit
#     "Running" for an hour having silently died inside, e.g. the 60-minute dead wait caused by
#     matching "Succeeded" when kubectl prints "Completed".
#
#     STALL_MIN MUST EXCEED THE SLOWEST NATURAL PROGRESS INTERVAL, or the alert is noise. It was
#     45 min and produced a false positive on llm-fulltrain: that job writes ONE result row per
#     completed 50-epoch run (~2800-3000 s each) and runs only 2-4 pods at a time, so the real gap
#     between increments is ~50 min. Raised to 90. Any job whose unit of work takes longer than
#     that needs a finer-grained progress counter, not a bigger threshold.
#
# EXIT CODES -- each one is an alert, and the reason is written to the log before exiting:
#   2 job reports failed>0     3 pod Error/CrashLoop/Evicted     4 stalled progress
#   5 container OOMKilled      6 memory >= MEM_ALERT_PCT of limit  7 internal failure rate high
#   0 all jobs complete
#
# WHAT IS DELIBERATELY NOT AN ALERT:
#   * a FEW model failures inside a shard. Some generated architectures do not train; that is
#     expected, the shard records it and moves on. Only the RATE matters (exit 7), so a systemic
#     break is caught while the normal trickle is merely noted.
#   * the other session's midd-* jobs. Not ours; never touched.
set -uo pipefail

NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
GA=$H/nn-gpt/ab/gpt/brute/ga/denoise    # GA arms write their models-dir here, not under sweep_out
POLL="${POLL:-300}"
STALL_MIN="${STALL_MIN:-90}"
LOG="${LOG:-$H/sweep_out/watchdog.log}"
JOBS="${JOBS:-dnseed-retrain llm-fulltrain randwalk refab-ctl refab-trt}"

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# -> "<failed> <total>" from screen_and_train.py's results_shard<N>.json files.
#
# A screened candidate "failed" when it produced no ep1 number at all -- rec["error"] set, or
# ep1 missing. NOT when it merely scored below the gate: a low-PSNR model is a valid
# measurement and the whole point of screening the full distribution. Conflating the two
# would make the A/B's own signal look like a malfunction.
screen_errors() {
  python3 - "$1" <<'PY' 2>/dev/null || echo "0 0"
import glob, json, os, sys
bad = tot = 0
for p in glob.glob(os.path.join(sys.argv[1], "results_shard*.json")) + \
         glob.glob(os.path.join(sys.argv[1], "results.json")):
    try:
        rows = json.load(open(p))
    except Exception:
        continue
    for r in rows if isinstance(rows, list) else []:
        tot += 1
        if r.get("error") or r.get("ep1") is None:
            bad += 1
print(bad, tot)
PY
}

# -> "<failed> <total>" for a job whose shards write JSONL result rows with a "status" field
errors() {
  local d=""
  case "$1" in
    dnseed-retrain) d="$H/sweep_out/dnseed_retrain" ;;
    llm-fulltrain)  d="$H/sweep_out/llm_fulltrain" ;;
    sidd-train)     d="$H/sweep_out/sidd_train" ;;
    baseline-train) d="$H/sweep_out/baselines" ;;
    # The screen stage writes results_shard<N>.json -- a JSON LIST, not JSONL, and only
    # at shard exit -- so it needs its own reader rather than the line-oriented one below.
    # Handled separately and returned early.
    refab-scr-ctl) screen_errors "$H/sweep_out/refab/ctl/screen"; return ;;
    refab-scr-trt) screen_errors "$H/sweep_out/refab/trt/screen"; return ;;
    *) echo "0 0"; return ;;
  esac
  cat "$d"/result_shard*.jsonl 2>/dev/null | python3 -c '
import sys, json
bad = tot = 0
for ln in sys.stdin:
    ln = ln.strip()
    if not ln:
        continue
    try:
        r = json.loads(ln)
    except Exception:
        continue
    tot += 1
    if str(r.get("status", "")).upper() not in ("OK", "PARTIAL"):
        bad += 1
print(bad, tot)
' 2>/dev/null || echo "0 0"
}

# progress counter per job -- one number that must increase over time
progress() {
  case "$1" in
    randwalk)
      local t=0 n
      for i in 0 1 2 3 4 5 6 7; do
        n=$(ls "$H/sweep_out/randwalk/rw_${i}_models/_epoch_stats" 2>/dev/null | wc -l)
        t=$((t+n))
      done
      echo "$t" ;;
    dnseed-retrain) cat "$H"/sweep_out/dnseed_retrain/result_shard*.jsonl 2>/dev/null | wc -l ;;
    refab-ctl)      ls "$H/sweep_out/refab/ctl/run/cand" 2>/dev/null | wc -l ;;
    refab-trt)      ls "$H/sweep_out/refab/trt/run/cand" 2>/dev/null | wc -l ;;
    # The SCREEN stage, not generation. Counts per-candidate dirs, because
    # screen_and_train.py writes results_shard<N>.json only once at exit -- a
    # results-file counter would read 0 for the whole run and trip the stall alert
    # at STALL_MIN every time. sdir (screen_and_train.py:243) is created before each
    # candidate is evaluated, so this increments roughly every ~80 s per shard.
    # The doubled "screen/screen" is real: --out is .../<arm>/screen and the script
    # appends its own "screen" component underneath.
    refab-scr-ctl)  ls "$H/sweep_out/refab/ctl/screen/screen" 2>/dev/null | wc -l ;;
    refab-scr-trt)  ls "$H/sweep_out/refab/trt/screen/screen" 2>/dev/null | wc -l ;;
    # GA arms: one _epoch_stats subdir per EVALUATED candidate (run_seed_ga.py:497), created
    # before training starts, so it ticks once per candidate rather than once per generation.
    # A generation counter would be far too coarse here: at pop 50 and ~3 min per candidate a
    # generation is ~2.5 h, which is longer than any sane stall threshold.
    #
    # NOTE these live in the GA's working dir under nn-gpt, NOT in sweep_out -- the launcher
    # passes --models-dir "$D/ga_<V>_models" where $D is the script directory.
    ga-dnseed2)       ls "$GA/ga_dnseed2_models/_epoch_stats"       2>/dev/null | wc -l ;;
    ga-dnseed2-act)   ls "$GA/ga_dnseed2-act_models/_epoch_stats"   2>/dev/null | wc -l ;;
    ga-dnseed2-fast)  ls "$GA/ga_dnseed2-fast_models/_epoch_stats"  2>/dev/null | wc -l ;;
    ga-dnseed2-wide)  ls "$GA/ga_dnseed2-wide_models/_epoch_stats"  2>/dev/null | wc -l ;;
    sidd-train)    cat "$H"/sweep_out/sidd_train/result_shard*.jsonl 2>/dev/null | wc -l ;;
    llm-fulltrain) cat "$H"/sweep_out/llm_fulltrain/result_shard*.jsonl 2>/dev/null | wc -l ;;
    *)             echo 0 ;;
  esac
}

declare -A last_n last_t
now=$(date +%s)
for j in $JOBS; do last_n[$j]=$(progress "$j"); last_t[$j]=$now; done
say "watchdog up: jobs=[$JOBS] poll=${POLL}s stall=${STALL_MIN}m"

while :; do
  sleep "$POLL"
  now=$(date +%s)
  all_done=1
  for j in $JOBS; do
    # does the job still exist?
    spec=$(kubectl get job "$j" -n $NS -o jsonpath='{.spec.completions} {.status.succeeded} {.status.failed}' 2>/dev/null)
    [ -z "$spec" ] && { say "NOTE  job/$j not found (deleted or never created) -- dropping it"; continue; }
    comp=$(echo "$spec" | awk '{print ($1==""?0:$1)}')
    succ=$(echo "$spec" | awk '{print ($2==""?0:$2)}')
    fail=$(echo "$spec" | awk '{print ($3==""?0:$3)}')

    if [ "${fail:-0}" -gt 0 ]; then
      say "ALERT job/$j has failed=$fail (succeeded=$succ/$comp)"
      kubectl get pods -n $NS -l job-name="$j" --no-headers 2>/dev/null | tee -a "$LOG"
      exit 2
    fi

    bad=$(kubectl get pods -n $NS -l job-name="$j" --no-headers 2>/dev/null \
          | awk '$3=="Error"||$3=="CrashLoopBackOff"||$3=="OOMKilled"||$3=="Evicted"{print $1" "$3}')
    if [ -n "$bad" ]; then
      say "ALERT job/$j has unhealthy pod(s):"; echo "$bad" | tee -a "$LOG"
      exit 3
    fi

    # OOM is reported in the CONTAINER's terminated reason, not reliably in the STATUS column --
    # an OOM-killed container frequently shows as plain "Error" there. Check both the current and
    # the previous termination so a kill is caught even after a restart.
    oom=$(kubectl get pods -n $NS -l job-name="$j" \
            -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.containerStatuses[*].state.terminated.reason}{" "}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}' \
            2>/dev/null | grep -i 'OOMKilled')
    if [ -n "$oom" ]; then
      say "ALERT job/$j OOM-KILLED:"; echo "$oom" | tee -a "$LOG"
      exit 5
    fi

    if [ "${succ:-0}" -ge "${comp:-1}" ]; then
      say "done  job/$j complete ($succ/$comp)"
      continue
    fi
    all_done=0

    # memory headroom: warn at MEM_WARN_PCT, alert at MEM_ALERT_PCT of the container limit
    for pod in $(kubectl get pods -n $NS -l job-name="$j" --no-headers 2>/dev/null \
                 | awk '$3=="Running"{print $1}'); do
      lim=$(kubectl get pod "$pod" -n $NS \
              -o jsonpath='{.spec.containers[0].resources.limits.memory}' 2>/dev/null)
      case "$lim" in *Gi) limkb=$(( ${lim%Gi} * 1024 * 1024 ));; *Mi) limkb=$(( ${lim%Mi} * 1024 ));; *) limkb=0;; esac
      [ "${limkb:-0}" -gt 0 ] || continue
      # PID 1 is the `bash -lc "... python ..."` wrapper: its command line contains the script
      # name, so pgrep-style matching returns the 3 MB wrapper instead of the real interpreter.
      # Filter on comm and exclude PID 1.
      rss=$(kubectl exec -n $NS "$pod" -- ps -eo pid,rss,comm --no-headers 2>/dev/null \
            | awk '$1!=1 && $3=="python"{s+=$2} END{print s+0}')
      [ "${rss:-0}" -gt 0 ] || continue
      pct=$(( rss * 100 / limkb ))
      if [ "$pct" -ge "${MEM_ALERT_PCT:-92}" ]; then
        say "ALERT $pod memory ${pct}% of ${lim} (rss ${rss}KB) -- OOM imminent"
        exit 6
      elif [ "$pct" -ge "${MEM_WARN_PCT:-80}" ]; then
        say "WARN  $pod memory ${pct}% of ${lim} (rss ${rss}KB)"
      fi
    done

    # systemic internal failure: alert when the error RATE is high, not on individual failures
    read -r nbad ntot <<< "$(errors "$j")"
    if [ "${ntot:-0}" -ge "${ERR_MIN_N:-10}" ]; then
      erate=$(( nbad * 100 / ntot ))
      if [ "$erate" -ge "${ERR_ALERT_PCT:-30}" ]; then
        say "ALERT job/$j internal failure rate ${erate}% ($nbad of $ntot rows) -- systemic, not the usual trickle"
        cat "$H"/sweep_out/*/result_shard*.jsonl 2>/dev/null \
          | grep -o '"error": "[^"]\{0,110\}' | sort | uniq -c | sort -rn | head -3 | tee -a "$LOG"
        exit 7
      elif [ "$nbad" -gt 0 ]; then
        say "note  job/$j $nbad of $ntot rows failed (${erate}%) -- below the ${ERR_ALERT_PCT:-30}% alert threshold"
      fi
    fi

    n=$(progress "$j")
    if [ "$n" -gt "${last_n[$j]}" ]; then
      last_n[$j]=$n; last_t[$j]=$now
      say "ok    job/$j progress=$n succeeded=$succ/$comp"
    else
      idle=$(( (now - ${last_t[$j]}) / 60 ))
      running=$(kubectl get pods -n $NS -l job-name="$j" --no-headers 2>/dev/null | grep -c Running)
      if [ "$idle" -ge "$STALL_MIN" ] && [ "${running:-0}" -gt 0 ]; then
        say "ALERT job/$j STALLED: progress stuck at $n for ${idle}m with $running pod(s) Running"
        exit 4
      fi
      say "wait  job/$j progress=$n (idle ${idle}m) succeeded=$succ/$comp"
    fi
  done
  if [ "$all_done" -eq 1 ]; then
    say "ALL JOBS COMPLETE"
    exit 0
  fi
done

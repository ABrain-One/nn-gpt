#!/usr/bin/env bash
# Emit a status report every EVERY generations of any run, and IMMEDIATELY on death.
#
# The 5-generation cadence is the routine check-in. The death check is the one that matters:
# `struct` crashed 40 s after a restart and sat dead for eleven hours, because a K8s Job that
# finished successfully and one whose process died both read 1/1, and the only evidence was a
# traceback 2.6 MB into an untailed log. Reporting only on a generation boundary would repeat
# exactly that failure -- a dead run never reaches its next boundary, so it would go silent
# and silence would look like progress.
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVERY="${EVERY:-5}"
POLL="${POLL:-120}"
TAG="${TAG:-llm_evolve_par3}"

declare -A last_gen last_state
first=1

while true; do
  json="$(python3 "$D/run_status.py" --json --tag "$TAG" 2>/dev/null)" || { sleep "$POLL"; continue; }

  # One line per run: name|state|gen|models. Parsed in python to avoid jq (not installed).
  # The python body sits inside SINGLE quotes, so double quotes are free here and no
  # backslash escaping is needed -- f"{r[\"name\"]}" is a SyntaxError before 3.12
  # ("f-string expression part cannot include a backslash") and silently emitted nothing.
  mapfile -t lines < <(printf '%s' "$json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for r in d["runs"]:
    state = "ALIVE" if r["alive"] else ("DONE" if r["note"]=="COMPLETE" else "DEAD")
    print("|".join([r["name"], state, str(r["gen"] or 0), str(r["models"]), r["note"]]))
print("|".join(["__CORPUS__", str(d["nn_rows"]), str(d["distinct_arch"])]))
')

  report=0; reasons=()
  for ln in "${lines[@]}"; do
    [[ "$ln" == __CORPUS__* ]] && continue
    IFS='|' read -r name state gen models note <<<"$ln"
    prev_g="${last_gen[$name]:-0}"; prev_s="${last_state[$name]:-}"

    # DEATH / STALL: report the transition once, the moment it happens.
    if [[ -n "$prev_s" && "$state" != "$prev_s" && "$state" == "DEAD" ]]; then
      report=1; reasons+=("!! $name went DEAD at gen $gen ${note:+($note)}")
    fi
    if [[ -n "$prev_s" && "$state" != "$prev_s" && "$state" == "DONE" ]]; then
      report=1; reasons+=("** $name COMPLETED all generations")
    fi
    # Routine cadence: crossed a multiple of EVERY.
    if (( gen > 0 && prev_g > 0 && gen / EVERY > prev_g / EVERY )); then
      report=1; reasons+=("$name reached gen $gen")
    fi
    last_gen[$name]="$gen"; last_state[$name]="$state"
  done

  if (( first )); then first=0; report=1; reasons=("watcher armed (every $EVERY gens + on death)"); fi

  if (( report )); then
    for r in "${reasons[@]}"; do echo "$r"; done
    python3 "$D/run_status.py" --tag "$TAG" 2>&1
    echo "---"
  fi
  sleep "$POLL"
done

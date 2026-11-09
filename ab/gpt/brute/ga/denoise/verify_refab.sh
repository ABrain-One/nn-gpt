#!/usr/bin/env bash
# Wait until both A/B arms have generated enough candidates, then VERIFY the treatment took effect.
#
# WHY THIS IS A GATE AND NOT A NICE-TO-HAVE. The whole experiment rests on one assumption: that
# splitting DnSeedGA into its own reference arm actually changes which references the generator
# sees. If it does not, both arms are the same experiment and every downstream number
# (preflight yield, deployability) is measuring noise. Verifying from the generated meta.json is
# the only way to know -- the simulation on the corpus predicted 4.4% -> 25%, but a prediction is
# not a measurement, and the pods run a different code path than the simulation did.
#
# Exit codes: 0 treatment confirmed | 3 treatment did NOT take effect (A/B is void) | 2 timed out
set -uo pipefail
H=/shared/ssd/home/b-b-ahmed/thesis
MIN_CAND="${MIN_CAND:-40}"     # enough for a stable share estimate; ~3 refs each => ~120 uses
TIMEOUT_MIN="${TIMEOUT_MIN:-480}"
LOG=$H/sweep_out/refab/verify.log
mkdir -p "$(dirname "$LOG")"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for >= $MIN_CAND candidates in BOTH arms (timeout ${TIMEOUT_MIN}m)"
deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
  c=$(ls "$H/sweep_out/refab/ctl/run/cand" 2>/dev/null | wc -l)
  t=$(ls "$H/sweep_out/refab/trt/run/cand" 2>/dev/null | wc -l)
  [ "$c" -ge "$MIN_CAND" ] && [ "$t" -ge "$MIN_CAND" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "TIMEOUT after ${TIMEOUT_MIN}m (ctl=$c trt=$t) -- generation never reached the threshold"
    exit 2
  fi
  sleep 120
done
say "threshold reached (ctl=$c trt=$t); measuring reference distribution"

python3 - "$H" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, glob, sys, collections
H = sys.argv[1]

def fam(n):
    # ("LLM-", "LLM_") -- both spellings: reference architectures cited in meta.json
    # may carry either, since the corpus spans the '_'->'-' rename. Matching only
    # one halves the measured LLM reference share and would fail the gate spuriously.
    for p, k in (("DnSeedGA-", "DnSeedGA"), ("DenoiseGA-", "DenoiseGA"),
                 (("LLM-", "LLM_"), "LLM")):
        if n.startswith(p):
            return k
    return "hand-written"

res = {}
for arm in ("ctl", "trt"):
    uses = collections.Counter()
    n = 0
    for m in glob.glob(f"{H}/sweep_out/refab/{arm}/run/cand/*/meta.json"):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        n += 1
        if d.get("primary_seed"):
            uses[fam(d["primary_seed"])] += 1
        for s in (d.get("supporting_seeds") or []):
            uses[fam(s)] += 1
    tot = sum(uses.values()) or 1
    res[arm] = (n, tot, uses)
    label = "CONTROL (GA merged)" if arm == "ctl" else "TREATMENT (GAseed split)"
    print(f"  {label:26} candidates={n:>4} ref_uses={tot:>5}")
    for k, v in sorted(uses.items()):
        print(f"      {k:14} {v:>5}  {100*v/tot:>5.1f}%")

dn_c = 100 * res["ctl"][2].get("DnSeedGA", 0) / (res["ctl"][1] or 1)
dn_t = 100 * res["trt"][2].get("DnSeedGA", 0) / (res["trt"][1] or 1)
print(f"\n  DnSeedGA share -- control {dn_c:.1f}%  treatment {dn_t:.1f}%  (production was 4.4%)")

# The treatment must both RISE and rise MATERIALLY. A couple of points could be sampling noise;
# the simulation predicted ~25%, so anything under 15% means the split is not being applied in the
# pod even if the numbers moved a little.
ok = dn_t >= 15.0 and dn_t > dn_c + 5.0
print(f"  VERDICT: {'treatment CONFIRMED' if ok else 'TREATMENT DID NOT TAKE EFFECT -- A/B IS VOID'}")
sys.exit(0 if ok else 3)
PYEOF
rc=${PIPESTATUS[0]}
say "verify exit rc=$rc"
exit $rc

#!/usr/bin/env bash
# LLM denoise generation, in-container. Three stages, in order:
#   1. MAC audit of the seed corpus -- the whole reason to run this in-container. On the
#      submit node measure_cost() has no torch, _seed_is_edge_legal fails OPEN, and every
#      seed passes unchecked. Here the gate is real, so this also settles whether
#      DenoiseUnet2F/7F/10F really are 14005/30152/29054 MMac or whether that comment is
#      stale.
#   2. Stage the LLM weights into a SHARED HF_HOME on /a/mm so later pods reuse them
#      instead of each pulling ~15 GB.
#   3. Generate n candidates per arm (cold / seeded) with full prompt capture.
#
# Deliberately ONE pod, not an indexed fan-out: the first run is a yield check, and 8 pods
# each pulling 15 GB into a share sitting at 99% is how the valcrops eviction happened.
set -uo pipefail

N="${N:-5}"
SUP="${SUP:-3}"
LLM="${LLM:-Qwen/Qwen2.5-Coder-7B-Instruct}"
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
OUTD=/a/mm/sweep_out/llm_gen
LOG="$OUTD/run.log"
mkdir -p "$OUTD"; : > "$LOG"
log(){ echo "$@" | tee -a "$LOG"; }

export HF_HOME=/a/mm/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=/a/mm/nn-gpt
mkdir -p "$HF_HOME"

log "=== LLM DENOISE GEN  n=$N/arm  sup=$SUP  llm=$LLM ==="
log "=== $(date -Is) on $(hostname) ==="
log "free on /a/mm: $(df -h /a/mm | awk 'NR==2{print $4" ("$5" used)"}')"

cd "$GA" || { log "FATAL cannot cd $GA"; exit 1; }

# ---- 1. MAC audit -------------------------------------------------------------------
log ""
log "--- MAC audit (cap 9053 MMac) ---"
# NOTE ON THE HEREDOC PLACEMENT. It must sit on `python` and BEFORE the pipe. Written as
#   python - 2>&1 | tee -a "$LOG" <<'PYEOF'
# bash attaches the heredoc to the LAST command in the pipeline -- so `tee` swallowed the
# script and echoed it into the log, while `python -` read the pod's empty stdin and exited
# silently. Both this audit and the weight staging below "ran" and produced nothing: the
# log filled with Python source and hf_cache stayed at 8.5K. Exit status was 0 throughout.
python -u - <<'PYEOF' 2>&1 | tee -a "$LOG"
import os, sys, glob
sys.path.insert(0, os.getcwd())
# edge_cost.measure_cost() lazily imports mutator.lemur_io, and the `mutator` PACKAGE
# lives in a different tree: ab/gpt/brute/ast. Putting only the denoise dir on sys.path
# lets `import edge_cost` succeed and then fails inside measure_cost with
# "No module named 'mutator'" for every model -- which is what the first audit reported.
# run_seed_ga.py adds both paths (its lines 31-32); this must do the same.
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")
import edge_cost as EC
nd = "/a/mm/nn-dataset/ab/nn/nn"
names = ["DenoiseUnet1F","DenoiseUnet2F","DenoiseUnet3F","DenoiseUnet4F","DenoiseUnet5F",
         "DenoiseUnet6F","DenoiseUnet7F","DenoiseUnet8F","DenoiseUnet9F","DenoiseUnet10F",
         "LiteDenoiseNet"]
print(f"{'model':<20}{'MMac':>10}  verdict")
for n in names:
    p = os.path.join(nd, f"{n}.py")
    if not os.path.exists(p):
        print(f"{n:<20}{'-':>10}  no source"); continue
    try:
        c = EC.measure_cost(open(p).read())
    except Exception as e:
        print(f"{n:<20}{'ERR':>10}  {type(e).__name__}: {e}"); continue
    if not c.get("ok"):
        print(f"{n:<20}{'?':>10}  unmeasurable: {c.get('err')}"); continue
    m = c["macs"] / 1e6
    print(f"{n:<20}{m:>10.0f}  {'OK' if m <= 9053 else 'OVER CAP'}")
PYEOF

# ---- 2. stage weights ---------------------------------------------------------------
log ""
log "--- staging $LLM into $HF_HOME ---"
python - "$LLM" <<'PYEOF' 2>&1 | tail -5 | tee -a "$LOG"
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=[
    "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("staged at", p)
PYEOF
SZ=$(du -sm "$HF_HOME" 2>/dev/null | cut -f1)
log "hf_cache now: ${SZ:-0} MB"
# A 7B in safetensors is ~15 GB. Anything under 1 GB means the staging silently no-op'd
# (as it did when the heredoc was mis-attached), and generation would then try to pull the
# weights itself mid-run. Fail here instead, where the cause is obvious.
if [ "${SZ:-0}" -lt 1024 ]; then
  log "FATAL weight staging produced ${SZ:-0} MB -- expected ~15000 MB. Not generating."
  exit 1
fi

# ---- 3. generate --------------------------------------------------------------------
log ""
log "--- generating (contract check first) ---"
python -u llm_denoise.py --check-contract 2>&1 | tee -a "$LOG" || {
  log "FATAL prompt contract drift -- arms not comparable, refusing to generate"; exit 1; }

python -u llm_denoise.py --mode both --n "$N" --supporting "$SUP" \
       --llm "$LLM" --out "$OUTD/run" 2>&1 | tee -a "$LOG"
R=$?
log "=== finished $(date -Is) rc=$R ==="
exit $R

#!/usr/bin/env bash
# Launch the DEPLOYABILITY prompt ablation: three arms x N candidates, generation only.
#
# THE THREE ARMS DIFFER IN EXACTLY ONE CONTIGUOUS BLOCK OF PROMPT TEXT:
#   on       the real DEPLOYABILITY block (rules 10-12)
#   placebo  a length-matched block of non-actionable text  -> isolates CONTENT from LENGTH
#   off      nothing                                        -> the plain pre-existing prompt
# ablation_prompt_deploy.py --check-design must pass before this runs, and this script refuses to
# launch if it does not: a placebo that has drifted in length or acquired actionable content makes
# the whole run uninterpretable, and that is cheaper to catch now than after 15 GPU-hours.
#
# --run-id IS PINNED ACROSS ALL THREE ARMS AND THAT IS NOT COSMETIC. pick_references seeds its RNG
# on f"{run_id}|{shard}|{i}", and run_id defaults to a wall-clock timestamp -- so three separately
# launched jobs would show the LLM three different sets of reference architectures. References are
# the most plausible route by which a banned operator gets copied into a candidate, so that would
# confound the very thing being measured. Pinning makes candidate i see identical primary and
# supporting references in every arm, which also makes the comparison PAIRED.
#
# GENERATION ONLY -- NO TRAINING, NO DB WRITES. The endpoint (does the emitted source contain an
# unconvertible construct, and does it actually convert) is a property of the source text, so this
# never trains anything and never touches the corpus DB. That is what makes 750 candidates
# affordable: ~5 h on three GPUs instead of the ~625 GPU-hours training them would cost.
#
# NO H100 ON THIS CLUSTER. mkjob.py has no GPU-type selector and the pool serves RTX 3090/4090;
# Qwen2.5-Coder-7B in fp16 needs ~15 GB and fits in 24 GB with room for a 4k-token generation.
#
#   bash launch_prompt_deploy_abl.sh                 # 250 per arm
#   N=100 bash launch_prompt_deploy_abl.sh           # smaller pilot
set -uo pipefail
GA_H=/shared/ssd/home/b-b-ahmed/thesis/nn-gpt/ab/gpt/brute/ga/denoise
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
POD=/a/mm
N="${N:-250}"
OUT_H="${OUT_H:-/shared/ssd/home/b-b-ahmed/thesis/sweep_out/promptdeploy}"
OUT=/a/mm/sweep_out/promptdeploy
# One id for the whole experiment: it names the output dirs AND seeds reference selection.
RUN_ID="${RUN_ID:-deployabl-$(date +%Y%m%d-%H%M%S)}"
# VARIANT selects which guidance block is the treatment. v1 = convertibility only (rules 10-12,
# tested 2026-08-12, null). v2 = adds the graph-size and attention rules, which target the failure
# that actually dominates: 37% of profiled LLM models convert fine and then run on the CPU.
VARIANT="${VARIANT:-v1}"
if [ "$VARIANT" = "v2" ]; then ARMS="v2 v2-placebo off"; PFX="dablv2"; else ARMS="on placebo off"; PFX="dabl"; fi
LOG="$OUT_H/launch.log"
mkdir -p "$OUT_H"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# ---- gate 1: the design must be valid before any GPU is spent
if ! python3 "$GA_H/ablation_prompt_deploy.py" --check-design --variant "$VARIANT" 2>&1 | tee -a "$LOG" | tail -1 | grep -q "DESIGN OK"; then
  say "FATAL --check-design did not pass; refusing to launch"
  exit 1
fi

# ---- gate 2: capacity. Three 1-GPU jobs; refuse rather than queue pods that cannot be created.
free_gpu=$(kubectl describe quota 2>/dev/null | awk '/requests.nvidia.com\/gpu/ {print $3-$2; exit}')
say "GPU headroom: ${free_gpu:-?} free"
if [ -n "${free_gpu:-}" ] && [ "$free_gpu" -lt 3 ]; then
  say "FATAL need 3 GPUs, only $free_gpu free -- wait for the rerank shards or a GA arm to finish"
  exit 1
fi

say "=== prompt deployability ablation: 3 arms x $N candidates, run_id $RUN_ID ==="
for ARM in $ARMS; do
  name="$PFX-$ARM"
  if kubectl get job "$name" >/dev/null 2>&1; then
    say "  $name exists already -- delete it first if you mean to re-run"
    continue
  fi
  y=$(mktemp /tmp/dabl-XXXXXX.json)
  # CWD nn-gpt AND PYTHONPATH nn-gpt ONLY -- copied verbatim from llm_evolve_par.sh:166, which is
  # the invocation that produced all 1,752 llmprod candidates.
  #
  # DO NOT PUT nn-dataset ON PYTHONPATH HERE. Both repos ship a top-level `ab` package
  # (nn-dataset/ab/nn and nn-gpt/ab/gpt), so whichever comes first wins and the other's submodules
  # vanish. With nn-dataset first, the first attempt died on `ModuleNotFoundError: No module named
  # 'ab.gpt'` at the LLM import -- and, worse, printed "[seeds] MAC CAP NOT ENFORCED: cost backend
  # unimportable" and carried on, so a run that survived that far would have generated candidates
  # with the 9053 MMac edge budget silently switched off.
  python3 "$GA_H/mkjob.py" --name "$name" --completions 1 --gpus 1 --cpu 4 \
      --mem 48Gi --scratch 30Gi \
      --command "cd $POD/nn-gpt && PYTHONPATH=$POD/nn-gpt python -u $GA/llm_denoise.py \
                 --mode seeded --n $N --seed-set db --deploy-guidance $ARM \
                 --run-id $RUN_ID --out $OUT/$RUN_ID/$ARM" \
      --out "$y" >/dev/null
  if kubectl apply -f "$y" >>"$LOG" 2>&1; then
    say "  launched $name -> $OUT/$RUN_ID/$ARM"
  else
    say "  APPLY FAILED for $name"
  fi
  rm -f "$y"
done

# ---- gate 3: the arms must not be silently degraded. Both of these strings appearing in a log
# mean the run is producing data that does not answer the question, and neither is fatal to the
# script itself -- which is exactly why they have to be checked for rather than relied upon to
# crash. Checked shortly after launch so a bad run is killed in minutes, not after 6 hours.
say "verifying the arms started cleanly (30s)..."
sleep 30
for ARM in $ARMS; do
  lg=$(kubectl logs "job/$PFX-$ARM" 2>/dev/null)
  [ -z "$lg" ] && { say "  $PFX-$ARM: no log yet (pod probably still Pending) -- re-check by hand"; continue; }
  if echo "$lg" | grep -q "MAC CAP NOT ENFORCED"; then
    say "  FATAL $PFX-$ARM: MAC cap disabled (ab.gpt unimportable) -- killing, data would be invalid"
    kubectl delete job "$PFX-$ARM" >/dev/null 2>&1
  elif echo "$lg" | grep -qE "ModuleNotFoundError|Traceback"; then
    say "  FATAL $PFX-$ARM: crashed on import -- killing"
    echo "$lg" | tail -5 | tee -a "$LOG"
    kubectl delete job "$PFX-$ARM" >/dev/null 2>&1
  else
    say "  $PFX-$ARM: clean start"
  fi
done

say "analyse when all three finish:"
say "  python3 $GA_H/ablation_prompt_deploy.py --run-dirs $OUT_H/$RUN_ID/{on,placebo,off} --convert"
say "note: --convert runs real TFLite conversion on CPU (~3.4 s/model, ~40 min for 750)"

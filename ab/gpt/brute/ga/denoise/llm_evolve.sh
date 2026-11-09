#!/usr/bin/env bash
# ITERATIVE LLM: successive generations, each conditioned on the previous generation's own
# survivors. This is the LLM's analogue of GA selection pressure and the one change that
# makes the two approaches comparable rather than merely co-run.
#
# Per generation:
#   1. elite_archive.py   rank everything the LLM has trained so far (last-3-epoch mean, not
#                         peak), admit greedily under the WL/cosine diversity gate, keep K.
#                         Hand-written seeds stay eligible as a FLOOR so the pool cannot walk
#                         downhill if the LLM's own output is worse than the seed corpus.
#   2. llm_denoise.py     generate N candidates seeded on that pool (--seed-set elite).
#   3. preflight.py       CPU build+forward filter. ~70% of candidates die here, and doing it
#                         on CPU is what makes a multi-generation loop affordable at all --
#                         on GPU that share cost ~80 s each.
#   4. screen_and_train   1-epoch screen of the survivors only (--only-ids).
#   5. screen_and_train   5-epoch train of those clearing the gate, into the DB, prefix
#                         LLM-Eg<N>-<arm> so every generation is separable in the DB forever.
#                         Hyphens, not underscores -- see naming.py.
#
# SINGLE POD, SEQUENTIAL, ONE GPU -- deliberately. Steps 4-5 write the LEMUR SQLite DB on
# NFSv3, where concurrent writers corrupt rather than block. The sharded screen is only safe
# because it is --screen-only; here the loop must own the DB.
#
# The whole point is that generation N+1 depends on generation N's TRAINED results, so the
# stages cannot be overlapped anyway.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
OUT="${OUT_OVERRIDE:-/a/mm/sweep_out/llm_evolve}"
ROOT=/scratch/root

GENS="${GENS:-5}"          # how many generations
N="${N:-100}"              # candidates generated per generation
K="${K:-8}"                # elite pool size
FLOOR="${FLOOR:-4}"        # hand-written seeds kept eligible
GATE="${GATE:-32.7}"       # ep1 dB floor for the 5-epoch train
EPOCHS="${EPOCHS:-5}"      # matches the GA's epoch_max=5
SUPPORTING="${SUPPORTING:-3}"
# SEED_SET is the reference pool the prompts are built from, and it is the ONE variable the
# matched-seed control needs. Default `elite` keeps every existing caller unchanged. `named`
# is the 23 hand-written seeds UNION the 18 families we built -- i.e. DnSeedGA's founding
# pool -- which is what makes an LLM-vs-GA comparison about the GENERATOR instead of about
# who was handed the better ancestors. Measured on the edge-legal crop metric, DnSeedGA's
# pool reaches 37.436 dB while the LLM's own elite pool tops out at 34.785, so the two arms
# have never started from comparable material.
SEED_SET="${SEED_SET:-elite}"
# The reference pool and the ELITE ANCHOR are two different levers and both have to move.
# Setting SEED_SET=named alone still leaves elite_archive drawing its floor seeds from
# "canonical", so 25% of prompts (--elite-share) and all of the selection pressure stay
# anchored on the weaker four. Default "canonical" preserves every existing caller.
FLOOR_SEED_SET="${FLOOR_SEED_SET:-canonical}"
# ADAPTER is the fine-tuned LoRA to generate with; empty = the base model, which is what every
# run before this one used. DEPLOY_GUIDANCE adds the deployability wording to the PROMPT, which
# is a different lever from DEPLOY_THR below: THR gates candidates after generation using the
# predictor, GUIDANCE tries to stop them being written in the first place. The gate's validated
# scope is unconditioned LLM generation, so both belong on for an LLM arm and neither has ever
# been wired into this loop.
ADAPTER="${ADAPTER:-}"
# A VALUE, not a flag: llm_denoise.py takes on|off|placebo|v2|v2-placebo and DEFAULTS to "on",
# so every earlier run already carried v1 guidance. "v2" adds the graph-size and attention
# rules, and graph size is what governs residency in this corpus -- which is the axis an LLM
# arm actually loses on. Empty here means "do not pass it", leaving llm_denoise.py's own default.
DEPLOY_GUIDANCE="${DEPLOY_GUIDANCE:-}"

mkdir -p "$OUT" "$ROOT/out"
for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
# PRIVATE_DB=1 replaces the db/ symlink with a pod-local COPY, so this run writes to a
# throwaway database and cannot collide with a production writer. Required for smoke tests:
# the loop's stage 2 writes the real DB, and two writers on the NFS-hosted SQLite file
# corrupt it rather than blocking. A smoke test that shares the DB is not a safe smoke test.
if [ "${PRIVATE_DB:-0}" = "1" ]; then
  rm -f "$ROOT/db"
  mkdir -p "$ROOT/db"
  cp "$SRC/db/ab.nn.db" "$ROOT/db/ab.nn.db" || { echo "FATAL: could not copy DB"; exit 1; }
  echo "PRIVATE_DB=1 -> writing to pod-local $ROOT/db/ab.nn.db (real DB untouched)"
fi
cd "$ROOT" || exit 1
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
[ -d "$ROOT/train" ] && [ -d "$ROOT/validation" ] || { echo "FATAL: data not visible"; exit 1; }
export HF_HOME=/a/mm/hf_cache HF_HUB_DISABLE_TELEMETRY=1
# TWO DIFFERENT ROOTS, and they must not be mixed:
#   generation  needs cwd=/a/mm/nn-gpt  -- ab/gpt/util/Const.py resolves conf/llm/*.json
#               RELATIVE TO CWD, so running it from the LEMUR root made it look for
#               /scratch/root/ab/gpt/conf/... and die with FileNotFoundError.
#   train/screen needs cwd=$ROOT        -- LEMUR roots itself at CWD and the denoise loader
#               resolves train/ and validation/ relative to CWD.
# Each step below therefore sets its own cwd and PYTHONPATH rather than inheriting one.
GPT_ROOT=/a/mm/nn-gpt

# Generation 0 is the completed one-shot run: its stage-2 output is what generation 1 evolves
# from. Later generations add their own.
STAGE2_DIRS="/a/mm/sweep_out/llm_stage2"
RUN_DIRS="/a/mm/sweep_out/llm_gen2/cold/run /a/mm/sweep_out/llm_gen2/seeded/run"

echo "############ LLM EVOLVE  gens=$GENS N=$N K=$K gate=$GATE ep=$EPOCHS  $(date -Is) ############"

for g in $(seq 1 "$GENS"); do
  GD="$OUT/gen$g"
  mkdir -p "$GD"
  echo
  echo "######## GENERATION $g/$GENS  $(date -Is) ########"

  # ---- 1. elites -------------------------------------------------------------
  echo "--- [$g.1] elite archive ---"
  PYTHONPATH="$ROOT:$GPT_ROOT" python -u "$GA/elite_archive.py" \
      --stage2-dirs $STAGE2_DIRS \
      --run-dirs $RUN_DIRS \
      --k "$K" --floor-seeds "$FLOOR" --floor-seed-set "$FLOOR_SEED_SET" --cos-max 0.95 \
      --out "$GD/elites.json" || { echo "FATAL gen$g: elite archive failed"; exit 1; }

  # ---- 2. generate -----------------------------------------------------------
  echo "--- [$g.2] generate $N candidates seeded on elites ---"
  ( cd "$GPT_ROOT" && PYTHONPATH="$GPT_ROOT" python -u "$GA/llm_denoise.py" \
        --mode seeded --n "$N" --supporting "$SUPPORTING" \
        ${ADAPTER:+--adapter "$ADAPTER"} \
        ${DEPLOY_GUIDANCE:+--deploy-guidance "$DEPLOY_GUIDANCE"} \
        --seed-set "$SEED_SET" --elite-file "$GD/elites.json" \
        --out "$GD/run" ) || { echo "FATAL gen$g: generation failed"; exit 1; }

  # ---- 3. preflight (CPU) ----------------------------------------------------
  echo "--- [$g.3] preflight (CPU) ---"
  PRE_OK=1
  # DEPLOY_THR>0 adds the source-level eviction gate (deploy_gate.py) to preflight. Set per
  # arm by launch_llm_gate_ab.sh; empty/0 leaves the pipeline byte-identical to earlier runs.
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$ROOT:$GPT_ROOT" python -u "$GA/preflight.py" \
      --run-dirs "$GD/run" --out "$GD/pre" \
      ${DEPLOY_THR:+--deploy-predict-thr "$DEPLOY_THR"} || PRE_OK=0
  PASS="$GD/pre/pass_ids.txt"

  # A CRASHED PREFLIGHT IS NOT A BARREN ONE -- see run3_evolve.sh for the full note. The old
  # `|| echo WARN` + unconditional skip made a died-on-candidate-3 run look identical to one
  # that legitimately found nothing novel. CUDA_VISIBLE_DEVICES= is belt-and-braces here:
  # this variant runs preflight in-process on the login node rather than as a pod, so it would
  # otherwise inherit whatever GPU the login shell can see.
  if [ "$PRE_OK" -eq 0 ]; then
    echo "FATAL gen$g: preflight exited non-zero -- stopping rather than guessing"
    exit 1
  fi
  if [ ! -s "$PASS" ]; then
    echo "gen$g: preflight completed, nothing was novel -- skipping to next generation"
    continue
  fi
  echo "gen$g: $(wc -l < "$PASS") of $N candidates survived preflight"

  # ---- 4. screen -------------------------------------------------------------
  echo "--- [$g.4] 1-epoch screen ---"
  PYTHONPATH="$ROOT:$GPT_ROOT" python -u "$GA/screen_and_train.py" \
      --run-dirs "$GD/run" --only-ids "$PASS" \
      --out "$GD/eval" --gate "$GATE" --screen-only \
      || echo "WARN gen$g: screen returned non-zero"

  # ---- 5. train survivors into the DB ---------------------------------------
  echo "--- [$g.5] ${EPOCHS}-epoch train of survivors ---"
  PYTHONPATH="$ROOT:$GPT_ROOT" python -u "$GA/analyse_screen.py" --eval-dir "$GD/eval" --gen-dir "$GD" \
      --gate "$GATE" --write-survivors "$GD/survivors.txt" || true
  if [ ! -s "$GD/survivors.txt" ]; then
    echo "gen$g: no candidate cleared $GATE dB -- nothing trained this generation"
    continue
  fi
  # Prefix carries the generation so the DB stays separable forever; screen_and_train
  # appends -cold/-seeded on top, giving e.g. LLM-Eg2-seeded-<hash>.
  # Hyphens only: '_' is LEMUR's stat-dir field separator and an underscored model
  # name breaks the 4-way split in Util.py:61, aborting the whole batch import.
  PYTHONPATH="$ROOT:$GPT_ROOT" python -u "$GA/screen_and_train.py" \
      --run-dirs "$GD/run" --only-ids "$GD/survivors.txt" \
      --skip-screen --epochs "$EPOCHS" --gate "$GATE" \
      --prefix "LLM-Eg${g}" --out "$GD/stage2" \
      || echo "WARN gen$g: stage 2 returned non-zero"

  # this generation's trained models become selectable by the next one
  STAGE2_DIRS="$STAGE2_DIRS $GD/stage2"
  RUN_DIRS="$RUN_DIRS $GD/run"
  echo "--- gen $g done $(date -Is) ---"
done

echo
echo "############ LLM EVOLVE COMPLETE $(date -Is) ############"
python -u - <<'PY'
import sqlite3
import os
c=sqlite3.connect('file:'+os.path.join(os.getcwd(),'db','ab.nn.db')+'?mode=ro',uri=True)
# GLOB, not LIKE: in LIKE, '_' is a single-char wildcard, so the old 'LLM_%'
# pattern silently matched anything starting "LLM"+any char. Be explicit that we
# want both the current 'LLM-' spelling and the legacy 'LLM_' rows (470 of them).
rows=c.execute("SELECT nn,MAX(epoch),MAX(accuracy) FROM stat "
               "WHERE nn GLOB 'LLM[-_]*' GROUP BY nn").fetchall()
import re, collections
byg=collections.defaultdict(list)
for nn,e,a in rows:
    # Accept either separator so pre-rename generations still bucket correctly.
    m=re.match(r'LLM[-_]Eg(\d+)[-_]', nn)
    byg['gen'+m.group(1) if m else 'gen0'].append(a*48)
print(f"  {len(rows)} LLM models in DB")
for g in sorted(byg):
    v=byg[g]
    print(f"    {g:<6} n={len(v):<4} best={max(v):.3f} dB  mean={sum(v)/len(v):.3f} dB")
PY

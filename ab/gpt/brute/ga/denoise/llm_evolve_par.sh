#!/usr/bin/env bash
# PARALLEL LLM EVOLVE -- coordinator runs ON THE LOGIN NODE, not in a pod.
#
# WHY NOT A COORDINATOR POD (the obvious design, and the one that fails). A pod cannot launch
# the shard jobs: the namespace's default service account has no job-creation rights
# (`kubectl auth can-i create jobs --as=system:serviceaccount:b-b-ahmed:default` -> no) and
# the image ships no kubectl binary. Granting the right is not available either -- this user
# cannot create roles or rolebindings. So orchestration lives where a working kubectl
# already exists: the login node.
#
# The login node has NO TORCH, so it may only run steps that are pure stdlib. That splits the
# loop cleanly:
#     dispatched as Jobs : elite archive, generate, preflight, screen, train   (need torch)
#     run locally        : analyse_screen, merge_shard_dbs                     (stdlib only)
#
# Measured serial generation 1 (4 h 33 m) vs this at 4 shards:
#     generate 150   2 h 02 m -> ~31 m    GPU, zero DB writes
#     preflight          4 m ->    4 m    CPU, one pod
#     screen            52 m -> ~13 m     GPU, --screen-only, zero DB writes
#     train          2 h 55 m -> ~44 m    GPU, WRITES the DB -> private copies + merge
#
# The train stage is the whole reason for merge_shard_dbs.py: SQLite on NFSv3 does not block
# concurrent writers, it interleaves and corrupts them. Each shard writes a pod-local copy;
# one process folds them back afterwards, dry-run first, every generation.
#
# GENERATIONS STAY SEQUENTIAL by construction -- generation N+1 conditions on N's trained
# elites, which is the entire feedback mechanism of this arm (the LLM's weights are frozen;
# nothing here fine-tunes anything). Only work WITHIN a generation is parallel.
#
# Run detached, because it outlives any shell:
#   nohup bash llm_evolve_par.sh > /shared/ssd/.../evolve_par.log 2>&1 &
set -uo pipefail

HOST=/shared/ssd/home/b-b-ahmed/thesis      # login-node view
POD=/a/mm                                   # in-container view of the same tree
GA_H="$HOST/nn-gpt/ab/gpt/brute/ga/denoise"
GA="$POD/nn-gpt/ab/gpt/brute/ga/denoise"
SRC="$POD/nn-dataset"
NS="${NS:-b-b-ahmed}"

GENS="${GENS:-35}"
N="${N:-150}"
K="${K:-8}"
FLOOR="${FLOOR:-4}"
GATE="${GATE:-32.7}"
EPOCHS="${EPOCHS:-5}"
SUPPORTING="${SUPPORTING:-3}"
SHARDS="${SHARDS:-4}"
# Reference selection under --seed-set db. Tournament-of-3 is the GA's own operator: enough
# pressure to prefer good references, not so much that it collapses onto the single best one
# the way an argmax -- or a frozen k=8 elite list -- does. ELITE_SHARE is the fraction of
# prompts still anchored on a measured-best model as primary.
# DIVERSITY THRESHOLD. Recalibrated on 4,174 real lineage-labelled pairs (recalibrate_scale.py):
# 0.90 separates known-same from known-different better than 0.95 -- Youden J 0.945 vs 0.904,
# catching 97.0% of true duplicates against 90.4%, at the cost of wrongly rejecting 2.4% of
# genuinely distinct pairs. The original 0.95 was the midpoint of a gap the calibration called
# empty; on real generated models that gap holds 1,450 pairs, so 0.95 is no longer sitting in
# open space. Expect a LOWER yield: a tighter line rejects more, and the models that survive
# are pairwise further apart by construction.
COSMAX="${COSMAX:-0.90}"
TOURNAMENT="${TOURNAMENT:-3}"
ELITE_SHARE="${ELITE_SHARE:-0.25}"
TAG="${TAG:-llm_evolve_par}"
OUT_H="$HOST/sweep_out/$TAG"
OUT="$POD/sweep_out/$TAG"
# OVERRIDABLE, and it has to be. These defaulted to production unconditionally, so a smoke
# test that passed its own --main was silently ignored and would have merged test models into
# the real DB. It only escaped because the run was cut short before step 6.
MAINDB_H="${MAINDB_H:-$HOST/nn-dataset/db/ab.nn.db}"
MAINDB="${MAINDB:-$SRC/db/ab.nn.db}"

mkdir -p "$OUT_H"

# NOTE ON THE SNAPSHOT IN STEP 5: shards take their private DB with SQLite's online backup
# API, not `cp`. The GA writes the production DB continuously, and copying a live SQLite file
# with cp can capture a torn mid-transaction state. merge_shard_dbs.py would catch that (it
# integrity_checks every shard and refuses the whole merge), but a run that dies at the merge
# has already burnt an hour of GPU. backup() takes a consistent snapshot of a live DB.

# Every job needs the same pod-local LEMUR root: nn-dataset's ab/ and README.md (so the root
# resolves from CWD) plus the data symlinks. Built inline because each pod gets a fresh
# /scratch emptyDir.
ROOTSETUP="R=/scratch/root; mkdir -p \$R/out; \
for l in ab README.md data train validation; do ln -sfn $SRC/\$l \$R/\$l; done"

wait_job() {   # wait_job <name> <completions>
  local name="$1" want="$2" tick=0 ok fail
  while :; do
    ok=$(kubectl get job "$name" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
    fail=$(kubectl get job "$name" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
    if [ "${ok:-0}" -ge "$want" ] 2>/dev/null; then
      echo "    $name: $ok/$want ok"; return 0
    fi
    if [ "${fail:-0}" -gt 0 ] 2>/dev/null; then
      echo "    $name: $fail FAILED (ok=${ok:-0}/$want)"
      kubectl logs -n "$NS" "job/$name" --tail=25 2>&1 | sed 's/^/      /'
      return 1
    fi
    tick=$((tick+1))
    [ $((tick % 60)) -eq 0 ] && echo "      ... $name ${ok:-0}/$want  $((tick*15))s"
    sleep 15
  done
}

run_job() {   # run_job <name> <completions> <gpus> <command>
  local name="$1" comp="$2" gpus="$3" cmd="$4"
  kubectl delete job "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1
  # A deleted job's pods linger briefly; applying immediately can collide on the name.
  sleep 3
  python3 "$GA_H/mkjob.py" --name "$name" --completions "$comp" --gpus "$gpus" \
      --command "$cmd" --out "/tmp/$name.json" >/dev/null || return 1
  kubectl apply -f "/tmp/$name.json" -n "$NS" >/dev/null || return 1
  wait_job "$name" "$comp"
}

# Generation 0 = the completed one-shot run; generation 1 = the serial evolve run. Both are
# already in the DB and both feed selection here, so killing the serial loop costs only its
# in-flight generation, never a trained model.
STAGE2="$POD/sweep_out/llm_stage2 $POD/sweep_out/llm_evolve/gen1/stage2"
RUNS="$POD/sweep_out/llm_gen2/cold/run $POD/sweep_out/llm_gen2/seeded/run $POD/sweep_out/llm_evolve/gen1/run"

echo "######## PARALLEL EVOLVE  gens=$GENS N=$N shards=$SHARDS gate=$GATE ep=$EPOCHS  $(date -Is) ########"
echo "coordinator: $(hostname)   out: $OUT_H"

for g in $(seq 1 "$GENS"); do
  GD="$OUT/gen$g"; GD_H="$OUT_H/gen$g"; mkdir -p "$GD_H"
  echo; echo "######## GENERATION $g/$GENS  $(date -Is) ########"

  echo "--- [$g.1] elite archive ---"
  run_job "lp-eli-$g" 1 0 \
    "$ROOTSETUP; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/elite_archive.py \
     --stage2-dirs $STAGE2 --run-dirs $RUNS --k $K --floor-seeds $FLOOR --cos-max 0.95 \
     --out $GD/elites.json" \
    || { echo "FATAL gen$g: elite archive"; exit 1; }

  # THE GATE'S VERDICT, FED BACK INTO THE NEXT PROMPT. Until now the diversity gate refused
  # ~32 of 147 candidates a generation and told the generator nothing, so it proposed the same
  # shapes again -- 812 near-duplicate rejections across the first run's 16 generations with no
  # adaptation at all. The gate computed the cosine to the nearest accepted model every time
  # and discarded it. This turns that number into instruction.
  FB=""
  if [ "$g" -gt 1 ]; then
    PREV="$OUT/gen$((g-1))"; PREV_H="$OUT_H/gen$((g-1))"
    if [ -f "$PREV_H/pre/preflight.json" ]; then
      python3 "$GA_H/rejection_digest.py" --pre "$PREV_H/pre" --run "$PREV_H/run" \
        --examples 2 --cos-max "$COSMAX" --out "$GD_H/feedback.txt" 2>&1 | sed 's/^/    /' || true
      [ -s "$GD_H/feedback.txt" ] && FB="--feedback-file $GD/feedback.txt"
    fi
  fi

  echo "--- [$g.2] generate $N across $SHARDS GPUs (wide pool, tournament select) ---"
  # --seed-set db, NOT elite. The elite pool is --k 8 and its four hand-written anchors
  # (35.1-35.2 dB) were never displaced in 16 generations, because the best model the LLM
  # ever produced reached 33.90 dB. Combined with llm_denoise's old `corpus[i % len(corpus)]`
  # reference pick, that made the prompt a pure function of i mod a small number: 2400
  # prompts were issued and 140 were distinct, the most-repeated 48 times. 34% of candidates
  # were then rejected as near-duplicates and 16% as already-in-dataset -- 50% of the yield
  # lost to asking one model the same question over and over. Yield was 1.6 models/gen.
  #
  # db draws from every edge-legal, architecturally-distinct model in the master DB (~1045)
  # and selects by tournament-of-3 on within-cohort percentile. Same 3000-prompt simulation:
  # 3000 distinct combinations, 685 distinct primaries. The elite file stays as an ANCHOR --
  # --elite-share of prompts still use a measured-best model as primary, so the wide pool
  # buys diversity without discarding the selection pressure that made seeded beat cold.
  run_job "lp-gen-$g" "$SHARDS" 1 \
    "cd $POD/nn-gpt && PYTHONPATH=$POD/nn-gpt python -u $GA/llm_denoise.py --mode seeded \
     --n $N --supporting $SUPPORTING --seed-set db --elite-file $GD/elites.json \
     --tournament $TOURNAMENT --elite-share $ELITE_SHARE $FB \
     --out $GD/run --id-prefix g$g --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS" \
    || { echo "FATAL gen$g: generation"; exit 1; }

  echo "--- [$g.3] preflight (CPU, import + class repair, dead-code strip) ---"
  # --arch-dedupe rejects, on CPU before any GPU is spent, any candidate whose ARCHITECTURE
  # (WL graph hash) duplicates another in this batch or anything already in the dataset. The
  # GA has gated on this since day one; the LLM never did, which is why the worst duplicate
  # group in the corpus was six LLM models that are one architecture, differing only by dead
  # classes they never instantiated.
  #
  # --strip-dead-code REPLACES --reject-dead-code. Both collapse those decorated copies onto
  # their true architecture so the gate can catch them, and both leave the corpus clean for
  # fine-tuning -- but rejecting cost 235 of 2364 candidates (9.9%), including the ones whose
  # architecture was genuinely novel and merely carried an unused class. Stripping keeps them.
  #
  # Class repair (on by default, --no-class-repair to disable) splices back helper classes the
  # candidate calls but never defines, taken from the references in its OWN prompt. That was
  # the largest single failure mode at 304 candidates (12.9%): the missing names were
  # SPBlock / ELUAvgBlock / DilBlock / LPPoolBlock / HaarDown, i.e. the helpers of the four
  # reference models every prompt carried. Measured on generations 10-16: fires on 108
  # candidates and takes their unbound called names from 188 to 1.
  # CUDA_VISIBLE_DEVICES= FORCES THE GRADED CHECK ONTO THE CPU -- see the identical note in
  # run3_evolve.sh. gpus=0 only affects SCHEDULING; the graded path builds every candidate via
  # mutator/lemur_io.py:169 (`'cuda' if torch.cuda.is_available() else 'cpu'`), so on a GPU
  # node it grabs a card. A contended card makes the check throw OutOfMemoryError, which is
  # CAUGHT and degraded to "exact check still applies" -- i.e. the gate silently stops
  # detecting near-duplicates and this generation's diversity numbers become fiction.
  PRE_OK=1
  run_job "lp-pre-$g" 1 0 \
    "$ROOTSETUP; cd \$R && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/preflight.py \
     --run-dirs $GD/run --out $GD/pre --arch-dedupe --arch-graded --cos-max $COSMAX \
     --arch-dedupe-db $MAINDB --arch-store $OUT/diversity_store.pt \
     --strip-dead-code" \
    || PRE_OK=0
  PASS_H="$GD_H/pre/pass_ids.txt"

  # A CRASHED PREFLIGHT IS NOT A BARREN ONE. `|| echo WARN` followed by an unconditional
  # `continue` made the two indistinguishable: the generation was skipped either way, with no
  # signal that anything broke. Distinguish them -- a job that failed aborts the run, a job
  # that completed and found nothing novel skips this generation quietly.
  if [ "$PRE_OK" -eq 0 ]; then
    echo "FATAL gen$g: preflight job did not succeed ($(wc -l < "$PASS_H" 2>/dev/null || echo 0) ids written)"
    exit 1
  fi
  [ -s "$PASS_H" ] || { echo "gen$g: preflight completed, nothing was novel -- skipping"; continue; }
  echo "    $(wc -l < "$PASS_H") of $N survived preflight"

  echo "--- [$g.4] 1-epoch screen across $SHARDS GPUs ---"
  run_job "lp-scr-$g" "$SHARDS" 1 \
    "$ROOTSETUP; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/screen_and_train.py \
     --run-dirs $GD/run --only-ids $GD/pre/pass_ids.txt --out $GD/eval --gate $GATE \
     --screen-only --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS" \
    || echo "    WARN gen$g: screen had failures"

  # Local: stdlib only.
  python3 "$GA_H/analyse_screen.py" --eval-dir "$GD_H/eval" --gen-dir "$GD_H" \
      --gate "$GATE" --write-survivors "$GD_H/survivors.txt" 2>&1 | tail -4
  [ -s "$GD_H/survivors.txt" ] || { echo "gen$g: none cleared $GATE dB"; continue; }
  echo "    $(wc -l < "$GD_H/survivors.txt") cleared $GATE dB"

  echo "--- [$g.5] ${EPOCHS}-epoch train across $SHARDS GPUs (private DBs) ---"
  mkdir -p "$GD_H/shard_db"
  run_job "lp-trn-$g" "$SHARDS" 1 \
    "set -e; S=\$JOB_COMPLETION_INDEX; $ROOTSETUP; \
     mkdir -p \$R/db && python -c \"import sqlite3,sys; a=sqlite3.connect(sys.argv[1]); \
b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()\" $MAINDB \$R/db/ab.nn.db; \
     cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/screen_and_train.py \
       --run-dirs $GD/run --only-ids $GD/survivors.txt --skip-screen --epochs $EPOCHS \
       --gate $GATE --prefix LLM-Pg${g} --out $GD/stage2 \
       --shard \$S --num-shards $SHARDS; \
     cp \$R/db/ab.nn.db $GD/shard_db/shard\$S.db" \
    || echo "    WARN gen$g: train had failures; merging whatever completed"

  echo "--- [$g.6] merge shard DBs (single writer, dry-run first) ---"
  ls "$GD_H"/shard_db/shard*.db >/dev/null 2>&1 || { echo "    no shard DBs; skipping"; continue; }
  python3 "$GA_H/merge_shard_dbs.py" --main "$MAINDB_H" \
      --shards "$GD_H"/shard_db/shard*.db --dry-run 2>&1 | tail -6 || \
      { echo "FATAL gen$g: merge dry-run failed -- DB untouched"; exit 1; }
  python3 "$GA_H/merge_shard_dbs.py" --main "$MAINDB_H" \
      --backup-dir "$HOST/sweep_out/db_backup" \
      --shards "$GD_H"/shard_db/shard*.db 2>&1 | tail -6 || \
      { echo "FATAL gen$g: merge failed -- backup in sweep_out/db_backup"; exit 1; }
  # BACKFILL arch_uid FOR WHAT WAS JUST MERGED. Without this the dedup set is frozen at the
  # last full audit: the next generation would be checked against architectures known two
  # generations ago, and every duplicate of this generation's output would pass while the
  # summary still claimed to be deduping.
  echo "--- [$g.7] backfill arch_uid for newly merged models ---"
  run_job "lp-uid-$g" 1 0 \
    "$ROOTSETUP; ln -sfn $SRC/db \$R/db; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt \
     python -u $GA/backfill_arch_uid.py --db $MAINDB" \
    || echo "    WARN gen$g: arch_uid backfill failed; next generation dedupes on a stale set"

  # Shard DBs are 11 MB each and their content is now in the main DB.
  rm -f "$GD_H"/shard_db/shard*.db

  STAGE2="$STAGE2 $GD/stage2"
  RUNS="$RUNS $GD/run"
  echo "--- gen $g done $(date -Is) ---"
done

echo; echo "######## PARALLEL EVOLVE COMPLETE $(date -Is) ########"

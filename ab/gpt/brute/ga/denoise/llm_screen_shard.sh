#!/usr/bin/env bash
# STAGE 1 of the LLM evaluation: 1-epoch screen of every candidate from denoise-llmgen2,
# sharded across GPUs. SCREEN ONLY -- no DB writes, so N pods can run concurrently without
# putting N writers on the NFS-hosted SQLite DB (screen_and_train.py refuses the unsafe
# combination outright).
#
# WHY SCREEN EVERYTHING RATHER THAN GATE IN-FLIGHT. results_shard*.json records ep1 for every
# candidate that trains at all, pass or fail. So the gate is a label here, not a filter: once
# this pass is done the ep1 DISTRIBUTION of all 516 is on disk and the production gate can be
# chosen from it and re-applied for free. 32.7 is carried over as the default because it is
# the measured floor that retains all six canonical families finishing >= 35 dB -- but that
# was measured on 18 hand-written models, and LLM output has a far wider spread. Fixing the
# gate before seeing the distribution would be guessing.
#
# Candidates are strided (i %% num_shards), not blocked, because ids run in strategy
# round-robin order; contiguous blocks would give each shard a skewed strategy mix.
#
# ---------------------------------------------------------------------------------------
# POD-LOCAL LEMUR ROOT -- this is not optional, it is what makes sharding possible at all.
#
# The first sharded attempt died on:
#     FileNotFoundError: '/a/mm/nn-dataset/out/nn/tmp/__init__.py'
# check_nn writes each candidate into a SHARED scratch package, out/nn/tmp/, and imports it.
# Eight pods doing that against one NFS directory delete each other's module mid-import. It
# is the same shared-writer hazard as the DB, one directory over, and it is a RACE: it will
# not reproduce on one pod, so it cannot be caught by a single-GPU smoke test.
#
# LEMUR picks its root by walking up from CWD for a directory holding both `ab/` and
# `README.md` (ab/nn/util/Const.py:__project_root_path). So a private root is just a
# directory with those two names symlinked in -- code, db and data stay shared and
# read-only, while `out/` becomes real, local and per-pod. /scratch is an emptyDir: pod-local
# disk, never NFS.
# ---------------------------------------------------------------------------------------
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
GEN=/a/mm/sweep_out/llm_gen2
OUT=/a/mm/sweep_out/llm_eval2
SHARD="${JOB_COMPLETION_INDEX:-0}"
NSHARDS="${NSHARDS:-8}"
GATE="${GATE:-32.7}"
ROOT=/scratch/root
mkdir -p "$OUT" "$ROOT/out"

echo "=== LLM screen shard $SHARD/$NSHARDS  gate=$GATE  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

# Shared and read-only: the package, the DB, the datasets. Private and writable: out/.
for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
cd "$ROOT" || exit 1
echo "pod-local LEMUR root: $ROOT  (out/ is local, ab|db|data are symlinks into $SRC)"

# Sanity: if the root check fails LEMUR silently falls back to some ancestor directory and
# we are back to shared out/ without noticing.
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
[ -d "$ROOT/train" ] && [ -d "$ROOT/validation" ] || { echo "FATAL: data not visible"; exit 1; }

PYTHONPATH="$ROOT" python -u "$GA/screen_and_train.py" \
    --run-dirs "$GEN/cold/run" "$GEN/seeded/run" \
    --out "$OUT" --gate "$GATE" --screen-only \
    --shard "$SHARD" --num-shards "$NSHARDS"
rc=$?
echo "=== shard $SHARD done rc=$rc $(date -Is) ==="
exit $rc

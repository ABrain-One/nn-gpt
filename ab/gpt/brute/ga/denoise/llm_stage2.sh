#!/usr/bin/env bash
# STAGE 2: 5-epoch dataset train of the screen survivors, into the DB.
#
# SINGLE POD, and it must stay that way: this is the only step that writes the LEMUR SQLite
# DB, which lives on NFSv3 where concurrent writers corrupt rather than block.
# screen_and_train.py refuses --num-shards>1 without --screen-only for exactly this reason.
#
# --skip-screen: stage 1 already ran (denoise-llmscreen2, results_shard*.json) and
# --only-ids carries just its 26 survivors, so re-screening would burn ~35 min of GPU
# recomputing numbers already on disk.
#
# --epochs 5 matches the GA's epoch_max=5 exactly, so LLM and GA entries are comparable.
# The arm (_cold/_seeded) is baked into the DB name because both arms otherwise share the
# LLM_Denoise-* stem and the split is unrecoverable once the run dirs age out.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
GEN=/a/mm/sweep_out/llm_gen2
EVAL=/a/mm/sweep_out/llm_eval2
OUT=/a/mm/sweep_out/llm_stage2
ROOT=/scratch/root
mkdir -p "$OUT" "$ROOT/out"
# Private root so the out/nn/tmp scratch package is pod-local; db/ is a symlink to the real
# DB, which is what we DO want to write -- one writer, no race.
for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
cd "$ROOT" || exit 1
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
[ -d "$ROOT/train" ] && [ -d "$ROOT/validation" ] || { echo "FATAL: data not visible"; exit 1; }
[ -s "$EVAL/survivors_32.7.txt" ] || { echo "FATAL: survivor list missing/empty"; exit 1; }

echo "=== stage 2 | $(wc -l < "$EVAL/survivors_32.7.txt") survivors | 5 epochs | $(date -Is) ==="
PYTHONPATH="$ROOT" python -u "$GA/screen_and_train.py" \
    --run-dirs "$GEN/cold/run" "$GEN/seeded/run" \
    --only-ids "$EVAL/survivors_32.7.txt" \
    --skip-screen --epochs 5 --gate 32.7 \
    --prefix LLM-Denoise --out "$OUT"
rc=$?
echo "=== stage 2 done rc=$rc $(date -Is) ==="
python -u - <<'PY'
import sqlite3
c=sqlite3.connect('file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro',uri=True)
# GLOB 'LLM[-_]Denoise*' covers both the current 'LLM-Denoise' spelling and the
# legacy 'LLM_Denoise' rows. ('_' is a wildcard in LIKE, so the old pattern was
# matching by accident rather than by intent.)
r=c.execute("SELECT nn,MAX(epoch),MAX(accuracy) FROM stat WHERE nn GLOB 'LLM[-_]Denoise*' GROUP BY nn ORDER BY 3 DESC").fetchall()
print(f"  {len(r)} LLM-Denoise model(s) in DB")
for nn,e,a in r[:40]: print(f"    {nn:<52} ep_max={e:<3} {a*48:.3f} dB")
PY
exit $rc

#!/usr/bin/env bash
# CORRECTED 9F CONTROL (2026-07-30).
#
# In denoise-sweep-1 the DenoiseUnet9F control pod exited in 0.0 min with
# "All trials have already been passed ... epoch: 50" and re-reported the STORED
# 35.48 dB. Cause: the LEMUR config key is (task, dataset, metric, nn, epoch_max) --
# batch and lr are NOT in it. 9F already holds 50 rows at epoch_max=50 (from a
# batch=16 run), so a batch=2 re-run at 50 epochs looks identical to remaining_trials()
# and is skipped. LiteDenoiseNet was unaffected (it only had epoch_max 1 and 5).
#
# FIX: n_optuna_trials=-1 -- negative means "this many ADDITIONAL trials"
# (train.py:36), which bypasses the already-passed check.
#
# Without this the sweep has only one valid control and cannot separate
# "recipe effect" from "architecture effect" for the 9F baseline.
#
# Same NFS-safety design as sweep_recipe.sh: pod-local db, export for serial merge.
set -uo pipefail

NAME="${FAMILY:-DenoiseUnet9F}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-2}"
LR="${LR:-5.26e-5}"

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out
WORK="${SCRATCH:-/scratch}/lemur_ctrl"
LOG="$OUTD/sweep_${NAME}_b${BATCH}_e${EP}_forced.log"
mkdir -p "$OUTD"; : > "$LOG"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== FORCED control family=$NAME batch=$BATCH lr=$LR epochs=$EP (n_optuna_trials=-1) ==="
log "=== started $(date -Is) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train"      "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md -> ab_root_path would resolve wrong"; exit 1; }

cd "$WORK"
export PYTHONPATH="$WORK"
python -c "
import ab.nn.train as T
from ab.nn.util.Const import ab_root_path, db_file
assert str(ab_root_path).startswith('${WORK}'), f'ROOT ESCAPED TO {ab_root_path}'
print('confirmed pod-local db:', db_file, flush=True)
T.main(config='img-denoising_denoise_psnr_${NAME}',
       nn_prm={'lr':${LR},'batch':${BATCH},'transform':'denoise','num_workers':6},
       epoch_max=${EP}, n_optuna_trials=-1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && R=OK || R=FAIL
log "=== $R $NAME finished $(date -Is) ==="

cp "$WORK/db/ab.nn.db" "$OUTD/db_${NAME}_b${BATCH}_e${EP}_forced.db" 2>>"$LOG" \
  && log "exported db_${NAME}_b${BATCH}_e${EP}_forced.db" || log "WARN db export failed"

# isolate the NEW trial: compare against the master DB's stored row count for this model
python - "$NAME" "$WORK" <<'PYEOF' | tee -a "$LOG"
import sys, sqlite3
name, work = sys.argv[1], sys.argv[2]
def rows(p):
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True).cursor()
    c.execute("""SELECT s.id,s.epoch,s.accuracy,
                        (SELECT value FROM prm WHERE uid=s.prm AND name='batch')
                 FROM stat s WHERE s.task='img-denoising' AND s.nn=?""", (name,))
    return c.fetchall()
old = {r[0] for r in rows("/a/mm/nn-dataset/db/ab.nn.db")}
new = [r for r in rows(f"{work}/db/ab.nn.db") if r[0] not in old]
if new:
    best = max(r[2] for r in new)
    b = {r[3] for r in new}
    print(f"RESULT {name} NEW-TRIAL: rows={len(new)} batch={b} best_acc={best:.4f} PSNR={best*48:.2f} dB")
    print("  curve: " + "  ".join(f"e{e}:{a*48:.2f}" for _, e, a, _ in sorted(new, key=lambda r: r[1]) if e % 5 == 0 or e == 1))
else:
    print(f"RESULT {name}: NO NEW ROWS -- still skipped, investigate")
PYEOF
[ "$R" = OK ]

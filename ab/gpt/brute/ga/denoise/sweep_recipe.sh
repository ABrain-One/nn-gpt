#!/usr/bin/env bash
# RECIPE SWEEP (2026-07-30)
#
# WHY: the 5-epoch populate_db pass measured every new family at batch=16 / lr=1e-4 / 5
# epochs. On this dataset (658 train pairs) that is only 205 optimizer updates -- FEWER
# than the 329 updates LiteDenoiseNet used to reach its 36.51 dB row (batch=2,
# lr=5.26e-5, ONE epoch, on CPU). The families were not outclassed, they were undertrained.
# This sweep re-runs them under LiteDenoiseNet's OWN recipe at epoch_max=50
# -> ~16450 updates, ~50x the original budget.
#
# CONTROLS ARE MANDATORY: LiteDenoiseNet and DenoiseUnet9F run the same recipe, so every
# number in the result table is like-for-like. Comparing a fresh 50-epoch run against the
# stored 5-epoch rows would repeat exactly the mistake this sweep exists to correct.
#
# DB ISOLATION (critical): /shared/ssd is NFSv3 and db/ab.nn.db is the project's
# source of truth. Concurrent SQLite writers across nodes over NFS risk corrupting it.
# So each pod rsyncs the LEMUR package to pod-local /scratch (~235 MB), trains against
# its OWN db copy, and exports only its result DB to sweep_out/. The shared master DB is
# opened READ-ONLY here and merged separately, serially, on the submit node.
#
# epoch_max is part of the LEMUR config key (train.py:69, sub_config + (epoch_max,)), so
# these 50-epoch runs are a distinct config from the stored 5-epoch rows and are NOT
# skipped by remaining_trials(). No n_optuna_trials=-1 needed.
set -uo pipefail

FAMILIES=(
  DenoiseMBConv        # 33.94 @5ep - best of the new set
  DenoiseLPPool        # 33.94
  DenoiseELUCtx        # 33.88
  DenoiseSplitterNetF  # 33.88
  DenoiseSplitterNet   # 33.81 - the CVPR2024 architecture
  DenoiseThunder       # 33.60 - the SIDD-lightweight architecture
  LiteDenoiseNet       # CONTROL: its 36.51 came from this exact recipe
  DenoiseUnet9F        # CONTROL: 35.48 @ep45 but at batch=16
)

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NAME="${FAMILIES[$IDX]}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-2}"
LR="${LR:-5.26e-5}"

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out
WORK="${SCRATCH:-/scratch}/lemur_${IDX}"
LOG="$OUTD/sweep_${NAME}_b${BATCH}_e${EP}.log"
mkdir -p "$OUTD"; : > "$LOG"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== sweep idx=$IDX family=$NAME batch=$BATCH lr=$LR epochs=$EP ==="
log "=== started $(date -Is) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG"

# --- pod-local LEMUR tree: package + db only; heavy data stays on NFS via symlink ---
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train"      "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md -> ab_root_path would resolve wrong"; exit 1; }
log "workdir $WORK ($(du -sh "$WORK" 2>/dev/null | cut -f1)), db $(ls -la "$WORK/db/ab.nn.db" | awk '{print $5}') bytes"

cd "$WORK"
export PYTHONPATH="$WORK"
python -c "
import ab.nn.train as T
from ab.nn.util.Const import ab_root_path, db_file
assert str(ab_root_path).startswith('${WORK}'), f'ROOT ESCAPED TO {ab_root_path}'
print('confirmed pod-local db:', db_file, flush=True)
T.main(config='img-denoising_denoise_psnr_${NAME}',
       nn_prm={'lr':${LR},'batch':${BATCH},'transform':'denoise','num_workers':6},
       epoch_max=${EP}, n_optuna_trials=1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && R=OK || R=FAIL
log "=== $R $NAME finished $(date -Is) ==="

# export this pod's DB for the serial merge on the submit node
cp "$WORK/db/ab.nn.db" "$OUTD/db_${NAME}_b${BATCH}_e${EP}.db" 2>>"$LOG" \
  && log "exported $OUTD/db_${NAME}_b${BATCH}_e${EP}.db" || log "WARN db export failed"

python - "$NAME" "$WORK" <<'PYEOF' | tee -a "$LOG"
import sys, sqlite3
name, work = sys.argv[1], sys.argv[2]
con = sqlite3.connect(f"file:{work}/db/ab.nn.db?mode=ro", uri=True); cur = con.cursor()
cur.execute("SELECT epoch,accuracy FROM stat WHERE task='img-denoising' AND nn=? ORDER BY epoch", (name,))
rows = cur.fetchall()
if rows:
    best = max(a for _, a in rows)
    print(f"RESULT {name}: rows={len(rows)} best_acc={best:.4f} PSNR={best*48:.2f} dB")
    print("  curve: " + "  ".join(f"e{e}:{a*48:.2f}" for e, a in rows if e % 5 == 0 or e == 1))
else:
    print(f"RESULT {name}: NO ROW")
PYEOF
[ "$R" = OK ]

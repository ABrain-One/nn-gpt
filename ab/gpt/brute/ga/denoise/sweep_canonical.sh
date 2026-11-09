#!/usr/bin/env bash
# CANONICAL-RECIPE 50-EPOCH RUN (2026-07-30)
#
# RECIPE: lr=1e-4, batch=16, epoch_max=50 -- verified from the DB as the recipe used for
# ALL SIX DenoiseUnet*F seeds (uniform: lr 0.0001 / batch 16 / epoch_max 50). This is the
# project standard and the only setting under which a new family can be compared to them.
#
# Supersedes the batch=2 / lr=5.26e-5 sweep (archived in sweep_out/wrong_recipe_b2/).
# That sweep was built around LiteDenoiseNet's 36.51 dB row, which is the ONE seed row
# that does NOT follow the standard: batch=2, epoch_max=1, gpu_type='No GPU' (CPU), two
# rows total. It is an outlier, not the baseline, and must not anchor a comparison.
#
# NO 9F POD: DenoiseUnet9F already holds a full 50-row curve at exactly this recipe
# (35.48 dB @ep45). Re-running it would only re-measure the control. Its stored curve IS
# the reference. LiteDenoiseNet IS included -- it has no canonical-recipe number at all,
# so its 50-epoch/batch-16 result is genuinely new information.
#
# Skip-safety: the new families hold only epoch_max=5 rows and LiteDenoiseNet only
# epoch_max 1 and 5. The config key is (task,dataset,metric,nn,epoch_max), so epoch_max=50
# is a fresh key for all 18 -> remaining_trials() will not skip them. (This is the trap the
# earlier 9F control pod fell into; it does not apply here.)
#
# NFS safety: /shared/ssd is NFSv3 and db/ab.nn.db is the source of truth. Each pod trains
# against a pod-local DB copy on /scratch and exports it for a serial merge on the submit
# node. No pod writes the shared DB.
set -uo pipefail

FAMILIES=(
  DenoiseMBConv DenoiseLPPool DenoiseELUCtx DenoiseSplitterNetF DenoiseSplitterNet
  DenoiseChanShuffle DenoiseLinearMix DenoiseThunder DenoiseMFDNet DenoiseSubPixel
  DenoisePoolDeconv DenoiseSelfAttn DenoiseParallelMR DenoiseLapPyr DenoiseDilatedFlat
  DenoiseReflectPad DenoiseGroupNorm
  LiteDenoiseNet
)

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NAME="${FAMILIES[$IDX]}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out/canonical
WORK="${SCRATCH:-/scratch}/lemur_${IDX}"
LOG="$OUTD/${NAME}.log"
mkdir -p "$OUTD"; : > "$LOG"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== CANONICAL idx=$IDX family=$NAME batch=$BATCH lr=$LR epochs=$EP ==="
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
       epoch_max=${EP}, n_optuna_trials=1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && R=OK || R=FAIL
log "=== $R $NAME finished $(date -Is) ==="

cp "$WORK/db/ab.nn.db" "$OUTD/db_${NAME}.db" 2>>"$LOG" \
  && log "exported db_${NAME}.db" || log "WARN db export failed"

# full per-epoch statistics for the NEW trial only (diffed against the master DB by stat.id)
python - "$NAME" "$WORK" <<'PYEOF' | tee -a "$LOG"
import sys, sqlite3, statistics as st
name, work = sys.argv[1], sys.argv[2]
def rows(p):
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True).cursor()
    c.execute("SELECT id,epoch,accuracy FROM stat WHERE task='img-denoising' AND nn=?", (name,))
    return c.fetchall()
old = {r[0] for r in rows("/a/mm/nn-dataset/db/ab.nn.db")}
new = sorted([r for r in rows(f"{work}/db/ab.nn.db") if r[0] not in old], key=lambda r: r[1])
if not new:
    print(f"STATS {name}: NO NEW ROWS -- run was skipped, investigate"); raise SystemExit
db = [a*48 for _, _, a in new]
eps = [e for _, e, _ in new]
bi = max(range(len(db)), key=lambda i: db[i])
print(f"STATS {name}: n_epochs={len(db)} max_epoch={max(eps)} "
      f"best_dB={db[bi]:.3f} best_epoch={eps[bi]} final_dB={db[-1]:.3f} "
      f"avg_dB={st.mean(db):.3f} std={st.pstdev(db):.3f} min_dB={min(db):.3f}")
print("CURVE " + name + " " + " ".join(f"{e}:{d:.3f}" for e, d in zip(eps, db)))
PYEOF
[ "$R" = OK ]

#!/usr/bin/env bash
# VALIDATION-NOISE TEST (2026-07-30)
#
# QUESTION. The canonical 50-epoch sweep put all 18 families inside 0.59 dB
# (34.66-35.25) and the ep5->ep50 rank correlation collapsed to +0.26. Two very
# different explanations fit that equally well:
#   (a) the architectures genuinely are near-identical on this data, or
#   (b) we cannot MEASURE a difference, because validation is 50 images x ONE centre
#       crop = 50 samples, ~3.3 megapixels total.
# Under (b) a 0.1 dB gap is sampling noise and the GA has been ranking ~2000 candidates
# on a measurement that cannot resolve them. Distinguishing the two decides whether the
# next move is better data or better measurement.
#
# METHOD. Identical training to the canonical sweep (DENOISE_TRAIN_CROPS=1, lr 1e-4,
# batch 16, 50 epochs) but evaluated on DENOISE_VAL_CROPS=8 -- a deterministic 8-cell
# grid per validation image, 400 crops instead of 50. Training is untouched, so any
# change in spread or in epoch-to-epoch jitter is a property of the MEASUREMENT.
#
# READ IT AS: if the between-model spread holds while the within-model jitter (last-10
# std) drops, the differences are real. If models reorder, the canonical ranking was noise.
#
# COMPARABILITY: val K>1 changes the metric. These numbers are comparable to EACH OTHER
# and to the 9F control in this run -- NOT to the 1,535 stored rows. That is why 9F is
# retrained here rather than quoting its stored 35.43.
#
# n_optuna_trials=-1 throughout: negative = ADDITIONAL trial, which sidesteps the
# "All trials have already been passed" skip (the config key ignores batch/lr AND the
# loader env, so a val-crop change alone looks identical to a completed run).
set -uo pipefail

FAMILIES=(
  DenoiseSubPixel DenoiseDilatedFlat DenoiseLPPool DenoiseELUCtx
  DenoiseMBConv DenoisePoolDeconv DenoiseLapPyr DenoiseUnet9F
)
IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NAME="${FAMILIES[$IDX]}"
EP="${EPOCHS:-50}"

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out/valcrops
WORK="${SCRATCH:-/scratch}/lemur_${IDX}"
LOG="$OUTD/${NAME}.log"
mkdir -p "$OUTD"; : > "$LOG"
log(){ echo "$@" | tee -a "$LOG"; }

log "=== VALCROPS idx=$IDX $NAME train_crops=${DENOISE_TRAIN_CROPS:-1} val_crops=${DENOISE_VAL_CROPS:-8} ep=$EP ==="
log "=== started $(date -Is) on $(hostname) ==="

mkdir -p "$WORK"
# EXCLUDE 'data' (added after this job was evicted once): nn-dataset/data/midd is an
# 81 GB MIDD download owned by other work in this tree. Copying it overflowed the 20 GiB
# /scratch emptyDir and K8s evicted all 8 pods ~90 s in, with exit 137 and a 137-byte log
# that showed nothing. The earlier canonical sweep survived only because data/ did not
# exist yet -- the exclude list was always too narrow, it just had not been tested.
#
# Guard rather than trust the list: measure the payload BEFORE copying and abort with a
# clear message if it cannot fit, so the next oversized dir fails loudly instead of as a
# silent eviction.
RS_EXCL=(--exclude='train' --exclude='validation' --exclude='.git'
         --exclude='out' --exclude='results' --exclude='data')
SZ_KB=$(rsync -an --stats "${RS_EXCL[@]}" "$SRC/" "$WORK/" 2>/dev/null \
        | awk -F': *' '/Total file size/{gsub(/[^0-9]/,"",$2); print int($2/1024)}')
log "rsync payload: $(( ${SZ_KB:-0} / 1024 )) MB"
if [ "${SZ_KB:-0}" -gt 10485760 ]; then     # >10 GiB: will not fit /scratch
  log "FATAL payload $(( SZ_KB / 1024 )) MB exceeds scratch budget -- widen the excludes"
  exit 1
fi
rsync -a "${RS_EXCL[@]}" "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"

cd "$WORK"
export PYTHONPATH="$WORK"
python -c "
import ab.nn.train as T
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('${WORK}'), f'ROOT ESCAPED TO {ab_root_path}'
T.main(config='img-denoising_denoise_psnr_${NAME}',
       nn_prm={'lr':1e-4,'batch':16,'transform':'denoise','num_workers':6},
       epoch_max=${EP}, n_optuna_trials=-1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && R=OK || R=FAIL
log "=== $R $NAME finished $(date -Is) ==="

cp "$WORK/db/ab.nn.db" "$OUTD/db_${NAME}.db" 2>>"$LOG" && log "exported db_${NAME}.db"

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
    print(f"STATS {name}: NO NEW ROWS -- skipped, investigate"); raise SystemExit
v = [a * 48 for _, _, a in new]
# last10 std is the key number: within-model jitter = how noisy the MEASUREMENT is.
print(f"STATS {name}: n={len(v)} best={max(v):.3f} last5={st.mean(v[-5:]):.3f} "
      f"last10={st.mean(v[-10:]):.3f} last10_std={st.pstdev(v[-10:]):.4f} "
      f"spike={max(v)-st.mean(v[-5:]):.3f}")
print("CURVE " + name + " " + " ".join(f"{e}:{a*48:.3f}" for _, e, a in new))
PYEOF
[ "$R" = OK ]

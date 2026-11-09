#!/usr/bin/env bash
# RETRAIN DenoiseUNet / DenoiseUNet2 UNDER THE CANONICAL RECIPE.
#
# WHY. The full LEMUR DB (nn-gpt/db/ab.nn.db) records these two at 40.54 dB on the SAME
# 'denoise' dataset, at epochs 7-8 -- 5.1 dB above anything our sweeps have measured, and
# reached in a fraction of the epochs. Both models exist as source in ab/nn/nn but have NO
# rows in our working DB, so they have never been trained under our protocol.
#
# Two explanations fit, and they have opposite consequences:
#   (a) the architectures really are that much better -> our seed pool has been ignoring
#       the best models we own, and every threshold set at 34-35 dB is set too low;
#   (b) those 40.54 dB rows came from a different loader/crop/eval protocol -> the number
#       is not comparable, and our ~35 dB ceiling stands.
# One 50-epoch run each settles it. Same recipe as the 18-family canonical sweep
# (lr 1e-4, batch 16, 50 epochs) so the result drops straight into that table.
#
# n_optuna_trials=-1: negative = ADDITIONAL trial. These have no rows in the working DB so
# a skip is unlikely, but the config key is (task,dataset,metric,nn,epoch_max) and ignores
# batch/lr -- the same trap that silently no-op'd the 9F control earlier. Cheap insurance.
set -uo pipefail

MODELS=(DenoiseUNet DenoiseUNet2)
IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NAME="${MODELS[$IDX]}"
EP="${EPOCHS:-50}"

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out/unet_orig
WORK="${SCRATCH:-/scratch}/lemur_${IDX}"
LOG="$OUTD/${NAME}.log"
mkdir -p "$OUTD"; : > "$LOG"
log(){ echo "$@" | tee -a "$LOG"; }

log "=== UNET-ORIG idx=$IDX $NAME ep=$EP lr=1e-4 batch=16 ==="
log "=== started $(date -Is) on $(hostname) ==="

mkdir -p "$WORK"
RS_EXCL=(--exclude='train' --exclude='validation' --exclude='.git'
         --exclude='out' --exclude='results' --exclude='data')
SZ_KB=$(rsync -an --stats "${RS_EXCL[@]}" "$SRC/" "$WORK/" 2>/dev/null \
        | awk -F': *' '/Total file size/{gsub(/[^0-9]/,"",$2); print int($2/1024)}')
log "rsync payload: $(( ${SZ_KB:-0} / 1024 )) MB"
if [ "${SZ_KB:-0}" -gt 10485760 ]; then
  log "FATAL payload $(( SZ_KB / 1024 )) MB exceeds scratch budget"; exit 1; fi
rsync -a "${RS_EXCL[@]}" "$SRC/" "$WORK/" >>"$LOG" 2>&1 || { log "FATAL rsync"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"

cd "$WORK"
export PYTHONPATH="$WORK"
python -u -c "
import ab.nn.train as T
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('${WORK}'), f'ROOT ESCAPED TO {ab_root_path}'
T.main(config='img-denoising_denoise_psnr_${NAME}',
       nn_prm={'lr':1e-4,'batch':16,'transform':'denoise','num_workers':6},
       epoch_max=${EP}, n_optuna_trials=-1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && R=OK || R=FAIL
log "=== $R $NAME finished $(date -Is) ==="

cp "$WORK/db/ab.nn.db" "$OUTD/db_${NAME}.db" 2>>"$LOG" && log "exported db_${NAME}.db"

python -u - "$NAME" "$WORK" <<'PYEOF' | tee -a "$LOG"
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
print(f"STATS {name}: n={len(v)} ep1={v[0]:.3f} best={max(v):.3f} "
      f"last5={st.mean(v[-5:]):.3f} last10={st.mean(v[-10:]):.3f}")
print("  vs full-LEMUR claim 40.54 dB -> "
      + ("REPRODUCED" if max(v) >= 39.0 else
         f"NOT reproduced (short by {40.54 - max(v):.2f} dB) -> protocol difference"))
print("CURVE " + name + " " + " ".join(f"{e}:{a*48:.3f}" for _, e, a in new))
PYEOF
[ "$R" = OK ]

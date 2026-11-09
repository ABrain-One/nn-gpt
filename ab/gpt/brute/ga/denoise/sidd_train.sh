#!/usr/bin/env bash
# Train our top architectures ON SIDD-Medium, so their SIDD numbers are comparable to published work.
#
# WHY TRAIN FROM SCRATCH ON SIDD RATHER THAN FINE-TUNE FROM THE AWGN CHECKPOINT. Published SIDD
# sRGB results are produced by training on SIDD-Medium. Warm-starting from a synthetic-noise
# checkpoint would be a different, non-standard recipe, and the whole point of this run is
# comparability. Training on SIDD is also what removes the domain gap that made the transfer
# numbers useless: AWGN-trained models scored 23.7-24.1 dB on the official validation blocks
# against a 22.7 dB no-op, i.e. barely better than returning the input.
#
# TRAIN/TEST SEPARATION, which is the thing that would invalidate everything. Training uses
# loader `sidd` = SIDD-Medium (160 scene instances). Evaluation uses loader `sidd_bench` = the
# official ValidationNoisyBlocksSrgb.mat blocks, drawn from the OTHER 40 scene instances. Disjoint
# by construction of the distribution. Our SIDD-Medium pool spans 158 scene ids, consistent with
# that. If anyone ever adds SIDD-Full or the validation scenes to the training pool, every number
# from this run becomes worthless -- see ab/nn/loader/sidd_bench.py for the full argument.
#
# METRIC. psnrrgb, not psnr. Published SIDD sRGB PSNR is computed on RGB; our default `psnr`
# metric is not the same quantity, and mixing them would make the comparison wrong in a way that
# is invisible in the final table.
#
# THE SIDD LOADER NEEDS A SYMLINK. sidd.py hardcodes SIDD_ROOT = cwd/sidd, and the repo's own
# nn-dataset/sidd is 320 DANGLING symlinks pointing at /home/s493053 (a user path that no longer
# exists). The real data is sweep_out/sidd_subset -- 731 pairs at 512x512. Same fix
# confirm_train.sh:60 uses.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-8}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
METRIC="${METRIC:-psnrrgb}"
MODELS_FILE="${MODELS_FILE:-/a/mm/sweep_out/sidd_bench/models_top20.txt}"
TOPN="${TOPN:-10}"

SRC=/a/mm/nn-dataset
OUTD="${OUTD:-/a/mm/sweep_out/sidd_train}"
WORK="${SCRATCH:-/scratch}/sidd_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
mkdir -p "$OUTD"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== SIDD TRAIN shard=$IDX/$NSHARDS ep=$EP metric=$METRIC $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
# the real SIDD-Medium pairs, NOT the repo's broken symlink tree
ln -sfn /a/mm/sweep_out/sidd_subset "$WORK/sidd"
[ -d "$WORK/sidd/original" ] || { log "FATAL sidd_subset/original missing"; exit 1; }
n_pairs=$(ls "$WORK/sidd/original" | wc -l)
log "SIDD-Medium training pairs visible: $n_pairs"
[ "$n_pairs" -gt 100 ] || { log "FATAL only $n_pairs pairs -- refusing to train"; exit 1; }
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }

cd "$WORK"; export PYTHONPATH="$WORK"

mapfile -t MODELS < <(head -n "$TOPN" "$MODELS_FILE")
log "candidate models: ${#MODELS[@]} (top $TOPN of $MODELS_FILE)"

done_keys=""
[ -f "$RES" ] && done_keys=$(python3 -c "
import json
ks=set()
for l in open('$RES'):
    if l.strip():
        try:
            r=json.loads(l)
            if r.get('status')=='OK': ks.add(r['model'])
        except Exception: pass
print(' '.join(sorted(ks)))
" 2>/dev/null)

i=-1
for M in "${MODELS[@]}"; do
  i=$((i+1))
  [ $((i % NSHARDS)) -eq "$IDX" ] || continue
  case " $done_keys " in *" $M "*) log "skip $M (done)"; continue;; esac
  log "--- $M ---"
  python - "$M" "$EP" "$BATCH" "$LR" "$RES" "$METRIC" <<'PYEOF' >> "$LOG" 2>&1
import json, os, sqlite3, sys, time
model, ep, batch, lr, res, metric = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                     float(sys.argv[4]), sys.argv[5], sys.argv[6])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

rec = {"model": model, "dataset": "sidd-medium", "metric": metric, "epochs": ep, "status": "FAIL"}
t0 = time.time()
try:
    con = sqlite3.connect(os.path.join(str(ab_root_path), 'db', 'ab.nn.db'))
    row = con.execute("select code from nn where name=?", (model,)).fetchone()
    con.close()
    if not row:
        raise RuntimeError("not in DB")
    prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
    import ab.nn.api as api
    # prefix keeps SIDD-trained weights in their own checkpoint namespace, so the AWGN-trained
    # checkpoint for the same architecture is not overwritten -- both are needed, they answer
    # different questions.
    _nm, acc, _, _ = api.check_nn(row[0], 'img-denoising', 'sidd', metric, prm,
                                  save_to_db=True, prefix='sidd', save_pth_weights=True)
    rec.update(status="OK", val_psnr=round(acc * 48, 4), acc=acc)
except Exception as e:
    import traceback
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["trace"] = traceback.format_exc()[-300:]
rec["seconds"] = round(time.time() - t0, 1)
with open(res, "a") as f:
    f.write(json.dumps(rec) + "\n")
print("  ->", json.dumps({k: v for k, v in rec.items() if k != 'trace'}))
PYEOF
done

# copy SIDD-trained checkpoints out of /scratch before the pod releases it
mkdir -p "$OUTD/ckpt"
python3 - "$OUTD/ckpt" <<'PYEOF' >> "$LOG" 2>&1
import os, shutil, sys
dst = sys.argv[1]
from ab.nn.util.Const import ab_root_path
root = os.path.join(str(ab_root_path), 'out', 'ckpt')
n = 0
if os.path.isdir(root):
    for d in os.listdir(root):
        s = os.path.join(root, d, 'best_model.pth')
        if os.path.isfile(s):
            os.makedirs(os.path.join(dst, d), exist_ok=True)
            shutil.copy2(s, os.path.join(dst, d, 'best_model.pth'))
            n += 1
print(f"  copied {n} checkpoint(s) to {dst}")
PYEOF

n_ok=$(grep -c '"status": "OK"' "$RES" 2>/dev/null || echo 0)
log "=== shard $IDX done: $n_ok OK $(date -Is) ==="

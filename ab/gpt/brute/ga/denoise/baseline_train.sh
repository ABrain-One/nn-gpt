#!/usr/bin/env bash
# Train the edge-legal HAND-WRITTEN baselines that were never run to convergence.
#
# WHY THIS IS THE DECISIVE JOB. The paper's central claim is that architecture search beats hand
# design under the 9,053 MMac edge budget. Measured, all at 50 epochs:
#
#   best searched                 DnSeedGA-99c89300   37.169 dB
#   best converged hand-written   DenoiseHalfInstNorm 35.644 dB   (2,926 MMac)
#     -> +1.525 dB for search
#
# That gap holds ONLY if no edge-legal hand-written model does better. Two were never trained
# past a single epoch, and both posted suspiciously high 1-epoch numbers with wildly inconsistent
# replicates:
#
#   DenoiseUnet5     1,463 MMac   37.436 dB @ 1 epoch   (other row: 33.080)
#   LiteDenoiseNet   3,530 MMac   36.511 dB @ 1 epoch   (other row: 33.884)
#
# 37.436 would BEAT the best searched model and invert the paper's claim. A 4.3 dB spread between
# two 1-epoch runs of the same model means neither number can be trusted, so this settles it with
# 50 epochs and repeats. REPEATS defaults to 3 because a single run cannot distinguish a real
# result from the same luck that produced 37.436 in the first place -- exactly the winner's-curse
# error the cross-dataset study already made once (1.90 dB that repeats reduced to 0.27 dB).
#
# LiteDenoiseNet is retrained separately by the reexport-anchor job under the re-export recipe;
# it is included here too because that job trains it for the full-resolution protocol while this
# one measures it on the same 256-crop protocol as every other baseline in the table. Different
# protocols, both needed, and mixing them in one table is what made the anchor unusable before.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-1}"
EP="${EPOCHS:-50}"
REPEATS="${REPEATS:-3}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
MODELS="${MODELS:-DenoiseUnet5 LiteDenoiseNet}"

SRC=/a/mm/nn-dataset
OUTD="${OUTD:-/a/mm/sweep_out/baselines}"
WORK="${SCRATCH:-/scratch}/bl_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
mkdir -p "$OUTD"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== BASELINE TRAIN shard=$IDX/$NSHARDS ep=$EP repeats=$REPEATS $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

# Same private-root pattern the other trainers use: rsync the repo into /scratch and symlink the
# heavy data, so the pod trains against its own DB copy and never writes the shared one.
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

# resume: skip (model, repeat) pairs already recorded
done_keys=""
[ -f "$RES" ] && done_keys=$(python3 -c "
import json,sys
ks=set()
for l in open('$RES'):
    if l.strip():
        try:
            r=json.loads(l)
            if r.get('status')=='OK': ks.add(f\"{r['model']}#{r['repeat']}\")
        except Exception: pass
print(' '.join(sorted(ks)))
" 2>/dev/null)
log "already done: $(echo "$done_keys" | wc -w) run(s)"

i=-1
for M in $MODELS; do
  for R in $(seq 0 $((REPEATS-1))); do
    i=$((i+1))
    [ $((i % NSHARDS)) -eq "$IDX" ] || continue
    case " $done_keys " in *" $M#$R "*) log "skip $M repeat $R (done)"; continue;; esac
    log "--- $M repeat $R ---"
    python - "$M" "$R" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1
import json, os, sqlite3, sys, time
model, rep, ep, batch, lr, res = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                  int(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

rec = {"model": model, "repeat": rep, "epochs": ep, "status": "FAIL"}
t0 = time.time()
try:
    con = sqlite3.connect(os.path.join(str(ab_root_path), 'db', 'ab.nn.db'))
    row = con.execute("select code from nn where name=?", (model,)).fetchone()
    con.close()
    if not row:
        raise RuntimeError(f"{model} not in DB")
    prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
    import ab.nn.api as api
    # save_to_db=True gives the per-epoch curve; the DB here is the pod's private /scratch copy
    # (asserted above), so nothing touches the shared database.
    _nm, acc, _, _ = api.check_nn(row[0], 'img-denoising', 'denoise', 'psnr', prm,
                                  save_to_db=True, prefix=f'bl{rep}', save_pth_weights=True)
    rec.update(status="OK", psnr_dB=round(acc * 48, 4), acc=acc)
except Exception as e:
    import traceback
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["trace"] = traceback.format_exc()[-300:]
rec["seconds"] = round(time.time() - t0, 1)
with open(res, "a") as f:
    f.write(json.dumps(rec) + "\n")
print("  ->", json.dumps({k: v for k, v in rec.items() if k != "trace"}))
PYEOF
  done
done

n_ok=$(grep -c '"status": "OK"' "$RES" 2>/dev/null || echo 0)
log "=== shard $IDX done: $n_ok OK row(s) in $RES $(date -Is) ==="

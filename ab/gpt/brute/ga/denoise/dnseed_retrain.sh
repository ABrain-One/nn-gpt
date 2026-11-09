#!/usr/bin/env bash
# Re-train the 376 DnSeedGA models for exactly 5 epochs, persisting ALL FIVE epoch stat files.
#
# WHY. DnSeedGA is the strongest arm we have (best AWGN PSNR 36.98, best zero-shot SIDD transfer
# 32.59 dB vs 27.31 for LLM, lowest latency 1.66 ms p50 with 3% off-NPU), but the DB kept only ONE
# row per model, at epoch 5 -- zero of 376 have intermediate epochs, because the GA persisted only
# the final result. Every other DenoiseGA stat dir in the repo holds a complete 1..5 curve, so
# exporting from the DB would publish 376 dirs that are incomplete relative to all 1,927 others.
# The curve cannot be reconstructed; it has to be re-measured. 5 epochs, matching the original.
#
# HOW THE FIVE FILES GET WRITTEN. Train.py:426 writes `join(save_path, f"{epoch}.json")` once per
# epoch, and Train.py:307 defaults save_path to stat_train_dir/<task>_<dataset>_<metric>_<model>.
# We pass save_path EXPLICITLY, named after the original model, for one specific reason: check_nn
# derives a model name from prefix+hash(code), and that hash is NOT recoverable as a plain md5 of
# the stored source (verified: 0/350 matched on the LLM names). Letting it auto-name would produce
# stat dirs under a NEW name that no longer matches the 376 models we are documenting. Passing
# save_path removes the guess entirely.
#
# save_to_db=True is required for the per-epoch JSON to be emitted at all, and is safe because the
# pod trains against its own /scratch DB copy (asserted below) which is never merged back.
#
# The existing single-file stat dirs are NOT deleted here. This writes to a staging directory; the
# swap into ab/nn/stat/train is a separate, reviewable step.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-8}"
EP="${EPOCHS:-5}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
TASK=img-denoising
DATASET=denoise
METRIC=psnr

SRC=/a/mm/nn-dataset
OUTD="${OUTD:-/a/mm/sweep_out/dnseed_retrain}"
STAGE="$OUTD/stat_train"
WORK="${SCRATCH:-/scratch}/dnsr_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
mkdir -p "$OUTD" "$STAGE"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== DNSEED RETRAIN shard=$IDX/$NSHARDS ep=$EP $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

# model list straight from the DB, deterministic order so shard assignment is stable
mapfile -t MODELS < <(python3 -c "
import sqlite3
con=sqlite3.connect('file:$WORK/db/ab.nn.db?mode=ro',uri=True)
for (n,) in con.execute(\"select name from nn where name like 'DnSeedGA-%' order by name\"):
    print(n)
")
log "DnSeedGA models: ${#MODELS[@]}"

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
log "already done: $(echo "$done_keys" | wc -w)"

i=-1
for M in "${MODELS[@]}"; do
  i=$((i+1))
  [ $((i % NSHARDS)) -eq "$IDX" ] || continue
  case " $done_keys " in *" $M "*) continue;; esac
  D="$STAGE/${TASK}_${DATASET}_${METRIC}_${M}"
  # a completed dir with all EP files means this model is finished even if the jsonl was lost
  if [ -d "$D" ] && [ "$(ls "$D"/*.json 2>/dev/null | wc -l)" -eq "$EP" ]; then
    log "skip $M (all $EP epoch files present)"; continue
  fi
  mkdir -p "$D"
  python - "$M" "$EP" "$BATCH" "$LR" "$RES" "$D" "$TASK" "$DATASET" "$METRIC" <<'PYEOF' >> "$LOG" 2>&1
import json, os, sqlite3, sys, time
model, ep, batch, lr, res, spath, task, dataset, metric = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
    sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

rec = {"model": model, "epochs": ep, "status": "FAIL"}
t0 = time.time()
try:
    con = sqlite3.connect(os.path.join(str(ab_root_path), 'db', 'ab.nn.db'))
    row = con.execute("select code from nn where name=?", (model,)).fetchone()
    con.close()
    if not row:
        raise RuntimeError("not in DB")
    prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
    import ab.nn.api as api
    _nm, acc, _, _ = api.check_nn(row[0], task, dataset, metric, prm,
                                  save_to_db=True, save_path=spath)
    n = len([f for f in os.listdir(spath) if f.endswith('.json')])
    rec.update(status="OK", psnr_dB=round(acc * 48, 4), epoch_files=n)
    if n != ep:
        rec["status"] = "PARTIAL"
        rec["warn"] = f"{n} epoch file(s), expected {ep}"
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

ok=$(grep -c '"status": "OK"' "$RES" 2>/dev/null || echo 0)
log "=== shard $IDX done: $ok OK $(date -Is) ==="

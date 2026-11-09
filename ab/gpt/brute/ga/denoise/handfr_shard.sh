#!/usr/bin/env bash
# RETRAIN THE HAND-DESIGNED ARM AT 50 EPOCHS, KEEPING THE WEIGHTS.
#
# WHY THIS EXISTS. The full-resolution Pareto comparison is unusable because evaluation
# coverage ranges 0-100% by producer arm, and the hand-designed arm is the worst at 31%:
# 28 of its models carry a screening PSNR and an on-device latency but no full-res score.
# Three of them (DenoiseScaleRec, DenoiseMoE, DenoiseUnfold) sit ON the screening front, so
# the gap is not cosmetic -- it is exactly the models that decide the comparison.
#
# THEY CANNOT SIMPLY BE EVALUATED: all 28 have source but NONE has a checkpoint. That is not
# bad luck. fulltrain50_shard.sh routes a bare (hash-free) name through the Optuna entry
# point T.main(), which does NOT pass save_pth_weights -- and ab/nn/util/Const.py sets
# default_save_pth_weights = False. So every hand-written model ever trained in this pipeline
# produced curves and DB rows and no weights, which is why ckpt/ holds 4,187 directories and
# not one of them is a bare name. This script passes save_pth_weights=True on that branch.
# That one keyword is the entire point of the file.
#
# PER-EPOCH SNAPSHOTS ARE ON. save_if_best picks best_model.pth by `current_score`, which here
# is 256x256 crop PSNR -- the metric documented in that same function as disagreeing with the
# full-res protocol IN ORDER, not merely by an offset. Selecting the epoch by crop and then
# reporting full-res would rebuild the bias this run exists to remove, so LEMUR_SAVE_EVERY_EPOCH
# keeps every epoch and the full-res pass chooses among them.
#
# RECIPE: lr=1e-4, batch=16, epoch=50, transform=denoise, num_workers=6 -- identical to
# sweep_canonical.sh and fulltrain50_shard.sh. Comparability is the whole point; do not tune it.
#
# THIS IS AN ADDENDUM. The paper's frozen corpus is not revised. These rows are new, carry the
# arm label hand50_fr, and must stay separately labelled.
#
# NO POD WRITES THE SHARED DB: pod-local copy on /scratch, exported for a serial merge.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-2}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
LIMIT="${LIMIT:-0}"                  # smoke test: train at most N models, 0 = all

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SUBSET="${SUBSET:-$GA/handfr_subset.json}"
OUTD="${OUTD:-/a/mm/sweep_out/handfr}"
WORK="${SCRATCH:-/scratch}/handfr_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
mkdir -p "$OUTD"

# Weights must land on the shared volume, not in the pod. This is the default in mkjob.py but
# is restated because a pod-local value here is indistinguishable from success until the pod
# exits and takes 50 epochs of weights with it.
export LEMUR_CKPT_DIR="${LEMUR_CKPT_DIR:-/a/mm/sweep_out/handfr/ckpt}"
export LEMUR_SAVE_EVERY_EPOCH=1
mkdir -p "$LEMUR_CKPT_DIR"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== HANDFR shard=$IDX/$NSHARDS ep=$EP batch=$BATCH lr=$LR limit=$LIMIT $(date -Is) ==="
log "    LEMUR_CKPT_DIR=$LEMUR_CKPT_DIR  LEMUR_SAVE_EVERY_EPOCH=$LEMUR_SAVE_EVERY_EPOCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mapfile -t MINE < <(python3 -c "
import json
m=json.load(open('$SUBSET'))['models']
m=m[$IDX::$NSHARDS]
print('\n'.join(m[:$LIMIT] if $LIMIT else m))")
log "assigned ${#MINE[@]} models: ${MINE[*]}"
[ "${#MINE[@]}" -gt 0 ] || { log "FATAL no models assigned"; exit 1; }

# Pod-local repo. data/ excluded (the MIDD tree is far larger than /scratch); big read-only
# trees symlinked. rsync 23/24 are "source files vanished" warnings caused by other shards'
# _temp_model_*.py and are NOT failures -- treating 24 as fatal once killed five healthy shards.
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='_temp_model_*' --exclude='__pycache__' \
         "$SRC/" "$WORK/" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ] && [ $rc -ne 23 ] && [ $rc -ne 24 ]; then
  log "FATAL rsync failed (exit $rc)"; exit 1
fi
[ $rc -ne 0 ] && log "rsync exit $rc (vanished source files) -- continuing"
ln -sfn "$SRC/train"      "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data"       "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md -> ab_root_path resolves wrong"; exit 1; }

cd "$WORK" || exit 1
export PYTHONPATH="$WORK"

done_names=""
[ -f "$RES" ] && done_names=$(python3 -c "
import json
print(' '.join(json.loads(l)['model'] for l in open('$RES') if l.strip()))" 2>/dev/null)

n_ok=0; n_fail=0; n_skip=0
for NAME in "${MINE[@]}"; do
  case " $done_names " in *" $NAME "*) n_skip=$((n_skip+1)); continue;; esac
  python - "$NAME" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1 && R=OK || R=FAIL
import os, sys, json, sqlite3, time, glob
name, ep, batch, lr, res = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                            float(sys.argv[4]), sys.argv[5])
from ab.nn.util.Const import ab_root_path, ckpt_dir
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

def stat_ids():
    c = sqlite3.connect('file:db/ab.nn.db?mode=ro', uri=True)
    return {r[0] for r in c.execute(
        "select id from stat where task='img-denoising' and nn=?", (name,))}

before = stat_ids()
rec = {"model": name, "status": "FAIL", "arm": "hand50_fr", "budget_epochs": ep}
t0 = time.time()
try:
    # These are all bare names with supported_hyperparameters(), so the Optuna entry point is
    # the right one -- but save_pth_weights defaults to False and that is the bug this run
    # exists to fix.
    #
    # n_optuna_trials MUST BE NEGATIVE. remaining_trials() computes
    # max(0, n_optuna_trials - n_passed_trials) and counts trials ALREADY RECORDED for this
    # model, so +1 yields zero remaining for every model here -- 15 of the 28 already carry
    # 50-epoch trials and the other 13 carry 5-epoch ones. A smoke test at +1 exited in 2
    # seconds with "2 recorded trial(s), 0 remaining" and trained nothing. A NEGATIVE value is
    # documented as "this many ADDITIONAL trials", which is what a deliberate retrain wants.
    # nn_prm pins the recipe, so the extra trial is a fixed-config rerun, not a search.
    import ab.nn.train as T
    T.main(config=f'img-denoising_denoise_psnr_{name}',
           nn_prm={'lr': lr, 'batch': batch, 'transform': 'denoise', 'num_workers': 6},
           epoch_max=ep, n_optuna_trials=-1, transform='denoise', num_workers=6,
           save_pth_weights=True)
    rec["status"] = "OK"
except Exception as e:
    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
rec["seconds"] = round(time.time() - t0)

# WEIGHTS ARE THE DELIVERABLE, so their absence is a FAILURE and not a footnote. A run that
# trains cleanly and saves nothing is the exact outcome this script was written to stop.
cdir = os.path.join(str(ckpt_dir), name)
best = os.path.join(cdir, 'best_model.pth')
snaps = sorted(glob.glob(os.path.join(cdir, 'epoch_*.pth')))
rec["ckpt_dir"] = cdir
rec["has_best_pth"] = os.path.isfile(best)
rec["n_epoch_snapshots"] = len(snaps)
if rec["has_best_pth"]:
    rec["best_pth_MB"] = round(os.path.getsize(best) / 1e6, 2)
elif rec["status"] == "OK":
    rec["status"] = "NO_WEIGHTS"

# The Optuna branch DOES call DB_Write.save_results per epoch (unlike the check_nn branch),
# so the trajectory is in the DB here rather than in per-epoch json files.
c = sqlite3.connect('file:db/ab.nn.db?mode=ro', uri=True)
new = sorted((e, a) for i, e, a in c.execute(
    "select id,epoch,accuracy from stat where task='img-denoising' and nn=?", (name,))
    if i not in before)
if new:
    curve = [[e, round(a * 48, 3)] for e, a in new]
    vals = [v for _, v in curve]
    bi = max(range(len(curve)), key=lambda k: curve[k][1])
    rec.update(n_db_rows=len(new), n_epochs=len(curve), curve=curve,
               best_dB=curve[bi][1], best_epoch=curve[bi][0], final_dB=curve[-1][1],
               avg_dB=round(sum(vals) / len(vals), 3),
               avg_dB_last10=round(sum(vals[-10:]) / len(vals[-10:]), 3))
elif rec["status"] == "OK":
    rec["status"] = "NO_ROWS"

open(res, "a").write(json.dumps(rec) + "\n")
print(f"RESULT {name} {rec['status']} best={rec.get('best_dB')} "
      f"pth={rec.get('has_best_pth')} snaps={rec.get('n_epoch_snapshots')} {rec['seconds']}s")
PYEOF
  if [ "$R" = OK ] && tail -1 "$RES" | grep -q '"status": "OK"'; then
    n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1))
    log "[shard $IDX] $NAME did not pass: $(tail -1 "$RES" | python3 -c 'import sys,json;r=json.loads(sys.stdin.read());print(r.get("status"),r.get("error",""))' 2>/dev/null)"
  fi
  log "[shard $IDX] $((n_ok+n_fail))/${#MINE[@]}  ok $n_ok  fail $n_fail  (skipped $n_skip)"
  cp "$WORK/db/ab.nn.db" "$OUTD/db_shard${IDX}.db" 2>>"$LOG"
done

log "=== shard $IDX DONE ok $n_ok fail $n_fail skipped $n_skip $(date -Is) ==="
cp "$WORK/db/ab.nn.db" "$OUTD/db_shard${IDX}.db" 2>>"$LOG" \
  && log "exported db_shard${IDX}.db" || log "WARN db export failed"
[ "$n_fail" -lt "${#MINE[@]}" ]

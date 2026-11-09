#!/usr/bin/env bash
# 50-EPOCH RUN over the BEAT-THE-BAR candidates (beatbar_subset.json).
#
# COPY of fulltrain50_shard.sh rather than an edit: ft50 pods are executing that file
# right now, and rewriting a script mid-execution is how a running sweep starts behaving
# as neither the old nor the new version. Divergence here is the richer summary block.
#
# WHY: the corpus is a 5-epoch corpus (2,824 of 2,958 models stop at epoch 5) and the
# 5-epoch ranking recovers only 3 of the true top-10 at convergence (Spearman 0.68, n=56).
# Cross-dataset transfer, the skip ablation, and any leaderboard all inherit that. This run
# produces the converged ground truth those claims need.
#
# RECIPE: lr=1e-4, batch=16, epoch_max=50, transform=denoise, num_workers=6 -- identical to
# sweep_canonical.sh, which is the only recipe under which a new model is comparable to the
# DenoiseUnet*F seeds. Do not "improve" it here; comparability is the whole point.
#
# SHARDING: pod i trains models[i::NSHARDS] SEQUENTIALLY. One rsync per pod rather than one
# per model (300 rsyncs of the repo would dominate the wall clock).
#
# NFS SAFETY: /shared/ssd is NFSv3 and db/ab.nn.db is the source of truth. Each pod trains
# against a pod-local DB copy on /scratch and exports it for a serial merge on the submit
# node. NO POD WRITES THE SHARED DB. Merge afterwards with merge_shard_dbs.py, dry-run
# first, at most 8 shards per call.
#
# RESUMABLE: every finished model is appended to the shard's result JSONL, and a re-run
# skips whatever is already recorded. A preempted pod loses at most one model's training.
#
# SKIP-SAFETY: the config key is (task,dataset,metric,nn,epoch_max). Every model in the
# subset holds only epoch_max<=5 rows -- select_fulltrain_subset.py excludes the 41 that
# already have an epoch_max=50 trial -- so epoch_max=50 is a fresh key and
# remaining_trials() will not silently skip them. This is the trap the 9F control pod hit.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-12}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SUBSET="${SUBSET:-$GA/fulltrain_subset.json}"
OUTD="${OUTD:-/a/mm/sweep_out/fulltrain50}"   # overridden by the smoke test so its
                                              # result file cannot make a real shard skip
WORK="${SCRATCH:-/scratch}/lemur_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
export CURVE_DIR="$OUTD/curves"
mkdir -p "$OUTD" "$CURVE_DIR"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== FULLTRAIN50 shard=$IDX/$NSHARDS epochs=$EP batch=$BATCH lr=$LR $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mapfile -t MINE < <(python3 -c "
import json,sys
m=json.load(open('$SUBSET'))['models']
print('\n'.join(m[$IDX::$NSHARDS]))")
log "assigned ${#MINE[@]} models"

# pod-local repo + DB; the big read-only trees stay SYMLINKED, never copied.
#
# data/ MUST BE EXCLUDED. sweep_canonical.sh's exclude list predates the MIDD download and
# omits it; data/midd is now a 23-sensor tree far larger than the 20Gi /scratch emptyDir,
# so copying it evicts the pod mid-rsync ("Usage of EmptyDir volume scratch exceeds the
# limit 20Gi", exit 137) with an empty log. The repo minus these is 271 MB.
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train"      "$WORK/train"
ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data"       "$WORK/data"      # MIDD/SIDD reachable without copying
du -sm "$WORK" 2>/dev/null | awk '{print "  pod-local tree: "$1" MB"}' | tee -a "$LOG"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md -> ab_root_path would resolve wrong"; exit 1; }

cd "$WORK"
export PYTHONPATH="$WORK"

done_names=""
[ -f "$RES" ] && done_names=$(python3 -c "
import json,sys
print(' '.join(json.loads(l)['model'] for l in open('$RES') if l.strip()))" 2>/dev/null)

n_ok=0; n_fail=0; n_skip=0
for NAME in "${MINE[@]}"; do
  case " $done_names " in *" $NAME "*) n_skip=$((n_skip+1)); continue;; esac
  t0=$(date +%s)
  # WHY check_nn AND NOT ab.nn.train.main(config=...).
  #
  # The Optuna entry point calls each model's supported_hyperparameters() to sample the
  # search space. GA- and LLM-generated models DO NOT DEFINE IT -- they were always trained
  # through check_nn with an explicit prm -- so main() resolves the function to None and
  # dies with "'NoneType' object is not callable" two seconds in, having written nothing.
  #
  # check_nn also takes the SOURCE rather than a registered module, which fixes the second
  # problem: 80 of the 300 subset models exist only as DB rows and have no ab/nn/nn/*.py.
  #
  # NAME PRESERVATION: train_new computes model_name = prefix + "-" + uuid4(code), and
  # uuid4 is just md5 of the whitespace-stripped source. Verified across the subset: 299 of
  # 300 names reproduce exactly under this rule, so passing the existing prefix writes the
  # 50-epoch rows under the ORIGINAL name instead of minting a new one. The exception is
  # the hand-written DenoiseSplitterNet (no hash suffix), which keeps the Optuna path.
  python - "$NAME" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1 && R=OK || R=FAIL
import os, sys, json, sqlite3, time
name, ep, batch, lr, res = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                            float(sys.argv[4]), sys.argv[5])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

# PER-EPOCH CURVES REQUIRE save_path -- WITHOUT IT THEY ARE SILENTLY DISCARDED.
# Train.train_n_eval writes one <epoch>.json per epoch inside its training loop, but the
# check_nn (is_code=True) branch only does so `if self.save_path`; otherwise it prints
# "[WARN] parameter `save_Path` set to null" and keeps nothing. The DB gets a single
# final-epoch row either way -- the per-epoch record exists ONLY as these files.
# Each file carries accuracy, train_loss, test_loss, train_accuracy, gradient_norm,
# samples_per_second, best_accuracy and best_epoch, so the curves also give convergence
# shape and overfitting onset, not just the endpoint.
CURVES = os.environ["CURVE_DIR"]
save_dir = os.path.join(CURVES, name)
os.makedirs(save_dir, exist_ok=True)

# NEW ROWS ARE IDENTIFIED BY DIFFING stat.id, NOT BY FILTERING ON epoch.
# check_nn is not consistent about what it records: across the corpus ~800 models carry a
# full per-epoch curve while ~2,000 carry only a single final-epoch row. An epoch>5 filter
# therefore reports NO_ROWS for a run that trained perfectly well (and silently hides the
# single-row case). Snapshot the ids first and take whatever is new.
def stat_ids():
    c = sqlite3.connect('file:db/ab.nn.db?mode=ro', uri=True)
    return {r[0] for r in c.execute(
        "select id from stat where task='img-denoising' and nn=?", (name,))}

before = stat_ids()
prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
prefix, _, _h = name.rpartition('-')
rec = {"model": name, "status": "FAIL"}
t0 = time.time()
try:
    if prefix:
        code = sqlite3.connect('db/ab.nn.db').execute(
            "select code from nn where name=?", (name,)).fetchone()[0]
        import ab.nn.api as api
        nm, acc, _, _ = api.check_nn(code, 'img-denoising', 'denoise', 'psnr', prm,
                                     save_to_db=True, prefix=prefix, save_path=save_dir)
        assert nm == name, f'NAME DRIFT: wrote {nm}, expected {name}'
        print(f'TRAINED {nm} {acc * 48:.3f} dB')
    else:                          # hand-written model: has supported_hyperparameters()
        import ab.nn.train as T
        T.main(config=f'img-denoising_denoise_psnr_{name}',
               nn_prm={'lr': lr, 'batch': batch, 'transform': 'denoise', 'num_workers': 6},
               epoch_max=ep, n_optuna_trials=1, transform='denoise', num_workers=6)
    rec["status"] = "OK"
except Exception as e:
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
rec["seconds"] = round(time.time() - t0)

c = sqlite3.connect('file:db/ab.nn.db?mode=ro', uri=True)
new = [(e, a) for i, e, a in c.execute(
    "select id,epoch,accuracy from stat where task='img-denoising' and nn=?", (name,))
    if i not in before]
if new:
    new.sort()
    rec.update(n_db_rows=len(new), db_final_dB=round(new[-1][1] * 48, 3))
elif rec["status"] == "OK":
    rec["status"] = "NO_ROWS"      # trained without recording -> must not count as success

# THE CURVE COMES FROM THE PER-EPOCH FILES, NOT THE DB. The is_code branch of the training
# loop never calls DB_Write.save_results, so the database holds exactly one row (the final
# epoch) no matter how long the run was; the trajectory lives only in save_dir.
curve, extra, per_epoch = [], {}, {}
for f in os.listdir(save_dir) if os.path.isdir(save_dir) else []:
    if not f.endswith('.json'):
        continue
    try:
        e = int(f[:-5])
    except ValueError:
        continue                    # summary files, not an epoch
    try:
        trials = json.load(open(os.path.join(save_dir, f)))
    except Exception:
        continue
    if not isinstance(trials, list) or not trials:
        continue
    t = max(trials, key=lambda d: d.get('accuracy') or 0)   # one trial per model here
    curve.append([e, round((t.get('accuracy') or 0) * 48, 3)])
    if e == ep:                     # carry the final epoch's richer diagnostics
        extra = {k: t.get(k) for k in ('train_loss', 'test_loss', 'train_accuracy',
                                       'gradient_norm', 'best_epoch', 'samples_per_second')
                 if t.get(k) is not None}
if curve:
    curve.sort()
    bi = max(range(len(curve)), key=lambda k: curve[k][1])
    # `extra` carries the trainer's OWN best_epoch, which would collide with the one
    # derived from the curve here (dict.update() raises on the duplicate keyword). They
    # are different measurements -- keep both, namespaced.
    vals = [d for _, d in curve]
    tail = vals[-10:] if len(vals) >= 10 else vals
    rec.update(n_epochs=len(curve), max_epoch=curve[-1][0], curve=curve,
               max_dB=curve[bi][1], best_dB=curve[bi][1], best_epoch=curve[bi][0],
               final_dB=curve[-1][1],
               avg_dB=round(sum(vals) / len(vals), 3),
               avg_last10_dB=round(sum(tail) / len(tail), 3),
               std_dB=round((sum((v - sum(vals)/len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5, 3),
               min_dB=min(vals), curve_dir=save_dir, final_metrics=extra)
    # one file per model holding EVERY epoch's full metric dict, alongside the summary
    with open(os.path.join(save_dir, "all_epochs.json"), "w") as fh:
        json.dump({"model": name, "epochs": {str(k): v for k, v in sorted(per_epoch.items())},
                   "summary": {k: rec[k] for k in
                               ("n_epochs", "max_dB", "best_epoch", "final_dB", "avg_dB",
                                "avg_last10_dB", "std_dB", "min_dB")}}, fh, indent=1)
    # peak well before the end == the fixed-budget endpoint understates the model
    if curve[bi][0] < 0.8 * ep:
        rec["peaked_early"] = True
elif rec["status"] == "OK":
    rec["status"] = "NO_CURVE"     # save_path silently ignored -> investigate, do not pass
open(res, "a").write(json.dumps(rec) + "\n")
print(f"RESULT {name} {rec['status']} best={rec.get('best_dB')} rows={rec.get('n_rows')} "
      f"{rec['seconds']}s")
PYEOF
  dt=$(( $(date +%s) - t0 ))


  # A python exit code of 0 is NOT proof of training: the trainer catches its own errors and
  # prints them, which is how the first smoke test reported "ok 1 fail 0" for a model that
  # produced nothing. Trust the recorded rows instead.
  if [ "$R" = OK ] && tail -1 "$RES" | grep -q '"status": "OK"'; then
    n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1))
  fi
  log "[shard $IDX] $((n_ok+n_fail))/${#MINE[@]}  ok $n_ok  fail $n_fail  (skipped $n_skip)"
  # export the pod-local DB as we go, so a preemption still leaves a mergeable artifact
  cp "$WORK/db/ab.nn.db" "$OUTD/db_shard${IDX}.db" 2>>"$LOG"
done

log "=== shard $IDX DONE ok $n_ok fail $n_fail skipped $n_skip $(date -Is) ==="
cp "$WORK/db/ab.nn.db" "$OUTD/db_shard${IDX}.db" 2>>"$LOG" \
  && log "exported db_shard${IDX}.db" || log "WARN db export failed"
[ "$n_fail" -lt "${#MINE[@]}" ]

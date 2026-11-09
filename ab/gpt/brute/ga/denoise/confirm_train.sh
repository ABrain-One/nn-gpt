#!/usr/bin/env bash
# Cross-dataset CONFIRMATION run: 8 models x 3 datasets x 3 repeats = 72 runs.
#
# PURPOSE. crossdata_train.sh measured every (model, dataset) cell EXACTLY ONCE, so its
# headline number -- "selecting on AWGN costs 1.90 dB on MIDD" -- has no error bar, and the
# whole 1.90 dB rests on a single draw of a single model (LLM_Pg33_seeded-8a7b5df scored 34.56
# on MIDD while AWGN's #2 and #3 picks scored 36.14 / 36.15). Three repeats per cell turns that
# into a measurement with a spread, and answers whether 8a7b5df is genuinely bad on real noise
# or simply had a bad run.
#
# PROTOCOL IS IDENTICAL TO crossdata_train.sh ON PURPOSE. 30 epochs, psnrrgb, the same 658-pair
# matched budget, the same midd_subset symlink tree and the same sidd_subset. Any deviation
# would make these numbers incomparable with the 40-model run they are meant to refine, and the
# comparison is the entire point. crossdata_train.sh is NOT edited or sourced -- it was
# executing in six pods when this was written, and bash re-seeks a running script by byte
# offset, so editing one mid-run makes the interpreter resume mid-line.
#
# WHY REPEATS ACTUALLY DIFFER. Nothing seeds the training: no torch.manual_seed / random.seed
# in Train.py, api.py or Util.py, and Util.py:179 builds the train loader with shuffle=True and
# no generator, so weight init and batch order differ run to run. The TEST DATA is fixed --
# denoise_awgn.py:98 seeds the added noise per image index -- so a repeat varies training and
# holds the evaluation constant. That is the variance we want; measured floor is 0.14 dB.
#
# RESUME IS GLOBAL, NOT PER-SHARD. done_keys is read from EVERY result_shard*.jsonl, not just
# this shard's. That is what makes re-sharding safe: this run starts on the 2 GPUs free tonight
# and can be relaunched with more shards once ft50 releases 5 more, without re-running anything
# or double-counting. ablation_train.sh deliberately does the opposite (whole bases pinned to a
# shard so within-base contrasts stay on one card) -- here every cell is compared ACROSS
# repeats of itself, and repeats are averaged, so a per-GPU offset lands in the mean of every
# arm alike rather than in a within-arm difference.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-2}"
EP="${EPOCHS:-30}"
REPS="${REPEATS:-3}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"

SRC=/a/mm/nn-dataset
CF="${CF:-/a/mm/sweep_out/confirm}"
WORK="${SCRATCH:-/scratch}/cf_${IDX}"
LOG="$CF/shard${IDX}.log"
RES="$CF/result_shard${IDX}.jsonl"
export CURVE_DIR="$CF/curves"
mkdir -p "$CF" "$CURVE_DIR"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== CONFIRM shard=$IDX/$NSHARDS ep=$EP reps=$REPS $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
# sidd.py hardcodes SIDD_ROOT = cwd/sidd. The repo's own nn-dataset/sidd is 320 BROKEN symlinks
# to a pre-migration /home/s493053 path and is excluded from the rsync so it cannot shadow this.
ln -sfn /a/mm/sweep_out/sidd_subset "$WORK/sidd"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

# The other session owns nn-dataset/data/midd. midd_subset is a tree of RELATIVE symlinks into
# it and nothing here writes through them; midd.py is not edited.
export MIDD_ROOT=/a/mm/sweep_out/midd_subset
export MIDD_PATCH_DIR=.          # the subset IS the patch tree (sensors sit at its top level)

mapfile -t JOBS < <(python3 - "$CF/subset.json" "$IDX" "$NSHARDS" "$REPS" <<'PY'
import json, sys
sub, idx, ns, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
models = json.load(open(sub))["models"]
# Interleave (model, dataset, repeat) triples. Sharding on a flat index over this ordering keeps
# each shard's mix of datasets balanced: 'denoise' trains fastest and 'midd' slowest, and the
# 40-model run partitioned so badly by dataset that two shards finished 20/20 while others sat
# at 4/20. Repeat varies fastest, so a shard that dies mid-run leaves whole cells complete.
work = [(m["model"], m["awgn_dB"], ds, r)
        for m in models for ds in ("denoise", "midd", "sidd") for r in range(reps)]
for i, w in enumerate(work):
    if i % ns == idx:
        print("\t".join(str(x) for x in w))
PY
)
log "assigned ${#JOBS[@]} (model,dataset,repeat) runs"

# GLOBAL resume: every shard's results, so re-sharding cannot re-run or double-count a cell.
done_keys=$(python3 - "$CF" <<'PY' 2>/dev/null
import glob, json, os, sys
ks = []
for f in glob.glob(os.path.join(sys.argv[1], "result_shard*.jsonl")):
    for line in open(f):
        if line.strip():
            try:
                ks.append(json.loads(line)["key"])
            except Exception:
                pass
print(" ".join(ks))
PY
)
log "already complete across all shards: $(echo $done_keys | wc -w)"

n_ok=0; n_fail=0; n_skip=0
for J in "${JOBS[@]}"; do
  IFS=$'\t' read -r MODEL AWGN DS REP <<< "$J"
  KEY="${MODEL}__${DS}__r${REP}"
  case " $done_keys " in *" $KEY "*) n_skip=$((n_skip+1)); continue;; esac

  python - "$MODEL" "$AWGN" "$DS" "$KEY" "$REP" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1
import os, sys, json, sqlite3, time
model, awgn, ds, key, rep, ep, batch, lr, res = (
    sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]),
    int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), sys.argv[9])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

con = sqlite3.connect('file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro', uri=True)
row = con.execute("select code from nn where name=?", (model,)).fetchone()
rec = {"key": key, "model": model, "dataset": ds, "repeat": rep, "awgn_dB": awgn,
       "status": "FAIL"}
if not row:
    rec["error"] = "model not in DB"
    open(res, "a").write(json.dumps(rec) + "\n")
    print(f"RESULT {key} NO_CODE"); sys.exit(0)

save_dir = os.path.join(os.environ["CURVE_DIR"], key)
os.makedirs(save_dir, exist_ok=True)
prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
t0 = time.time()
try:
    import ab.nn.api as api
    # save_to_db=True is REQUIRED for per-epoch JSON: Train.train_n_eval writes <epoch>.json
    # inside `if self.save_to_db:`, so False silently discards every epoch. Isolation is by
    # FILESYSTEM -- this pod trains against its own rsynced /scratch copy (asserted above),
    # which is never exported or merged, so the corpus never sees these rows.
    # First parameter is nn_code and is POSITIONAL -- passing code= raises
    # TypeError: check_nn() got an unexpected keyword argument 'code'
    _nm, acc, _, _ = api.check_nn(row[0], 'img-denoising', ds, 'psnrrgb', prm,
                                  save_to_db=True, prefix=f'cf{ds}', save_path=save_dir)
    rec["status"] = "OK"
    rec["reported_dB"] = round(acc * 48, 3)
except Exception as e:
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
rec["seconds"] = round(time.time() - t0)

curve = []
for f in (os.listdir(save_dir) if os.path.isdir(save_dir) else []):
    if not f.endswith(".json"):
        continue
    try:
        e = int(f[:-5]); trials = json.load(open(os.path.join(save_dir, f)))
    except Exception:
        continue
    if isinstance(trials, list) and trials:
        t = max(trials, key=lambda d: d.get("accuracy") or 0)
        curve.append([e, round((t.get("accuracy") or 0) * 48, 3)])
if curve:
    curve.sort()
    bi = max(range(len(curve)), key=lambda k: curve[k][1])
    db = [c[1] for c in curve]
    rec.update(n_epochs=len(curve), curve=curve, best_dB=curve[bi][1],
               best_epoch=curve[bi][0], final_dB=curve[-1][1],
               avg_dB=round(sum(db)/len(db), 3))
elif rec["status"] == "OK":
    rec["status"] = "NO_CURVE"
open(res, "a").write(json.dumps(rec) + "\n")
print(f"RESULT {key} {rec['status']} best={rec.get('best_dB')} {rec['seconds']}s")
PYEOF

  if tail -1 "$RES" 2>/dev/null | grep -q '"status": "OK"'; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
  log "[shard $IDX] $((n_ok+n_fail))/${#JOBS[@]}  ok $n_ok  fail $n_fail  (skipped $n_skip)"
done
log "=== shard $IDX DONE ok $n_ok fail $n_fail skipped $n_skip $(date -Is) ==="
[ "$n_fail" -lt "${#JOBS[@]}" ]

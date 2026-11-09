#!/usr/bin/env bash
# Train the 40-model cross-dataset subset on MAI / MIDD / SIDD and record per-epoch curves.
#
# THE QUESTION: does an architecture's ranking on synthetic AWGN predict its ranking on REAL
# sensor noise? Everything here is arranged so that the answer is about the noise and nothing
# else.
#
#   MATCHED DATA BUDGET. All three datasets are cut to 658 training pairs (MAI's native size,
#   the smallest of the three). At MIDD's or SIDD's native size the comparison would confound
#   noise difficulty with data volume, and "real noise is harder" would be indistinguishable
#   from "we trained on 40x less of it".
#
#   MATCHED METRIC. psnrrgb everywhere. ft50 measured AWGN with 'psnr' (Y-channel), which runs
#   ~0.3-0.5 dB higher and is NOT interchangeable -- so MAI is retrained here rather than
#   reusing ft50's numbers. Reusing them would have been free and wrong.
#
#   MATCHED EPOCHS. 30 for every (model, dataset) pair.
#
# MIDD ABSOLUTE dB IS NOT QUOTABLE. midd.py's per-epoch test path takes ONE 256x256 centre
# crop per full-resolution test image (~0.8% of an 8 MP frame). Its own docstring measures the
# bias as model-dependent -- 0.16 dB high for DenoiseSplitterNetX, 0.32 dB for DenoiseSplitterSP,
# compressing a 0.82 dB gap to 0.65 dB. That is fine for RANKING, which is all this experiment
# needs, but no number from here may be set against SplitterNet's published 37.92 dB. Use
# ab/nn/util/eval_midd_paper.py for anything that leaves the repo.
#
# THE OTHER SESSION'S MIDD DATA IS READ-ONLY. MIDD_ROOT points at sweep_out/midd_subset, a tree
# of SYMLINKS into nn-dataset/data/midd. Nothing under data/midd is written, and midd.py is not
# edited -- their midd-rank-x20 / midd-x30 jobs import the same loader.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-6}"
EP="${EPOCHS:-30}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
XD="${XD:-/a/mm/sweep_out/crossdata}"
WORK="${SCRATCH:-/scratch}/xd_${IDX}"
LOG="$XD/shard${IDX}.log"
RES="$XD/result_shard${IDX}.jsonl"
export CURVE_DIR="$XD/curves"
mkdir -p "$XD" "$CURVE_DIR"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== CROSSDATA shard=$IDX/$NSHARDS ep=$EP $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' --exclude='sidd' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
# sidd.py hardcodes SIDD_ROOT = cwd/sidd, so the subset is linked into place rather than
# passed by env. The repo's own nn-dataset/sidd is 320 BROKEN symlinks to a pre-migration
# /home/s493053 path and is excluded from the rsync above so it cannot shadow this.
ln -sfn /a/mm/sweep_out/sidd_subset "$WORK/sidd"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

export MIDD_ROOT=/a/mm/sweep_out/midd_subset
export MIDD_PATCH_DIR=.          # the subset IS the patch tree (sensors sit at its top level)

mapfile -t JOBS < <(python3 - "$XD/subset.json" "$IDX" "$NSHARDS" <<'PY'
import json, sys
sub, idx, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
models = json.load(open(sub))["models"]
# Shard over (model, dataset) pairs so all three datasets for one model can run on DIFFERENT
# GPUs. That is safe here in a way it is NOT for the skip ablation: this experiment compares
# a model against ITSELF ACROSS DATASETS via rank position, and ranks are computed within a
# dataset, so a per-GPU offset shifts a whole dataset's column uniformly and cancels out of
# the rank correlation. The ablation's within-base dB differences had no such protection.
work = [(m["model"], m["awgn_dB"], ds)
        for m in models for ds in ("denoise", "midd", "sidd")]
for i, w in enumerate(work):
    if i % ns == idx:
        print("\t".join(str(x) for x in w))
PY
)
log "assigned ${#JOBS[@]} (model,dataset) runs"

done_keys=""
[ -f "$RES" ] && done_keys=$(python3 -c "
import json
print(' '.join(json.loads(l)['key'] for l in open('$RES') if l.strip()))" 2>/dev/null)

n_ok=0; n_fail=0; n_skip=0
for J in "${JOBS[@]}"; do
  IFS=$'\t' read -r MODEL AWGN DS <<< "$J"
  KEY="${MODEL}__${DS}"
  case " $done_keys " in *" $KEY "*) n_skip=$((n_skip+1)); continue;; esac

  python - "$MODEL" "$AWGN" "$DS" "$KEY" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1
import os, sys, json, sqlite3, time
model, awgn, ds, key, ep, batch, lr, res = (
    sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), sys.argv[8])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

con = sqlite3.connect('file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro', uri=True)
row = con.execute("select code from nn where name=?", (model,)).fetchone()
rec = {"key": key, "model": model, "dataset": ds, "awgn_dB": awgn, "status": "FAIL"}
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
    # FILESYSTEM -- this pod trains against its own rsynced /scratch copy (asserted above)
    # which is never exported or merged, so the corpus never sees these rows.
    # First parameter is nn_code, not code -- passing code= raises
    # TypeError: check_nn() got an unexpected keyword argument 'code'
    _nm, acc, _, _ = api.check_nn(row[0], 'img-denoising', ds, 'psnrrgb', prm,
                                  save_to_db=True, prefix=f'xd{ds}', save_path=save_dir)
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

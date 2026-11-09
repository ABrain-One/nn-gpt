#!/usr/bin/env bash
# Train the skip-connection ablation grid. ALL SHARDS ON dgx-h100 BY DESIGN.
#
# HARDWARE MUST NOT VARY WITHIN THIS EXPERIMENT. The effects under test sit near a measured
# 0.14 dB run-to-run noise floor (89 corpus models with repeated 5-epoch runs). Different
# GPUs select different cuDNN kernels and differ numerically, so splitting shards across the
# H100 and the RTX fleet would confound hardware with whichever variants happened to land
# where -- exactly the way to manufacture a 0.1 dB "effect". Every comparison here is
# within-base and within-experiment, so absolute dB need not match the RTX-trained corpus.
#
# MEMORY: 8Gi per pod. ft50's inherited 48Gi limit was measured at 2Gi peak / 1Gi current
# (cgroup memory.peak inside a running pod), i.e. 24x oversized, which is what made the
# namespace memory quota -- not GPUs -- the binding resource (544Gi of 560Gi used with 5
# GPUs still free). 8Gi keeps 4x headroom over the measured peak and lets this run start
# without disturbing ft50.
#
# NOTHING IS WRITTEN TO THE CORPUS DB. These are synthetic ablation variants, not dataset
# models: check_nn runs with save_to_db=False so they never enter `nn`/`stat` and never
# occupy arch_uid space. Results live in the per-run JSON + this shard's result file.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-2}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
REPEATS="${REPEATS:-3}"          # seed variation: same config, independent runs

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
ABL="${ABL:-/a/mm/sweep_out/ablation}"
OUTD="${OUTD:-$ABL/train}"
WORK="${SCRATCH:-/scratch}/abl_${IDX}"
LOG="$OUTD/shard${IDX}.log"
RES="$OUTD/result_shard${IDX}.jsonl"
export CURVE_DIR="$OUTD/curves"
mkdir -p "$OUTD" "$CURVE_DIR"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== ABLATION TRAIN shard=$IDX/$NSHARDS ep=$EP repeats=$REPEATS $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

# SHARDED BY BASE, NOT BY RUN INDEX. The RTX fleet is MIXED -- measured across ft50's eight
# shards: 2 on RTX 4090 (Ada) and 6 on RTX 3090 (Ampere). Different architectures select
# different cuDNN kernels and differ numerically, and every effect this experiment measures
# is a WITHIN-BASE contrast (variant vs its own G0B0L0 cell) sitting near a 0.14 dB noise
# floor. Round-robin over run index would scatter one base's cells across both GPU models,
# so part of any measured difference would be silicon rather than topology.
#
# One base per shard keeps each contrast on a single GPU. Comparisons ACROSS bases are never
# made directly (the base is a random effect), so bases differing in hardware is harmless.
# gpu_type is recorded in every per-epoch JSON, so the assignment stays auditable.
mapfile -t JOBS < <(python3 - "$ABL/manifest.json" "$REPEATS" "$IDX" "$NSHARDS" <<'PY'
import json, sys
man, reps, idx, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = [r for r in json.load(open(man)) if r.get("valid") and r.get("path")]
bases = sorted({r["base"] for r in rows})           # deterministic order
mine = {b for i, b in enumerate(bases) if i % ns == idx}
# REPEATS IS PER-ROW NOW, WITH THE ENV AS FALLBACK. After ablation_rescope.py the two arms
# carry different repeat counts on purpose: the factorial resolves 0.11 dB pooled over all 8
# bases at 3 repeats, but a lateral placement lives on only 5 bases, so it needs 5 repeats to
# reach the same 0.11 dB. Taking `reps` from the environment for every row would silently
# flatten that back to one number and re-underpower the placement arm -- the exact defect the
# rescope existed to remove.
work = [(r["path"], r["base"], r["cell"], r["kind"], s)
        for r in rows if r["base"] in mine
        for s in range(int(r.get("repeats") or reps))]
print(f"# shard {idx}: bases {sorted(mine)}", file=sys.stderr)
for w in work:
    print("\t".join(str(x) for x in w))
PY
)
log "assigned ${#JOBS[@]} (variant,repeat) runs"

done_keys=""
[ -f "$RES" ] && done_keys=$(python3 -c "
import json
print(' '.join(json.loads(l)['key'] for l in open('$RES') if l.strip()))" 2>/dev/null)

n_ok=0; n_fail=0; n_skip=0
for J in "${JOBS[@]}"; do
  IFS=$'\t' read -r VPATH BASE CELL KIND REP <<< "$J"
  KEY="${BASE}__${CELL}__r${REP}"
  case " $done_keys " in *" $KEY "*) n_skip=$((n_skip+1)); continue;; esac

  python - "$VPATH" "$KEY" "$BASE" "$CELL" "$KIND" "$REP" "$EP" "$BATCH" "$LR" "$RES" <<'PYEOF' >> "$LOG" 2>&1
import os, sys, json, time
vpath, key, base, cell, kind, rep, ep, batch, lr, res = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
    int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9]), sys.argv[10])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

save_dir = os.path.join(os.environ["CURVE_DIR"], key)
os.makedirs(save_dir, exist_ok=True)
code = open(vpath).read()
prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
rec = {"key": key, "base": base, "cell": cell, "kind": kind, "repeat": rep, "status": "FAIL"}
t0 = time.time()
try:
    import ab.nn.api as api
    # save_to_db=True IS REQUIRED TO GET CURVES, AND IS STILL SAFE HERE.
    # Train.train_n_eval writes its per-epoch <epoch>.json INSIDE `if self.save_to_db:`,
    # so save_to_db=False -- the obvious choice for "don't pollute the corpus" -- silently
    # discards every epoch and yields NO_CURVE, which is exactly what the smoke test hit.
    # Isolation comes from the FILESYSTEM instead: this pod trains against its own rsynced
    # copy at /scratch/abl_<shard>/db/ab.nn.db (asserted above via ab_root_path) and that
    # DB is NEVER exported or merged -- unlike fulltrain50_shard.sh, which deliberately
    # copies its pod-local DB out. The rows die with the pod; the corpus never sees them.
    _nm, acc, _, _ = api.check_nn(code, 'img-denoising', 'denoise', 'psnr', prm,
                                  save_to_db=True, prefix='abl', save_path=save_dir)
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
    rec.update(n_epochs=len(curve), curve=curve, best_dB=curve[bi][1],
               best_epoch=curve[bi][0], final_dB=curve[-1][1])
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

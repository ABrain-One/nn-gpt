#!/usr/bin/env bash
# Retrain the selected models that have no checkpoint, keeping weights, then evaluate every
# selected model at the published full resolution.
#
# WHY ANY TRAINING IS NEEDED AT ALL. Full-resolution RGB PSNR needs trained weights. There are
# 1,679 checkpoints on disk across two stores, but coverage is 42% of the profiled set and it is
# not random: the DnSeedGA family has ZERO checkpoints while holding 7 of the top 8 places in the
# on-device Final Score ranking. So the models that most need re-measuring under the published
# protocol are precisely the ones with nothing to load.
#
# METRIC IS 'psnr', NOT 'psnrrgb', AND THAT IS DELIBERATE. Every pre-existing checkpoint --
# including LiteDenoiseNet's, the anchor this run is built around -- was selected by 256x256
# centre-crop Y-PSNR. Util.py:203 documents that this selector can inverse-rank against the
# full-resolution protocol, so it is a flawed rule; but applying a BETTER rule only to the models
# retrained here would advantage them over the anchor and destroy the comparison. Same flawed rule
# for all sixteen, and LEMUR_SAVE_EVERY_EPOCH retains the snapshots so a full-res-selected variant
# can be computed later as a sensitivity check.
#
# CHECKPOINTS MUST BE COPIED OUT OF /scratch. Util.py:198 builds the path as
# `ckpt_dir / model_name`, and ckpt_dir derives from ab_root_path -- which is the pod's rsynced
# /scratch copy. Anything left there dies with the pod, so each model's directory is copied to
# /a/mm/sweep_out/reexport/ckpt/ as soon as it is written.
#
# SHARDED OVER MODELS, ONE GPU EACH. Nothing here is compared within a shard -- each model is
# measured against a published number, not against its neighbours -- so there is no within-base
# contrast to protect and the mixed 4090/3090 fleet is harmless.
set -uo pipefail

IDX="${JOB_COMPLETION_INDEX:-${1:-0}}"
NSHARDS="${NSHARDS:-5}"
EP="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
RX="${RX:-/a/mm/sweep_out/reexport}"
WORK="${SCRATCH:-/scratch}/rx_${IDX}"
LOG="$RX/train_shard${IDX}.log"
RES="$RX/train_shard${IDX}.jsonl"
OUTCK="$RX/ckpt"
export LEMUR_SAVE_EVERY_EPOCH=1
mkdir -p "$RX" "$OUTCK"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== REEXPORT TRAIN shard=$IDX/$NSHARDS ep=$EP $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' --exclude='data' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"
ln -sfn "$SRC/data" "$WORK/data"
[ -f "$WORK/README.md" ] || { log "FATAL no README.md"; exit 1; }
cd "$WORK"; export PYTHONPATH="$WORK"

mapfile -t JOBS < <(FORCE="${FORCE:-}" python3 - "$RX/subset.json" "$IDX" "$NSHARDS" <<'PY'
import json, os, sys
sub, idx, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(sub))
STORES = ["/a/mm/sweep_out/reexport/ckpt", "/a/mm/sweep_out/ckpt", "/a/mm/nn-dataset/out/ckpt"]

# FORCE retrains a model even though a checkpoint for it already exists somewhere.
# NEEDED FOR THE ANCHOR. LiteDenoiseNet was skipped by the has-weights test above and evaluated
# from /a/mm/nn-dataset/out/ckpt/LiteDenoiseNet/best_model.pth, which is dated 2026-06-10 and
# comes from an unrelated run -- while the other ten models were trained fresh under this recipe.
# It scored 33.247 dB full-res RGB, barely over the 32.34 no-op, with psnr_min 19.76 against
# 26.5-27.2 for every GA model. That is not a comparable number, so the anchor could not be used.
# The fresh checkpoint is written to reexport/ckpt/, so the June one is never overwritten.
force = {s for s in os.environ.get("FORCE", "").split() if s}
need = []
for m in d["models"]:
    if m["model"] not in force and any(
            os.path.exists(os.path.join(s, m["model"], "best_model.pth")) for s in STORES):
        continue
    need.append((m["model"], m["src"]))
for i, w in enumerate(need):
    if i % ns == idx:
        print("\t".join(w))
PY
)
log "assigned ${#JOBS[@]} models to retrain"

n_ok=0; n_fail=0
for J in "${JOBS[@]}"; do
  IFS=$'\t' read -r MODEL SRCF <<< "$J"
  python - "$MODEL" "$RX/src/$SRCF" "$EP" "$BATCH" "$LR" "$RES" "$OUTCK" <<'PYEOF' >> "$LOG" 2>&1
import json, os, shutil, sys, time
model, srcf, ep, batch, lr, res, outck = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]),
    sys.argv[6], sys.argv[7])
from ab.nn.util.Const import ab_root_path
assert str(ab_root_path).startswith('/scratch/'), f'ROOT ESCAPED TO {ab_root_path}'

new_name = None
rec = {"model": model, "status": "FAIL"}
prm = {'lr': lr, 'batch': batch, 'transform': 'denoise', 'epoch': ep, 'num_workers': 6}
t0 = time.time()
try:
    import ab.nn.api as api
    # save_pth_weights=True is the whole reason this runs. save_to_db=True is required for the
    # per-epoch JSON, and is safe because this pod trains against its own /scratch DB copy
    # (asserted above) which is never exported or merged.
    _nm, acc, _, _ = api.check_nn(open(srcf).read(), 'img-denoising', 'denoise', 'psnr', prm,
                                  save_to_db=True, prefix='rx', save_pth_weights=True)
    new_name = _nm            # the ONLY reliable key to the checkpoint directory; see below
    rec.update(status="OK", crop_psnr_y=round(acc * 48, 3))
except Exception as e:
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"

# Copy the checkpoint out of /scratch before the pod can take it away.
#
# THE DIRECTORY IS NAMED AFTER THE PREFIX, NOT THE MODEL. check_nn(prefix='rx') renames the model,
# so Util.py:198 writes to  <root>/out/ckpt/rx-<32 hex>/best_model.pth  -- e.g. the run for
# DnSeedGA-20bab3230791d09509d138a198760e60 landed in rx-20bab3230791d09509d138a198760e60. The first
# version of this looked for the ORIGINAL name, found nothing, recorded NO_WEIGHTS for all six
# models, and the pod then deleted /scratch. 92 minutes of H100-class training on five shards was
# lost to a naming assumption -- the weights existed the whole time, one directory over.
#
# Matching on the trailing hash rather than the prefix keeps this working if the prefix changes.
# AND THE HASH TRICK ONLY WORKS IF THE NAME ALREADY CONTAINS THE HASH. It does for generated
# models (DnSeedGA-<32 hex> -> rx-<same 32 hex>), which is why the fix above worked for them and
# the bug looked closed. A HAND-WRITTEN name has no hash: for LiteDenoiseNet, model.rsplit('-')[-1]
# is the literal string "LiteDenoiseNet", while check_nn had written to rx-1d38207d9ddae140f6..., a
# fresh id unrelated to the name. Nothing matched, the row was recorded NO_WEIGHTS, and /scratch was
# reclaimed -- discarding a completed 50-epoch run that had reached 35.231 dB crop. That run was the
# anchor the whole re-export exists to establish.
#
# check_nn RETURNS the new name and it was being thrown away into `_nm`. That is the authoritative
# key, so try it first and keep the heuristics only as a fallback.
suffix = model.rsplit('-', 1)[-1]
ckroot = os.path.join(str(ab_root_path), 'out', 'ckpt')
cands = ([os.path.join(ckroot, new_name)] if new_name else [])
cands += [os.path.join(ckroot, model), os.path.join(ckroot, model.split('.')[-1])]
if os.path.isdir(ckroot):
    cands += [os.path.join(ckroot, d) for d in os.listdir(ckroot)
              if d.endswith(suffix) or suffix in d]
saved = None
for d in cands:
    if os.path.isdir(d):
        dst = os.path.join(outck, model)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(d):
            try:
                shutil.copy2(os.path.join(d, f), os.path.join(dst, f))
            except Exception as e:
                print(f"  copy failed {f}: {e}")
        saved = dst
        break
rec["ckpt_saved"] = saved
rec["has_best"] = bool(saved and os.path.exists(os.path.join(saved, 'best_model.pth')))
if rec["status"] == "OK" and not rec["has_best"]:
    rec["status"] = "NO_WEIGHTS"          # trained but produced nothing loadable -- not a success
rec["seconds"] = round(time.time() - t0)
open(res, "a").write(json.dumps(rec) + "\n")
print(f"RESULT {model} {rec['status']} crop={rec.get('crop_psnr_y')} {rec['seconds']}s -> {saved}")
PYEOF

  if tail -1 "$RES" 2>/dev/null | grep -q '"status": "OK"'; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
  log "[shard $IDX] $((n_ok+n_fail))/${#JOBS[@]}  ok $n_ok  fail $n_fail"
done
log "=== shard $IDX training done ok $n_ok fail $n_fail $(date -Is) ==="

# Shard 0 waits for the others, then evaluates EVERY selected model at full resolution. One shard
# does the evaluation because fullres_eval.py appends to a single JSONL and resumes by model name;
# running it in five pods concurrently would interleave writes into that file.
if [ "$IDX" -eq 0 ]; then
  log "--- shard 0: waiting for the other shards to finish training ---"
  for i in $(seq 1 240); do
    live=$(ls "$RX"/train_shard*.jsonl 2>/dev/null | wc -l)
    tot=$(cat "$RX"/train_shard*.jsonl 2>/dev/null | wc -l)
    want=$(python3 -c "
import json,os
d=json.load(open('$RX/subset.json'))
S=['/a/mm/sweep_out/ckpt','/a/mm/nn-dataset/out/ckpt']
print(sum(1 for m in d['models'] if not any(os.path.exists(os.path.join(s,m['model'],'best_model.pth')) for s in S)))")
    [ "${tot:-0}" -ge "${want:-99}" ] && { log "all $tot/$want training rows in"; break; }
    sleep 60
  done
  log "--- full-resolution RGB PSNR, 50 frames at 3200x2432 ---"
  python3 "$GA/fullres_eval.py" --subset "$RX/subset.json" --val "$SRC/validation" \
      --out "$RX/fullres.jsonl" 2>&1 | tee -a "$LOG"
  log "=== full-res evaluation complete ==="
fi
[ "$n_fail" -lt "${#JOBS[@]}" ] || [ "${#JOBS[@]}" -eq 0 ]

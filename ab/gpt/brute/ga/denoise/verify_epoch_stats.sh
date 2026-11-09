#!/usr/bin/env bash
# VERIFY the GA per-epoch statistics hook (2026-07-30).
#
# Claim under test: passing save_path into api.check_nn makes Train.train_n_eval emit a
# per-epoch <epoch>.json on its is_code branch, so run_seed_ga.epoch_stats can recover a
# real learning curve for a GA-generated model. Before the hook, all 131 DenoiseGA-*
# models carried exactly ONE stat row (epoch=5) and the curve was unrecoverable.
#
# Trains a real denoise model through the exact GA entry point (api.check_nn with
# prefix="DenoiseGA") for 3 epochs and asserts n_epochs == 3, not 1.
#
# save_to_db MUST be True: train_n_eval guards the per-epoch save_results call behind
# `if self.save_to_db:`, so save_to_db=False would write no JSON and the test would be
# vacuous. Hence the pod-local DB copy -- the master DB on NFS is never opened for write.
set -uo pipefail

SRC=/a/mm/nn-dataset
OUTD=/a/mm/sweep_out/verify
WORK="${SCRATCH:-/scratch}/lemur_verify"
LOG="$OUTD/verify_epoch_stats.log"
mkdir -p "$OUTD"; : > "$LOG"
log(){ echo "$@" | tee -a "$LOG"; }

log "=== verify epoch_stats hook $(date -Is) on $(hostname) ==="
mkdir -p "$WORK"
rsync -a --exclude='train' --exclude='validation' --exclude='.git' \
         --exclude='out' --exclude='results' "$SRC/" "$WORK/" >>"$LOG" 2>&1 || {
  log "FATAL rsync failed"; exit 1; }
ln -sfn "$SRC/train" "$WORK/train"; ln -sfn "$SRC/validation" "$WORK/validation"

cd "$WORK"
export PYTHONPATH="$WORK:/a/mm/nn-gpt/ab/gpt/brute/ga/denoise"
python - >>"$LOG" 2>&1 <<'PYEOF'
import os, sys, json, tempfile
from ab.nn.util.Const import ab_root_path, db_file
assert "/scratch" in str(ab_root_path), f"ROOT ESCAPED TO {ab_root_path}"
print("pod-local db:", db_file, flush=True)

sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise")
import run_seed_ga as G
import ab.nn.api as api

code = open("/a/mm/nn-dataset/ab/nn/nn/DenoiseThunder.py").read()
stat_dir = tempfile.mkdtemp(prefix="epstats_")
EPOCHS = 3
prm = {"lr": 1e-4, "batch": 16, "transform": "denoise",
       "epoch": EPOCHS, "num_workers": 6}

print(f"--- check_nn(save_path={stat_dir}) for {EPOCHS} epochs ---", flush=True)
db_name, acc, a2t, q = api.check_nn(code, G.TASK, G.DATASET, G.METRIC, prm,
                                    save_to_db=True, prefix="DenoiseGA",
                                    save_path=stat_dir)
print(f"db_name={db_name} acc={acc:.4f} -> {acc*G.PSNR_SCALE:.3f} dB", flush=True)

files = sorted(os.listdir(stat_dir))
print("per-epoch files written:", files, flush=True)

st = G.epoch_stats(db_name, G.PSNR_SCALE, stat_dir)
print("epoch_stats(JSON) =", json.dumps(st, indent=2), flush=True)

db_only = G.epoch_stats(db_name, G.PSNR_SCALE, None)
print("epoch_stats(DB fallback) n_epochs =",
      (db_only or {}).get("n_epochs"), flush=True)

ok = st is not None and st["n_epochs"] == EPOCHS and len(st["curve"]) == EPOCHS
print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
      f"(expected n_epochs={EPOCHS}, got {(st or {}).get('n_epochs')}; "
      f"DB-only path gives {(db_only or {}).get('n_epochs')})", flush=True)
sys.exit(0 if ok else 1)
PYEOF
R=$?
grep -E "VERDICT|per-epoch files|db_name=|n_epochs|ROOT ESCAPED|Error|Traceback" "$LOG" | tail -20
log "=== exit=$R ==="
exit $R

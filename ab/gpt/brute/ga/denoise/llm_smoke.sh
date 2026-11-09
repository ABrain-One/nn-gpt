#!/usr/bin/env bash
# SMOKE TEST -- exercise the whole chain end to end on 1 GPU before committing 550
# candidates to it. Validates, in order:
#   1. llm_denoise.py composing from Denoise_shared_blocks + the strategy configs
#      (only ever dry-run so far -- never with a model loaded)
#   2. real generation + extraction for BOTH arms
#   3. screen_and_train.py, which has never been executed at all: 1-epoch screen,
#      the ep1 gate, full train, and the DB write with prefix LLM-Denoise
# Tiny on purpose: 2 candidates per arm, 2-epoch "full" train, low gate so at least one
# survivor exercises stage 2.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
OUT=/a/mm/sweep_out/llm_smoke
mkdir -p "$OUT"
export HF_HOME=/a/mm/hf_cache HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=/a/mm/nn-gpt
cd "$GA" || exit 1

echo "=== [1/3] generate: 2 cold + 2 seeded ==="
python -u llm_denoise.py --mode both --n 2 --supporting 3 --out "$OUT/run" || exit 1

echo
echo "=== [2/3] what came out ==="
ls "$OUT/run/cand" 2>/dev/null
echo "candidate.py files: $(ls "$OUT"/run/cand/*/candidate.py 2>/dev/null | wc -l)"

echo
echo "=== [3/3] screen + train (gate 25 dB so stage 2 is exercised, 2 epochs) ==="
# MUST run from the nn-dataset tree. Stage 1 needs nn-gpt on the path (ab.gpt.util.LLM),
# stage 3 needs nn-dataset (ab.nn.loader.denoise + LEMUR root + CWD-relative train/ data).
# Running stage 3 from the nn-gpt dir is what made the first smoke run fail 4/4 with
# "No module named 'ab.nn.loader.denoise'" while LEMUR rooted itself at /a/mm/nn-gpt.
cd /a/mm/nn-dataset || exit 1
PYTHONPATH=/a/mm/nn-dataset python -u "$GA/screen_and_train.py" \
       --run-dir "$OUT/run" --out "$OUT/eval" --gate 25.0 --epochs 2 || exit 1

echo
echo "=== DB rows written with prefix LLM-Denoise ==="
python -u - <<'PY'
import sqlite3
c=sqlite3.connect('file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro',uri=True)
# GLOB 'LLM[-_]Denoise*' matches the current 'LLM-Denoise' spelling and the legacy
# 'LLM_Denoise' rows alike. LIKE is wrong here: '_' is a single-char wildcard in LIKE,
# so the old pattern matched by accident rather than by intent.
r=c.execute("SELECT nn,COUNT(*),MAX(accuracy) FROM stat WHERE nn GLOB 'LLM[-_]Denoise*' GROUP BY nn").fetchall()
print(f"  {len(r)} LLM-Denoise model(s) in DB")
for nn,n,a in r: print(f"    {nn}  rows={n}  best={a*48:.3f} dB")
PY
echo "=== SMOKE COMPLETE ==="

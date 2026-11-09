#!/usr/bin/env bash
# Reference-arm A/B generation, run from a POD-LOCAL root so it is not I/O starved.
#
# WHY THIS WRAPPER EXISTS. The first attempt ran `cd /a/mm/nn-gpt && python llm_denoise.py ...`
# straight from the shared hostPath and produced ZERO candidates in 2.5 hours. Diagnosis:
#   * ab.nn.util.Const.__project_root_path() walks up from the CWD looking for ab/ + README.md.
#     With CWD=/a/mm/nn-gpt it resolves the LEMUR root to nn-gpt, so Const.db_file becomes
#     /a/mm/nn-gpt/db/ab.nn.db -- a 1,464 MB, 19,473-model, all-tasks database with ZERO denoise
#     models in it. The reference pool never used it (db_path() correctly finds nn-dataset, which
#     is why the production run reported 2,785 references) but the file was still being read.
#   * Both pods sat in D-state (wait_on_page_bit_common) at ~0.33 MB/s while 13 training pods and
#     the farm submitter competed for the same filesystem. ~40 s of CPU in 30 min.
#
# The fix is two lines of setup, matching what every TRAINING script here already does:
#   1. rsync a minimal nn-dataset root into /scratch and cd there, so Const resolves the LEMUR root
#      locally and its incidental db read hits a 43 MB local file instead of a 1.5 GB shared one.
#   2. DENOISE_DB points db_path() at that local copy too, so the reference-corpus read is local.
# The DB is copied with sqlite3's backup() API, never `cp` -- a plain copy of a live database can
# capture a torn page.
#
# WHAT THIS DOES NOT CHANGE: the generation itself. Same adapter, same prompts, same n, same
# tournament settings as llmprod-a10.yaml. Only where bytes are read from.
#
# IS THERE AN UPSTREAM PATTERN FOR THIS? No -- checked against current main (2026-08-08). Every
# ab/gpt/act/ runner does `from ab.gpt.util.Const import ab_root_path` and uses it directly: no
# /scratch root, no sqlite backup(), no chdir anywhere in the tree. The only sanctioned env
# overrides are AB_GPT_NNGPT_DIR / NNGPT_DIR_OVERRIDE (Const.py:33) and those redirect OUTPUT
# (nngpt_dir), not the DB or the project root. There is no DB-path override at all; db_file is
# purely derived (db_dir / 'ab.nn.db').
#
# Upstream does not need one: their runners execute one at a time, so reading the 1.5 GB shared DB
# costs minutes. Ours runs alongside 13 training pods on one filesystem, where the same read took
# 35+ min at 0.7 MB/s -- and 2.5 h without producing a candidate when the farm submitter was also
# active, against a 9-minute normal startup measured from llmprod's own meta.json timestamps.
#
# So this wrapper is additive, not a fork: it modifies NO upstream file. It uses upstream's own
# root-resolution mechanism as designed -- __project_root_path() walks up from the CWD -- and simply
# points it at a valid local root. If output redirection is ever needed here, use AB_GPT_NNGPT_DIR
# rather than adding a second mechanism.
set -uo pipefail

ARM="${ARM:?set ARM=ctl|trt}"
MERGE="${DENOISE_MERGE_GA_ARMS:?set 0 or 1}"
N="${N:-438}"
ADAPTER="${ADAPTER:-/a/mm/sweep_out/run3/cycle10/adapter}"

SRC=/a/mm/nn-dataset
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
OUT=/a/mm/sweep_out/refab/$ARM
ROOT="${SCRATCH:-/scratch}/refab_$ARM"
mkdir -p "$OUT"
LOG="$OUT/gen.log"

log(){ echo "$@" | tee -a "$LOG"; }
log "=== REFAB GEN arm=$ARM merge_ga_arms=$MERGE n=$N $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

# MERGED local LEMUR root. Both Const modules share ONE root: ab/gpt/util/Const.py line 1 does
# `from ab.nn.util.Const import ab_root_path` and then `gpt_dir = ab_root_path / 'ab' / 'gpt'`.
# So a root holding only nn-dataset's ab/nn satisfies the DB lookup but makes ab.gpt.util.Const
# look for <root>/ab/gpt/conf/llm/*.json and die with FileNotFoundError -- which is exactly how the
# first fixed attempt failed. The root must therefore carry ab/nn AND ab/gpt.
#
# ab/gpt/brute is EXCLUDED: it is 6.7 GB of our own sweep artifacts and nothing here imports from
# it (llm_denoise.py is invoked by absolute path out of /a/mm). conf+util is ~12 MB.
#
# Excludes are ANCHORED with a leading slash. A bare --exclude='db' matches ANY path component
# named db, which silently stripped ab/nn/util/db/ -- a python package -- and produced
# "ModuleNotFoundError: No module named 'ab.nn.util.db'" plus a warning that the 9053 MMac cap was
# not being enforced.
mkdir -p "$ROOT/db" "$ROOT/ab"

# WHAT GOES IN, and why each piece is required. Validated on the host before this ever ran; every
# item below is here because leaving it out produced a concrete failure:
#   README.md + version + pyproject.toml   __project_root_path() needs ab/ + README.md to pick the
#                                          root, and Const.get_version() (Const.py:22) then demands
#                                          `version` AND `pyproject.toml` in that same dir or it
#                                          raises FileNotFoundError at import time.
#   ab/__init__.py                         makes ab a package
#   ab/nn minus nn/ and stat/              util, loader, metric, transform, api. nn/ (140 MB of
#                                          model sources) and stat/ (188 MB, 13.8k json) are NOT
#                                          needed: nn_dir() resolves absolutely to /a/mm, and the
#                                          reference pool comes from the DB. Copying them made the
#                                          root ~1.1 GB and filled the disk.
#   ab/gpt minus brute/                    ab.gpt.util.Const derives gpt_dir from the SAME
#                                          ab_root_path (ab/gpt/util/Const.py:1), so a root without
#                                          ab/gpt dies on
#                                          <root>/ab/gpt/conf/llm/*.json FileNotFoundError.
#                                          brute/ is 6.7 GB of our own artifacts, imported from
#                                          /a/mm by absolute path instead.
#
# EXCLUDE PATTERNS ANCHOR AT THE TRANSFER ROOT, WHICH IS THE PARENT OF THE SOURCE ARG. Copying
# "$SRC/ab/nn" makes paths read `nn/...`, so the pattern for ab/nn/nn is '/nn/nn' -- NOT '/nn',
# which matches the top-level directory and excludes EVERYTHING (that produced a root with no
# ab/nn/util/Const.py at all). Likewise ab/gpt/brute is '/gpt/brute', not '/brute', which matches
# nothing. And a bare 'db' with no slash matches ANY component named db, which silently removed
# ab/nn/util/db/ -- a python package -- yielding "ModuleNotFoundError: No module named
# 'ab.nn.util.db'". Verified by `rsync -an --stats`: 392 files / 1.7 MB for ab/nn, and the
# resulting root imports both Const modules cleanly.
rsync -a --exclude='/nn/nn' --exclude='/nn/stat' \
         "$SRC/ab/nn" "$ROOT/ab/" >>"$LOG" 2>&1 || { log "FATAL rsync ab/nn failed"; exit 1; }
rsync -a --exclude='/gpt/brute' \
         /a/mm/nn-gpt/ab/gpt "$ROOT/ab/" >>"$LOG" 2>&1 || { log "FATAL rsync ab/gpt failed"; exit 1; }
cp -p "$SRC/ab/__init__.py" "$ROOT/ab/__init__.py" 2>/dev/null || : > "$ROOT/ab/__init__.py"
for f in README.md version pyproject.toml; do
  cp -p "$SRC/$f" "$ROOT/$f" 2>/dev/null || { log "FATAL cannot copy $f"; exit 1; }
done

for need in "$ROOT/README.md" "$ROOT/version" "$ROOT/pyproject.toml" \
            "$ROOT/ab/nn/util/Const.py" "$ROOT/ab/nn/util/db/Calc.py" \
            "$ROOT/ab/gpt/util/Const.py" "$ROOT/ab/gpt/conf/llm"; do
  [ -e "$need" ] || { log "FATAL local root missing $need"; exit 1; }
done
log "local root ok ($(du -sm "$ROOT" | cut -f1) MB): ab/nn(-nn,-stat) + ab/gpt(-brute)"

python3 -c "
import sqlite3,sys
a=sqlite3.connect(sys.argv[1]); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()
" "$SRC/db/ab.nn.db" "$ROOT/db/ab.nn.db" >>"$LOG" 2>&1 || { log "FATAL db backup failed"; exit 1; }
log "local root $ROOT  db $(du -m "$ROOT/db/ab.nn.db" | cut -f1) MB"

# prove the local DB is the right corpus BEFORE spending H100 time: a run whose pool lacks
# DnSeedGA cannot test a treatment that promotes DnSeedGA, and that is exactly how the first
# attempt would have failed silently had it ever finished.
python3 - "$ROOT/db/ab.nn.db" <<'PYEOF' >>"$LOG" 2>&1 || { log "FATAL corpus check failed"; exit 1; }
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dn = list(con.execute("select count(*) from nn where name like 'DnSeedGA-%'"))[0][0]
tot = list(con.execute("select count(*) from nn"))[0][0]
print(f"[corpus] local db: {tot} models, {dn} DnSeedGA")
assert dn > 100, f"only {dn} DnSeedGA in the local db -- the A/B would be meaningless"
PYEOF

cd "$ROOT"
export DENOISE_DB="$ROOT/db/ab.nn.db"
export DENOISE_MERGE_GA_ARMS="$MERGE"
export PYTHONPATH="/a/mm/nn-gpt:$ROOT"

python -u "$GA/llm_denoise.py" \
  --mode seeded --n "$N" --supporting 3 --seed-set db \
  --adapter "$ADAPTER" \
  --tournament 3 --elite-share 0 \
  --out "$OUT/run" --id-prefix "r$ARM" \
  --shard 0 --num-shards 1 >>"$LOG" 2>&1
rc=$?
log "=== arm $ARM done rc=$rc, candidates: $(ls "$OUT/run/cand" 2>/dev/null | wc -l) $(date -Is) ==="
exit $rc

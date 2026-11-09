#!/usr/bin/env bash
# GREENLIT DB POPULATE (2026-07-27): persist each NEW family to the shared LEMUR DB under its
# CLEAN name, so it becomes --seeds-resolvable and gets a measured PSNR (which then lets us
# rank "best" for the distinct-10 GA seeding). The families are already registered as arch
# files in ab/nn/nn/ (copied from new_seeds/); this trains each by name via ab.nn.train
# (1 fixed-hp trial) and save_to_db writes the stat row under the clean basename.
#
# SEED SET (2026-07-28): defaults to the 17 DISTINCT new families (NEW_DISTINCT_FILES), NOT all
# 22. The 5 excluded are the clones seeds_distinct.sh already drops from the pool -- the norm
# cluster InstNorm/LayerNorm/ResBN/StatNorm (cos 0.94-0.998 to GroupNorm) and SplitterNetL
# (cos 0.995 to SplitterNet). Training them would spend GPU time measuring near-duplicates that
# no downstream launcher seeds from. Set SEED_SET=all to fall back to all 22.
#
# REQUIRES A GPU (this is a training pass). NOTE: git push/commit stays USER-MANUAL.
# Usage: ./populate_db.sh [EPOCHS]   (default 5 = the standard denoise eval budget)
#        SEED_SET=all ./populate_db.sh 5   # include the 5 dropped clones
set -uo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
source "$D/seeds_distinct.sh"
EP="${1:-5}"
LOG="$D/populate_db.log"; : > "$LOG"

case "${SEED_SET:-distinct}" in
  all)      FILES="$ALL_NEW_FILES";      SETNAME="all 22 new families" ;;
  *)        FILES="$NEW_DISTINCT_FILES"; SETNAME="17 DISTINCT new families" ;;
esac
N=$(echo $FILES | wc -w)

echo "=== populate DB: $SETNAME ($N) @ ${EP}ep (dataset=denoise metric=psnr) ===" | tee -a "$LOG"
echo "=== started $(date -Is) on $(hostname) ===" | tee -a "$LOG"
for f in $FILES; do
  name=$(basename "$f" .py)
  echo "--- training $name ---" | tee -a "$LOG"
  # one fixed-hp trial; save_to_db persists under the clean arch-file name
  "$PY" -c "
import ab.nn.train as T
T.main(config='img-denoising_denoise_psnr_${name}',
       nn_prm={'lr':1e-4,'batch':16,'transform':'denoise','num_workers':6},
       epoch_max=${EP}, n_optuna_trials=1, transform='denoise', num_workers=6)
" >> "$LOG" 2>&1 && echo "  OK $name" | tee -a "$LOG" || echo "  FAIL $name (see log)" | tee -a "$LOG"
done
echo "=== finished $(date -Is) ===" | tee -a "$LOG"

# report the measured PSNR for exactly the families this run targeted (real dB = accuracy * 48)
echo "=== measured PSNR ===" | tee -a "$LOG"
NAMES=$(for f in $FILES; do basename "$f" .py; done | tr '\n' ' ')
# NOTE: no 2>/dev/null here. It was suppressing this block's OWN errors, so when the
# summary failed it printed an empty table and the run still looked clean -- the numbers
# had to be re-queried by hand afterwards. Errors belong in the log.
"$PY" - "$NAMES" <<'PYEOF' | tee -a "$LOG"
import sys, sqlite3, statistics
names = sys.argv[1].split()
con = sqlite3.connect("file:/a/mm/nn-dataset/db/ab.nn.db?mode=ro", uri=True)
cur = con.cursor()
# full per-epoch statistics, not just the peak: on a short budget a model whose best is
# its LAST epoch was still improving when the cap hit, so `best` alone is a floor.
h = f"{'family':<22}{'n_ep':>5}{'max_ep':>7}{'best_dB':>9}{'@ep':>5}{'final':>8}{'avg':>8}{'std':>7}{'min':>8}"
print(h); print("-" * len(h))
for n in names:
    cur.execute("SELECT epoch, accuracy FROM stat WHERE task='img-denoising' AND nn=? "
                "ORDER BY epoch", (n,))
    rows = [(e, a) for e, a in cur.fetchall() if a is not None]
    if not rows:
        print(f"{n:<22}{'NO ROW':>44}"); continue
    by = {}
    for e, a in rows:
        if e not in by or a > by[e]:
            by[e] = a
    eps = sorted(by); v = [by[e] * 48 for e in eps]
    bi = max(range(len(v)), key=lambda i: v[i])
    climb = "^" if eps[bi] == max(eps) else " "
    sd = statistics.pstdev(v) if len(v) > 1 else 0.0
    print(f"{n:<22}{len(v):>5}{max(eps):>7}{v[bi]:>9.2f}{eps[bi]:>4}{climb}"
          f"{v[-1]:>8.2f}{statistics.mean(v):>8.2f}{sd:>7.2f}{min(v):>8.2f}")
print("\n^ = best epoch is the last epoch: budget was binding, value is a floor.")
PYEOF
echo "=== done ===" | tee -a "$LOG"

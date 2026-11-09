#!/usr/bin/env bash
# STAGE 2 of the refab A/B: 1-epoch screen of one arm's preflight survivors, sharded across GPUs.
#
# WHY THIS EXISTS AS ITS OWN SCRIPT. verify_refab.sh already proved the TREATMENT took effect --
# the DnSeedGA reference share moves 4.5% (control, == 4.4% production) -> 25.9% (treatment).
# That is necessary but says nothing about whether the intervention HELPED: a generator can be
# handed better references and still emit models that do not build, or build and denoise worse.
# The A/B is only answered by putting both arms through the identical screen and comparing the
# ep1 PSNR distributions. Until then "treatment confirmed" is a statement about prompts, not
# about models.
#
# ONE ARM PER JOB, NOT BOTH IN ONE. Not because the ids would collide -- checked, and they do
# not: the arms are prefixed rctl_/rtrt_ and share 0 of 438 ids, so a combined run would in fact
# stay attributable. The reason is operational: separate --out means one arm failing, timing out
# or needing a re-run never touches the other's results, and the two ep1 distributions stay in
# separate files so comparing them cannot accidentally pool them. Cheap insurance on the
# measurement the whole A/B rests on.
#
# SCREEN ONLY -- no DB writes, so N pods run concurrently without putting N writers on the
# NFS-hosted SQLite DB (screen_and_train.py refuses that combination outright).
#
# POD-LOCAL LEMUR ROOT -- not optional; see llm_screen_shard.sh for the full account. check_nn
# writes each candidate into out/nn/tmp/ and imports it, so pods sharing one NFS out/ delete
# each other's module mid-import. It is a RACE and will not reproduce on a single pod.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
SRC=/a/mm/nn-dataset
ARM="${ARM:?set ARM=ctl or ARM=trt}"
REFAB=/a/mm/sweep_out/refab/$ARM
OUT=$REFAB/screen
SHARD="${JOB_COMPLETION_INDEX:-0}"
NSHARDS="${NSHARDS:-4}"
# Label, not filter: screen_and_train records ep1 for every candidate that trains at all, so
# the full distribution lands on disk and any gate can be re-applied afterwards for free.
GATE="${GATE:-32.7}"
ROOT=/scratch/root
mkdir -p "$OUT" "$ROOT/out"

echo "=== refab screen | arm=$ARM shard $SHARD/$NSHARDS gate=$GATE | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

for l in ab README.md db data train validation; do
  [ -e "$SRC/$l" ] && ln -sfn "$SRC/$l" "$ROOT/$l"
done
cd "$ROOT" || exit 1
echo "pod-local LEMUR root: $ROOT  (out/ local; ab|db|data symlinked into $SRC)"
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
[ -d "$ROOT/train" ] && [ -d "$ROOT/validation" ] || { echo "FATAL: data not visible"; exit 1; }

# Screen only what preflight cleared. The rejects were never going to produce a PSNR, and at a
# measured ~80 s per screened candidate the dead ones are pure GPU waste. If pass_ids.txt is
# absent, fail loudly rather than silently screening all 438 and blowing the budget.
IDS=$REFAB/preflight/pass_ids.txt
[ -s "$IDS" ] || { echo "FATAL: $IDS missing or empty -- run preflight.py for arm $ARM first"; exit 1; }
echo "preflight survivors for $ARM: $(wc -l < "$IDS")"

PYTHONPATH="$ROOT" python -u "$GA/screen_and_train.py" \
    --run-dirs "$REFAB/run" --only-ids "$IDS" \
    --out "$OUT" --gate "$GATE" --screen-only \
    --shard "$SHARD" --num-shards "$NSHARDS"
rc=$?
echo "=== arm=$ARM shard $SHARD done rc=$rc $(date -Is) ==="
exit $rc

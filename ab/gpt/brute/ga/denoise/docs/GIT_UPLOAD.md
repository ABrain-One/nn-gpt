# Denoise upload runbook — read this before every push

Scope is **denoise only**. All git commands are the user's to run; the tooling here never commits,
pushes, or stages. Last updated 2026-08-11, after batch 1 landed.

---

## 0. Push history

| commit | what | files |
|---|---|---|
| `4f4f8dda948` | seed models + per-epoch stats | — |
| `befad57a9ad` | 132 distinct generated models (DenoiseGA-*/DenoiseSkip-*) | 132 |
| `b4af36b9de5` | per-epoch training stats for those 132 | 132 |
| **`6d4dd25fa04`** | **denoise: add 500 distinct models + per-epoch stats (batch 1/5)** | **3,880** |

Branch: `add-denoise-models-and-stats` → `origin/add-denoise-models-and-stats`
(remote `https://github.com/ABrain-One/nn-dataset.git`). Batch 1 = 500 `.py` + 3,380 stat `.json`.
Ledger records those 500 as uploaded.

View a push:
`https://github.com/ABrain-One/nn-dataset/commit/<hash>` — GitHub truncates the diff above a few
hundred files and paginates directory listings at 1,000 entries. That is display-only; trust the
file count in the header, or `git ls-remote origin add-denoise-models-and-stats`.

---

## 1. THE ONE THING THAT MUST NOT HAPPEN

```
git add -A        # NO
git commit -a     # NO
```

This working tree has **252,223 tracked files missing from disk**, nearly all belonging to *other*
tasks (`img-captioning_coco_*` etc.) because it is a pruned local copy, not a full mirror. Either
command would stage every one as a **deletion** and wipe other people's stats. Always add by
explicit pathspec.

---

## 2. The procedure that works

```bash
cd /shared/ssd/home/b-b-ahmed/thesis/nn-dataset
D=/shared/ssd/home/b-b-ahmed/thesis/nn-gpt/ab/gpt/brute/ga/denoise
N=2                                    # batch number

# 1. stage — --pathspec-from-file avoids a several-thousand-path command line
git add --pathspec-from-file=$D/docs/upload_batches/batch_00$N.txt

# 2. verify the staged set. MUST be all 'A', zero 'D', nothing outside the two denoise dirs
git diff --cached --name-status > /tmp/st.txt
awk '{print $1}' /tmp/st.txt | sort | uniq -c
awk '{print $2}' /tmp/st.txt | grep -vcE '^ab/nn/nn/|^ab/nn/stat/train/'

# 3. commit
git commit -m "denoise: add 500 distinct models + per-epoch stats (batch $N/5)"

# 4. VERIFY THE COMMIT MOVED before pushing — must print a new hash and 1
git log --oneline -1
git rev-list --count origin/add-denoise-models-and-stats..HEAD

# 5. push, and read the output
git push origin add-denoise-models-and-stats

# 6. ONLY if the push transferred objects
python3 $D/upload_batches.py --confirm $N
```

---

## 3. Mistakes already made — do not repeat

**`--confirm` after a no-op push.** A push printing `Everything up-to-date` transferred nothing.
That happened once: the commit step was skipped, push was a no-op, and `--confirm 1` recorded 500
models as uploaded that existed only in the index. `upload_batches.py` excludes ledger entries from
future batches, so those 500 would have been **silently skipped forever**. Rolled back by hand.
**Rule: confirm only after seeing `<old>..<new>` in the push output.**

**Git identity was a placeholder.** `user.email` was `you@example.com` and `user.name` was unset,
producing `fatal: empty ident name`. Earlier commits used `Asad Ahmed <s493053@julia2.julia.lan>`
— a hostname on the retired julia2 machine, which GitHub cannot attribute to an account. Set
repo-local (no `--global`, shared machine):
```bash
git config user.name "Asad Ahmed" && git config user.email "<real address>"
git config user.name && git config user.email      # read back
```

**`git add` expands directories.** A batch list holds ~1,014 *pathspecs*, but 514 of them are stat
*directories*, so git stages their contents: 496 dirs x 5 epochs + 18 x 50 = 3,380 files, + 500
`.py` = **3,880**. Seeing 3,880 from a 1,014-line list is correct, not a bug. Expect ~3,000-3,900
per batch; batch 5 is smaller (300 models).

**Git queries here need long timeouts.** `git diff --cached` over this repo exceeded a 90 s timeout
and returned empty, which was misread as "index empty, commit already done". Use 300-600 s and
check the exit code.

---

## 4. Never commit these

```
nn-gpt/ab/gpt/brute/ga/denoise/*.pre-*        # bulk_profile, run_seed_ga, skip_operators backups
sweep_out/reexport/subset.json.orig-june-anchor
docs/upload_ledger.json.bak
```

- `ab/nn/nn/DenoiseUnet{3F,6F,8F,9F}.py`, `ab/nn/nn/LiteDenoiseNet.py` — the **BN-poisoning guard**,
  modified and deliberately **unstaged**. A code review raised a **0.0-loss** finding on it, and this
  is the code that *writes the stats*, so settle that before any stats-bearing push. An earlier
  version of this document told you to stage these; that advice was wrong.
- `data_sidd_bench/*.mat` — 460 MB; `ab/nn/loader/sidd_bench.py` documents the download URL instead.
- `nn-dataset/sidd/` — 320 dangling symlinks to `/home/s493053`. Real data is `sweep_out/sidd_subset`.
- anything under `sweep_out/` — run artifacts, private DB copies, checkpoints.

---

## 5. Tooling

| script | purpose |
|---|---|
| `upload_batches.py --status / --plan / --confirm N` | plan batches of 500, dedup by `arch_uid2`, ledger |
| `validate_upload.py [--batch N] [--csv f]` | pre-commit checks + per-model max/avg PSNR |
| `export_db_models.py --prefix P --apply` | recover model sources from the DB |
| `export_db_stats.py --prefix P --apply` | rebuild stat dirs from `stat` + `prm` |
| `dnseed_retrain.sh` | re-train for genuinely missing epoch curves |

**Dedup key is `arch_uid2`**, the project's definition of distinctness — not filename, not code hash
(which misses structurally identical models differing by whitespace/ordering). Verified: the DB's
`arch_uid` column *is* `arch_uid2` (recomputed 25/25 matches). Models absent from the DB get their
uid from `docs/arch_uid2_computed.json`. Without dedup the corpus would ship ~136 redundant
architectures — one `arch_uid2` group holds **49** copies of the same architecture.

`validate_upload.py` checks: source parses and defines `Net`; epoch files contiguous `1..epoch_max`
(taken from the data, not assumed); each file a non-empty list carrying `metric_psnr`/`accuracy`;
max/avg PSNR computable and max agreeing with `best_accuracy`. Batch 1 passed 500/500.

---

## 6. Open items for the next push

**Batches 2–5 plans are STALE.** After they were written, 376 DnSeedGA models were exported and
their retrain started. Delete `docs/upload_batches/batch_00{2,3,4,5}.txt` and the matching ledger
entries, then re-plan. Batch 1's 500 are ledger-locked and will not be re-offered.

**DnSeedGA — 376 models, why they needed a retrain.** The strongest arm we have (best AWGN PSNR
36.98; best zero-shot SIDD transfer 32.59 dB vs 27.31 for LLM; **37.842 dB trained on SIDD-Medium,
+3.66 dB over published DenoiseLite**; lowest latency 1.66 ms p50, 3% off-NPU). They had no `.py`
and no stat dirs — DB only. Sources were exported; stats could **not** be, because the DB kept a
single row at epoch 5 for every one of the 376 (zero have intermediate epochs), while all 1,927
existing DenoiseGA dirs hold a full 1..5 curve. `dnseed_retrain.sh` re-measures them; early results
confirm all 5 epoch files per model. When it finishes: swap the staged dirs in
`sweep_out/dnseed_retrain/stat_train/` over the single-file ones, re-validate, re-plan.

**Two decisions still needed:**

1. **Ordering.** Batch 1 ordered by the DB's 5-epoch PSNR. DnSeedGA sits mid-pack there (~33.7 dB)
   but holds the corpus's best converged numbers — which live in `sweep_out/fulltrain50`, *not* the
   DB. Order later batches by converged PSNR where available, so the strongest models go out next
   rather than last?
2. **`beats_noop`.** 196 models never beat the 33.021 dB no-op floor (166 of them in batch 5).
   Include with the flag set — the schema exists for exactly this, and negative examples are useful
   for predictor training — or hold them back? Note the DB flags only 84 as `beats_noop=0` while the
   epoch stats show 196, so that column is under-populated and worth backfilling.

**Also outstanding:** 31 code-hash duplicate groups in `ab/nn/nn/` from an earlier audit, never
cleaned up.

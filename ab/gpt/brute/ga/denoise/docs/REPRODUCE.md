# Reproducing the denoise corpus and the deployability gate

Written 2026-11-09. Scope is denoise only.

## 1. What lives where

| what | repository / release | path |
|---|---|---|
| the corpus — model sources + per-epoch training stats | `ABrain-One/nn-dataset`, branch `add-denoise-models-and-stats` | `ab/nn/nn/<Model>.py`, `ab/nn/stat/train/img-denoising_denoise_psnr_<Model>/` |
| data loaders and metrics the experiments use | same | `ab/nn/loader/`, `ab/nn/metric/` |
| the generators (GA + LLM), export/upload tooling, the gate | `ABrain-One/nn-gpt`, branch `brute-skip-operator-and-cli` | `ab/gpt/brute/ga/denoise/` |
| paper-1 analysis + figure sources | same | `ab/gpt/brute/ga/denoise/paper1/` |
| measured data behind every table (latency, features, labels, Pareto, zero-cost, gate) | figshare, `lemur-denoise-artifact` | — |
| run artifacts, checkpoints, private DB copies | **not released** — `sweep_out/`, ~114 GB | — |

After the pending push the branch holds **3,957 distinct architectures**, which is exactly the
corpus the manuscript describes: 3,155 already on the branch plus the 802 in batches 7–8.

## 2. Corpus accounting — all 5,100 denoise models on disk

`upload_batches.py --status` is the authority. It reconciles git, the ledger, the DB and the
filesystem, and every model falls into exactly one bucket:

```
3155  already uploaded
 802  READY TO UPLOAD (distinct architectures)
 517  no arch_uid2 (not in the DB and not in docs/arch_uid2_computed.json)
 310  below the 33.021 dB no-op floor
 217  duplicate arch_uid of an already-uploaded model
  98  no training stats -- inert in LEMUR
   1  duplicate arch_uid within this upload
----
5100
```

Why each is held back:

- **no arch_uid2 (517).** Distinctness is `arch_uid2` and nothing else, so a model whose identity
  cannot be computed cannot be shown to be distinct. These are recoverable: `arch_uid2.py` works on
  source text and so applies even to models that will not import. Running it over these 517 is the
  single largest remaining addition to the corpus.
- **below the no-op floor (310).** 33.021 dB is what the identity function scores on this
  validation set at the 256-crop Y metric — copy the noisy input to the output and you get that. A
  model under it has learned something actively harmful; 19 of them score below 20 dB. They are
  held out by a **stated criterion** rather than quietly dropped, and
  `upload_batches.py --plan --include-below-noop` ships them if negative examples are wanted for
  predictor training. `docs/upload_batches/below_noop_plan.txt` is that plan, already written (renamed from
  `batch_009.txt` so a future `--plan` cannot overwrite it).
- **duplicate arch_uid (217 + 1).** Structurally identical to something already on the branch.
  Code-hash dedup would miss these; they differ only in whitespace or statement order.
- **no training stats (98).** A model directory with no stat dir is inert in LEMUR — it imports
  as an architecture with no measurement attached.

## 3. Known quality caveats in the pending 802

- **281 of the 802 ship a single epoch file instead of a 1..5 curve** (134 in batch 7, 41 in batch
  8 before the one drop, and 106 of the held-back batch 9). The stat is valid and LEMUR reads the
  budget from `max(stat.epoch)`, not `prm.epoch_max`, so they import correctly — but they carry no
  learning curve. `dnseed_retrain.sh` re-measures models in this state.
- **11 models in the held-back batch 9 have `max PSNR` disagreeing with `best_accuracy`** by 0.01
  to 0.05 dB. That column is under-populated in the DB; the per-epoch files are authoritative.
- **16 models in batch 8 sit exactly at 33.021 dB**, i.e. at the floor, not below it. They pass the
  `< floor` test and are included.
- `ab/nn/nn/DenoiseUnet{3F,6F,8F,9F}.py` and `LiteDenoiseNet.py` carry an uncommitted
  BN-poisoning guard. A code review raised a 0.0-loss finding against it and this is the code that
  *writes the stats*, so it stays out of any stats-bearing push until that is settled.

## 4. Importing the corpus

Stat directories are `<task>_<dataset>_<metric>_<model>` and `ab/nn/util/Util.py:61` unpacks them
with a positional 4-way `split('_')`. Every one of ours is
`img-denoising_denoise_psnr_<Model>`, so the corpus needs only `ab/nn/loader/denoise.py` and
`ab/nn/metric/psnr.py`, both already upstream. One underscore inside any field aborts the entire
import, not just the bad model — `python3 naming.py --repo <nn-dataset>` is the pre-push gate and
must exit 0.

The other loaders and metrics in this push serve the transfer and full-resolution experiments:
`awgn-color.py` and `denoise-awgn.py` (synthetic AWGN), `sidd.py` and `sidd-bench.py` (real
sensor noise; the loader documents the download URL rather than shipping 460 MB of `.mat`),
`psnrrgb.py` and `ssimrgb.py` (full-resolution RGB, where the no-op floor is 30.88 dB).

## 5. Regenerating the gate

1. Features come from `paper1/analysis/` or, released, `gate/extract_features.py` in the artifact.
2. `paper1/analysis/gate_refit_heldout.py` fits it: IRLS logistic, L2 = 2.0, SEED = 20261103,
   5-fold OOF, 5,000 bootstrap. The frozen coefficients are `docs/deploy_gate_model.json` — do not
   refit in place; that file is what every reported number was computed against.
3. `paper1/analysis/logo_gate.py` is the out-of-domain check: trained on every producer arm but
   one, tested on the arm it has never seen.

`paper1/README.md` maps every script to the artifact file it produces.

## 6. Pushing

`docs/GIT_UPLOAD.md` is the runbook. The two rules that matter: never `git add -A` or
`git commit -a` in the nn-dataset tree (it is a pruned copy — 252,223 tracked files are missing
from disk and either command stages them as deletions), and never run
`upload_batches.py --confirm N` after a push that printed `Everything up-to-date`.

# Plan: cross-dataset validity, skip ablation, and a publishable article

Written 2026-08-05. Supersedes ad-hoc sequencing. Every cost below assumes **8 GPUs max**
(user constraint) and ~38 min per 50-epoch MAI train (5-ep median 209 s x10, plus overhead).

## The premise this plan does NOT assume

The stated goal is "our MAI models should perform good on MIDD and SIDD too." That is a
hypothesis with an unfavourable prior, and the plan is built to *test* it rather than to
demonstrate it:

- MAI denoising data is JPEG-compressed smartphone-pipeline imagery; MIDD and SIDD are real
  sensor noise. A denoiser trained at one noise characteristic degrades sharply at another
  — this is the standard blind-vs-non-blind result, not a defect of our models.
- Selection pressure makes it worse, not better. 2,822 architectures were *selected* on MAI
  scores. Anything MAI-specific in the degradation has been actively optimised into the
  population.
- Therefore the expected zero-shot outcome is a large absolute PSNR drop on both datasets.

Two claims are defensible and worth the compute; one is not:

| claim | defensible? |
|---|---|
| The MAI *ranking* predicts the MIDD/SIDD ranking (search space transfers) | yes, measurable |
| Retrained on MIDD/SIDD, our architectures are competitive at matched MACs | yes, measurable |
| Our MAI-trained models are good denoisers on MIDD/SIDD as-is | **probably false — do not build the paper on it** |

If Phase 1 shows zero-shot collapse, that is a *result* ("architecture search transfers,
weights do not"), not a setback. The paper must be written so either outcome is publishable.

## Phase 0 — convergence ground truth (RUNNING)

`ft50.json`, 8 shards x ~38 models, `fulltrain50_shard.sh`, 300 stratified models at 50 ep.
**~22-25 h.** Resumable per model; pod-local DBs merged serially afterwards.

Why first: the corpus is a 5-epoch corpus and 5-ep recovers 3/10 of the converged top-10
(Spearman 0.68, n=56). Every claim downstream inherits that.

**Gate.** Report rho(5ep -> 50ep) at n=300 and top-k overlap.
- rho >= 0.8 -> the 5-ep corpus is a usable proxy; the full 2,822 can be cited as a search
  space with the 300 as its converged core.
- rho < 0.5 -> the 5-ep leaderboard must never be published as a ranking, and the paper's
  headline number becomes the 300-model converged set.

## Phase 1 — zero-shot transfer probe (CHEAP, run alongside Phase 0)

Inference only, using the **1,660 existing `best_model.pth` checkpoints**. 1 GPU, a few
hours, no contention with Phase 0's 8.

- Evaluate every checkpointed model on **held-out MIDD sensors** (train on sensors A-N,
  never touch the held-out ones — otherwise this measures sensor memorisation).
- Report: absolute PSNR drop MAI -> MIDD, and Spearman rho between the MAI ranking and the
  MIDD zero-shot ranking.
- Add SIDD to the same probe once Phase 2 lands.

This de-risks Phase 3 for almost nothing: if the ranking correlation is near zero
zero-shot, Phase 3's retraining design matters much more and should be scoped up.

Caveat to state in any writeup: 1,660 of 2,822 models have weights, and they are the older
GA cohorts. The probe is a large sample, not the whole corpus.

## Phase 2 — SIDD acquisition (NO GPU, start immediately, parallel)

`data/sidd` does not exist. Required before any SIDD claim:

1. Download SIDD-Medium sRGB (~20 GB).
2. Convert with `data/sidd_to_pairs.py` into `<SIDD_ROOT>/{original,denoised}`.
3. Verify pair counts match; the loader raises on mismatch.
4. **Use the standard SIDD protocol** (SIDD-Medium train, SIDD validation/benchmark patches
   for test). `ab/nn/loader/sidd.py` currently does its own deterministic 10% split of a
   320-pair pool — fine for internal comparison, **not** comparable to published numbers.
   Reviewers will check this. Either match the official split or state the deviation loudly.

## Phase 3 — cross-dataset retraining (Task 1 proper)

150-model stratified subset (reuse the Phase 0 selector) x {MIDD, SIDD} at 50 epochs.
**~24 h on 8 GPUs.** Deliverables:

- Converged per-dataset leaderboards.
- rho(MAI, MIDD), rho(MAI, SIDD), rho(MIDD, SIDD) — the actual transfer result.
- Cross-sensor holdout on MIDD, disclosed as a subset deviation from the MIDD paper's
  full-set protocol (the loader docstring already flags this; the paper must too).

## Phase 4 — skip-connection ablation (Task 2)

**Cannot be mined from the corpus.** Observational reads are confounded and actively
misleading: global residual correlates with *lower* score (-0.56 dB), because residual
models here are systematically smaller; the concat "effect" is +0.70 dB in the mean but
+0.01 dB at the median, i.e. entirely a broken-tail artifact.

Controlled design, generated fresh:

- **6-8 base architectures** spanning families (U-Net, NAFNet-lite, Restormer-lite,
  SplitterNet, flat-dilated, LapPyr).
- **Skip grid:** none / global residual only / long encoder-decoder concat / short
  intra-block residual / dense / global+long. Crossed with placement: early-only,
  late-only, all levels.
- **MAC-matched:** compensate width so every variant stays within ~5% of its base.
  Otherwise the experiment ablates capacity and mislabels it topology.
- **3 seeds**, 50 epochs. ~144 runs, **~12 h on 8 GPUs.**

Pre-registered hypothesis: the global input->output residual dominates; internal skip
placement is second-order. State it before running so the result is a test, not a story.

## Phase 5 — baselines (required for any venue)

DnCNN, FFDNet, NAFNet, Restormer-lite at matched MACs, on all three datasets, 50 ep.
~20-30 runs, **~3 h.** Without this table the paper has no anchor and will be desk-rejected
by any denoising reviewer.

## Phase 6 — on-device benchmark (NO GPU, parallel)

2,511 converted TFLite artifacts already exist. adb loop with `benchmark_model` + NNAPI
delegate on the Dimensity 7300; record per-model latency and delegation coverage.

This is the **genuine differentiator**. NAS-Bench-101/201/301 and HW-NAS-Bench have no real
phone-NPU denoising numbers. Everything else in this plan makes the work correct; this is
what makes it novel.

## Phase 7 — article

Honest assessment of venue:

- **CVPR main track**: not on current evidence, and not on this plan alone. Reviewers will
  want converged training (Phase 0/3), multi-dataset (Phase 1/3), baselines (Phase 5),
  multi-seed, and a novelty story stronger than "we generated many architectures."
- **MAI workshop @ CVPR** — the natural home. MIDD is Flepp/Ignatov CVPR 2024; the
  edge-MAC-cap + on-device-latency framing is exactly its scope.
- **NTIRE** — second fit.
- **NeurIPS Datasets & Benchmarks** — the right structural match for a benchmark paper,
  if the corpus (not the method) is the contribution.

Paper skeleton and which phase feeds each element:

| element | from |
|---|---|
| converged leaderboard, 300 models | Phase 0 |
| "does 5-ep proxy work" (methodological contribution) | Phase 0 gate |
| weights-vs-architecture transfer | Phase 1 |
| cross-dataset ranking transfer | Phase 3 |
| skip-topology causal figure | Phase 4 |
| comparison to published denoisers | Phase 5 |
| on-device latency + delegation coverage | Phase 6 |
| LLM-generation ablation (corpus as SFT data) | NOT YET DESIGNED — see below |

**Unbuilt and needed if the LLM angle is claimed:** a controlled experiment showing a model
fine-tuned on this corpus generates better architectures than the base model. Run 3's 20
cycles produced 89 kept models but no baseline arm, so it currently proves nothing about
the fine-tuning claim. Either design that arm or drop the claim from the paper.

## Wall clock

| phase | GPU | wall clock | can overlap? |
|---|---|---|---|
| 0 convergence | 8 | 22-25 h | running |
| 1 zero-shot probe | 1 | 3-4 h | yes, alongside 0 |
| 2 SIDD acquisition | 0 | download-bound | yes |
| 3 cross-dataset retrain | 8 | ~24 h | after 0 |
| 4 skip ablation | 8 | ~12 h | after 3 |
| 5 baselines | 8 | ~3 h | after 4 |
| 6 phone benchmark | 0 | ~1 day manual | any time |

Roughly **3 days of continuous 8-GPU compute** plus acquisition and device time.

## Standing risks

1. Publishing any 5-epoch ranking before the Phase 0 gate.
2. Single-seed claims anywhere in the top band (0.14 dB noise vs 0.69 dB top-100 spread).
3. SIDD split non-standard -> numbers not comparable to literature.
4. MIDD sensor subset -> must be disclosed, not buried.
5. Claiming universal MAI->MIDD/SIDD quality without the Phase 1/3 evidence.

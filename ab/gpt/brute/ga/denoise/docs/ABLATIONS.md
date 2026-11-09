# Ablations — results, sample sizes, and what each one cannot say

Every number below is measured. Where an earlier claim was superseded, **both the old and the
corrected value are kept**, because in two cases the size and direction of the error is itself the
result. Read the "cannot say" line of each section before quoting it.

Last updated 2026-08-07.

---

## 0. Reading rules

1. **Quote the right noise floor.** 0.207/0.231/0.182 dB (AWGN/MIDD/SIDD, `psnrrgb`, 30 ep) for
   cross-dataset work; **0.142 dB** pooled within-cell for the skip factorial (`psnr`, 50 ep).
2. **33.021 dB is the no-op.** Any run at exactly that value is degenerate, not a data point.
3. **Sequential stages need conditional denominators.** See `LLM.md §4`.
4. **Single-draw correlations at n≈36 are not reproducible.** See §1 → §2, where ρ +0.79 became a
   1.90 dB regret claim that repeats then reduced to 0.27 dB.

---

## 1. Cross-dataset transfer (COMPLETE — superseded by §2)

120/120 cells, 0 failures. 40 models × {AWGN, MIDD, SIDD}, single draw each.

- Single-draw Spearman: **+0.790 / +0.779 / +0.886** at n=36.
- Restricted to the top-10: collapses to **+0.248 / −0.055**.
- Regret@1: **1.90 dB**.
- Health: 26/40 score higher on real noise than on synthetic. One domain failure —
  `DnSeedGA-279df2ade`: 28.62 dB AWGN → **10.46 MIDD / 16.79 SIDD**.

**Cannot say:** anything about reproducibility — one run per cell. The top-10 collapse was read as
"transfer is weak"; §2 shows that reading was itself mostly winner's curse.

---

## 2. Cross-dataset confirmation (COMPLETE — this supersedes §1)

72/72 cells, 0 failures. 8 models (union of top-5 by each dataset) × 3 datasets × **3 repeats**.
`confirm_select.py` / `confirm_train.sh` / `confirm_report.py`, `k8s/confirm-train-2.yaml`.
Triples are interleaved as `(model, dataset, repeat)` so each shard gets a balanced dataset mix —
fixing the 40-model run's one-dataset-per-shard flaw.

| quantity | value |
|---|---|
| top-8 Spearman | **+0.929** (corrected +1.034 — indistinguishable from perfect) |
| regret@1 | **0.27 dB**, CI **[0.00, 1.01]** |
| regret@3 | 0.07 dB |
| regret@5 | 0.00 dB |
| AWGN's #1 pick changes in | **63% of resamples** |
| noise floor AWGN / MIDD / SIDD | 0.207 / 0.231 / 0.182 dB |

`LLM_Pg33_seeded-8a7b5df` fell **35.83 → 34.96 dB** on repeat.

**Conclusion:** ranking on synthetic noise costs ~0.27 dB with a CI including zero. Selecting the
top-3 rather than the top-1 removes essentially all of it. **The earlier 1.90 dB figure was
winner's curse and must not be quoted.**

---

## 3. Skip-connection factorial (COMPLETE — 267/267 runs, 0 failures; G1 arm VOID)

`ablation_factorial.py`, report at `sweep_out/ablation/train/factorial_report.json`.
Within-base paired contrasts, 96 runs, 7 clean bases.

| factor | observational | **controlled** | 95% CI | per-base agreement |
|---|---|---|---|---|
| Lateral skip (L) | +0.531 | **−0.159** | [−0.248, −0.070] | **7/7** |
| Block residual (B) | −0.155 | **+0.148** | [+0.007, +0.288] | 6/7 |
| L × B | — | −0.103 | [−0.374, +0.169] | not resolved |

**Both observational signs REVERSED.** The corpus-level correlation tracked depth and width, which
covary with topology; holding the base fixed inverts it.

- Pooled within-cell sd **0.142 dB**; MDE **0.043 dB** — both effects clear it 3–4×.
- Cells, centred: G0B1L0 **+0.179** (best), G0B0L0 −0.020, G0B1L1 −0.031, G0B0L1 **−0.128** (worst).
  Span **0.307 dB**.
- Lateral connections change MACs by **0.3%** → the cost is topology, not compute.
- `DnSeedGA-146bae38` excluded before analysis: repeat sd **10.69 dB** (unstable base, not an effect).

### The G1 arm is void — codegen bug, not a finding

**All 96 G1 runs landed on exactly 33.021 dB.** The bases already ended
`return clamp(out_conv(d0) + identity, 0, 1)`; the G1 transform wrapped it as
`return x + clamp(out_conv(d0) + identity, 0, 1)`. That doubled residual makes the identity an
attractor. **No claim about global skips is supported.** Regenerating the transform to *replace*
rather than re-add the residual is required to answer it.

**Cannot say:** anything about global skips. All 267 runs are now in and the numbers above are
FINAL — re-running the analysis on the complete set reproduced the interim estimates exactly
(L −0.159, B +0.148), so the extra runs did not move them.

---

## 4. Prompt-engineering ablation (COMPLETE, retrospective)

`ablation_prompt.py`. 1,752 candidates, ~425 per style, from `llmprod`. Full detail in `LLM.md §5`.

- **Yield is flat**: 13.3–15.9%, **none of six pairwise comparisons resolves**. Reasonably powered
  null, not an underpowered shrug.
- **Six conditional stage effects resolve** — style changes *where* candidates die, not how many
  survive. Step-by-step: +14.9 pp forward failures, −8.1 pp exact duplicates. Constraint-based:
  −7.8 pp forward failures. Conversational: +7.1 pp duplicates. Direct instruction: +4.8 pp over the
  MMac cap.
- **Novelty by cosine distance is flat** (median nearest_cos ≈0.92, nothing resolves). This coexists
  with the resolved exact-duplicate difference: `arch_uid` collision is exact structural identity,
  cosine is a graded embedding distance. The exact-identity result is better powered — lead with it.

**Cannot say:** which style produces better or faster models. `db_name` is null for all 249 screened
rows (nothing converged, nothing profiled), and the 1-epoch screen has no range — median 32.72 dB
against the 33.021 no-op, only 19 of 238 above it.

**To close it:** a *fresh* generation on a **fixed common seed set** — the existing pool cannot be
paired, only **2 of 167** seeds have a survivor in all four styles and 138 are style-exclusive, so
seed is confounded with style. ~20 seeds × 4 styles, then train at 30 ep and profile. Against the
0.142 dB sd, 8 models per style resolves 0.3 dB.

---

## 5. On-device / Qualcomm AI Hub (552 profiled)

| producer | n | p50 | p90 | over 10 ms | fully off NPU |
|---|---|---|---|---|---|
| LLM episodes | 332 | 42.94 ms | 72.11 ms | 57% | **38%** |
| GA main (DenoiseGA) | 139 | 3.38 ms | 84.95 ms | 45% | 4% |
| Hand-written | 50 | 3.85 ms | 50.90 ms | 14% | 6% |
| GA seeded (DnSeedGA) | 31 | **1.66 ms** | 5.96 ms | 3% | 3% |

- **Off-NPU penalty 43×** (1.66 → 70.98 ms).
- Corpus-weighted **≈41%** would not ship (raw 47% over the LLM-first sample; weights
  2,049 DenoiseGA / 470 LLM / 376 DnSeedGA / 63 hand-written). **Every one passes the 9,053 MMac cap** —
  MACs encode neither operator support nor the cost of depth.
- **Two distinct failure modes**: LLM models are slow because 38% fall off the NPU; DenoiseGA models
  are slow from depth/width with only 4% off-NPU — invisible to operator signatures.
- Generational v75→v81 speedup **1.63×**, ρ=0.9919, **0/82 slower**.
- **ρ(PSNR, latency) ≈ −0.03** — accuracy and speed are essentially uncorrelated in this corpus.
- `LiteDenoiseNet`: **0.88 ms** S26U / 1.12 ms S24, 100% NPU, 53 layers, score 591, **rank 3 of 129**.

**Do not quote a corpus median latency from this sample.** p50 reads 1.72 ms at n=128 and 8.24 ms at
n=552 — a *sampling* artifact, because the bulk batch was submitted LLM-first (332 of 552 are LLM,
the slowest arm, against 470 LLM in a 2,958-model corpus). Use per-family medians.

### Anchor against published numbers

`LiteDenoiseNet` ours **36.511 dB** (Y, 256 crop, no distillation) vs published **37.66 val /
37.58 test** (distilled from a 41.6M teacher). Our best GA architecture **37.17 dB** beats our own
LiteDenoiseNet by 0.66 dB under matched training.

**Cannot say:** that we beat the published model — the training recipes differ (no distillation).
The matched-recipe comparison is the defensible one.

---

## 6. Predictor validation (COMPLETE)

See `LLM.md §7` for the table. Headline: the in-house surrogate is **ρ = −0.178** against truth
(n=326) while claiming +0.693 by its own CV; **epoch-3 measured PSNR (+0.728, 82% top-50) beats
every learned predictor** and is what selection uses. `beats_baseline: false` for the fine-tune on
selection; it is used only for calibration (MAE 0.509 dB).

Partial correlation ρ(stock prediction, final | epoch-3) = **+0.056** — the stock predictor adds
essentially nothing once epoch-3 is known.

---

## 7. Repair-pass validation (COMPLETE — the new pass did NOT beat the old one)

| pass | fix rate on 120 real `forward` failures |
|---|---|
| old `preflight.repair_channels` (regex over the error string) | 2/120 = **1.7%** |
| new `repair_shapes.py` (hooks + module surgery), budget 12 | 1/120 = 0.8% |
| new `repair_shapes.py`, budget 48 | 2/120 = 1.7% |
| new `repair_shapes.py`, budget 48 + correct `IN_SHAPE` | 3/120 = **2.5%** |

**No improvement.** The rewrite is architecturally better — no error-string parsing, indifferent to
kwargs/variables/nesting, verifies the emitted patch from source, and explains every decline — but it
targets a class that barely exists in this population.

### The sizing was wrong, and this is the correction

`sweep_out/ablation/forward_census.json` (n=434) puts "conv `in_channels` vs actual input" at
**44%** of forward failures. **That census is a DIFFERENT POPULATION** — the ablation's generated
variants, 2026-08-05 — not `llmprod`. Measured directly on llmprod's own failures:

| class | llmprod (n=120) | ablation census (n=434) |
|---|---|---|
| elementwise add/mul shape mismatch | **49.2%** | 16.1% |
| concat non-channel dim mismatch | **14.2%** | 5.1% |
| missing attribute (module never created) | 9.2% | 8.8% |
| conv channel mismatch | **4.2%** | **44.0%** |
| view/reshape wrong element count | 4.2% | 0.7% |
| other TypeError / RuntimeError / resource | 18.9% | 30.5% |

**On llmprod, 63.4% of forward failures are elementwise + concat — precisely the classes the pass
excludes by design.** Conv-channel is **4.2%**, i.e. 5 candidates in 120.

An `IN_SHAPE` BUG INFLATED THAT FIGURE AND WAS FIXED. `repair_shapes.py` initially passed an
unbatched `(3, 256, 256)`. `in_shape` is BATCHED `(N, C, H, W)` (`Train.py:210`, documented as
"(8, 3, 32, 32) for a batch size 8") and **1,055 of 2,958 DB models read `in_shape[1]`** as their
channel count, so the unbatched tuple made them build a 256-channel stem — manufacturing conv
mismatches that do not exist under the real harness. With the correct shape, conv-channel drops
9.2% -> 4.2% and the repaired count rises 2 -> 3 of 120. `preflight.py:50` was always correct
`(1, 3, 256, 256)`, so **the funnel numbers elsewhere in these docs are unaffected**.

### Corrected yield projection

An earlier estimate in this project claimed channel repair would take yield 14.7% → ~20%. **That
was computed from the wrong census and is wrong.** Perfect repair of the conv class is
5/120 × 739 ≈ 31 candidates × 30.9% downstream conversion ≈ **10 extra passes → 14.7% → 15.3%**.

**Moving yield materially requires handling elementwise/concat alignment**, which means inserting
1×1 projections or interpolation — layers the generator did not ask for. That is a topology change
and a product decision, not a bug fix.

### DECISION 2026-08-07: channel repair is ABANDONED. Do not revisit.

`repair_shapes.py` is **not wired into `preflight.py` or `run3_evolve.sh`** and must not be. It is
kept as a validated negative result and a diagnostic tool (`--list`, `--out`), nothing more.

Rationale, all measured:

- It barely beats the pass it replaces: **2.5% vs 1.7%** (3 candidates vs 2, out of 120).
- The target class is only **4.2%** of llmprod forward failures — 5 candidates in 120.
- Forward-failure rate is **flat across 20 SFT cycles** (63.8% vs 64.7%, not resolved) and **flat
  across all four prompt styles** (42–59% conditional). It is a property of the base model's code
  generation, not of anything in our control.
- **Generation is the cheap step.** At 14.7% yield you generate 6.8 candidates per usable model;
  perfect channel repair makes that 6.3. Generating more candidates costs no GPU training and
  strictly dominates engineering a repair.

Where yield effort should go instead: the adapter (cycle12 → cycle10, a resolved +4.7 pp, already
applied), the MMac number written literally into the prompt (67 rejections), and diversity-aware
prompting to reduce the 375 dup-gate losses **without relaxing the gate**.

### Bugs found and fixed while validating (keep these)

1. **`MAX_FIXES` was 12; measured need is up to 20.** A channel change at the stem cascades through
   every U-Net stage, so fix count scales with depth. Now 48. This converted one candidate.
2. **Declines never carried the partial fix list**, so a candidate that applied 60 fixes reported
   `n_fixes: 0` and every decline was undiagnosable. All return paths now go through `_decline()`.
3. **No oscillation detection.** A cascade revisiting a path is not converging; it now stops and
   says so instead of silently burning the budget.
4. **In-process execution was fatal.** The first run was SIGKILLed at candidate 26 of 120 with no
   traceback and no output — a generated net allocating enormous weights at construction. Each
   candidate now runs in a subprocess with `RLIMIT_AS` and a timeout, results appended to `.jsonl`
   as they land, so one bad candidate costs one row instead of the whole run.

---

## 8. Still open

| item | state |
|---|---|

| 720×480 + 1088×1920 re-export | `reexport-gpu` running; verdict box on the artifact's On-device page is gated on it |
| Bulk profiling | 431/800 |
| Latency predictor | 552 labels available, **unfitted** — needs held-out validation vs a baseline |
| KTO reward → MAI Final Score | needs the H100 |
| G1 factorial arm | needs regeneration (see §3) |
| 31 code-hash duplicate groups in `ab/nn/nn/` | cleanup before publishing |
| `julia2.md` CVL/K8s runbook | out of date |

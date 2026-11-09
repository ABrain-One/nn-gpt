# Seed-set ledger — which seeds fed which run

Append a block per run. The seed pool is the one thing that changes between runs and is
*not* recoverable afterwards from the outputs alone: the GA writes candidates, not its
inputs, and the LLM corpus is derived at runtime from whatever the DB happened to hold.

Rules that earned their place here:

- **Record measured MACs, never inherited ones.** Every "edge-legal" claim before
  2026-07-30 traced back to a code comment, not a measurement in this tree. Three separate
  mechanisms each made an unverified seed look legal — see the MAC section below.
- **Record the dB definition.** `last5` (mean of the final 5 epochs) and `best` (max over
  50) rank the corpus differently; `best` rewards volatility.
- **Record the selection rule, not just the names.** "Top 10" is meaningless without
  saying top by what, and whether family duplicates were collapsed.

---

## MAC ground truth (measured 2026-07-30, in-container, cap 9053 MMac)

First direct measurement from this tree. Everything prior was quoted from
`run_seed_ga.py`'s `DEFAULT_SEEDS` comment.

| model | MMac | verdict |
|---|---:|---|
| DenoiseUnet5F | 1463 | OK |
| DenoiseUnet8F | 2737 | OK |
| DenoiseUnet6F | 3272 | OK |
| LiteDenoiseNet | 3530 | OK |
| DenoiseUnet1F | 3944 | OK |
| DenoiseUnet3F | 4958 | OK |
| **DenoiseUnet9F** | **9053** | OK — **0.4 MMac of headroom** |
| DenoiseUnet4F | 12585 | **OVER CAP (1.4x)** |
| DenoiseUnet2F | 14005 | OVER CAP (1.5x) |
| DenoiseUnet10F | 29054 | OVER CAP (3.2x) |
| DenoiseUnet7F | 30152 | OVER CAP (3.3x) |

Two things to carry forward:

1. **DenoiseUnet4F is over cap and was not on any denylist.** The comment named only
   2F/7F/10F, so `KNOWN_OVER_CAP` in `llm_denoise.py` named only those three, and the
   "just use the F variants" rule admitted 4F. Add it.
2. **DenoiseUnet9F sits at 99.996% of budget.** The cap was almost certainly derived from
   it. Its lineage can only shrink — any mutation that grows it is rejected pre-training.
   It is the highest-PSNR seed (35.43 dB) and the one with no room to evolve. If 9F
   descendants get culled disproportionately, swap it for a seed with headroom.

Why this was not known sooner — three independent ways the cap silently passed everything:

- `_seed_is_edge_legal()` **fails open**: any exception returns `True`. Correct inside the
  GA (a thop hiccup must not empty the pool), indistinguishable from a clean pass.
- `measure_cost()` needs torch — absent on the submit node.
- `measure_cost()` lazily imports `mutator.lemur_io`, and the `mutator` package lives in
  `ab/gpt/brute/ast`, a different tree from `edge_cost.py`. Callers that add only the
  denoise dir to `sys.path` get `ModuleNotFoundError` for every model.

`mac_audit.py` exists for this and treats unmeasurable as **UNKNOWN, not as a pass**.

---

## Run 2026-07-30 — GA `denoise-ga-d10v2`

- **Job** `k8s/denoise-ga-d10v2.json` → `launch_distinct10_v2.sh`
- **Seed list** `BEST10_V2_DB` in `seeds_distinct.sh`
- **Config** pop 50, gens 30, epoch 5, crossover block, skip-ops 0.3, elitism 5,
  tournament 3, `--edge-filter --mac-cap 9053 --diversity-gate --save-to-db`
- **Selection rule** highest `last5` dB, **one model per architectural family** (the
  `DenoiseUnet*F` lineage collapsed to its best member, 9F)

| # | seed | last5 dB | MMac | headroom |
|---|---|---:|---:|---:|
| 1 | DenoiseUnet9F | 35.43 | 9053 | **0.0%** |
| 2 | DenoiseSubPixel | 35.25 | 4247 | 53% |
| 3 | DenoiseDilatedFlat | 35.15 | 4162 | 54% |
| 4 | DenoiseLPPool | 35.12 | 4443 | 51% |
| 5 | DenoiseELUCtx | 35.11 | 3965 | 56% |
| 6 | DenoiseMBConv | 35.08 | 4812 | 47% |
| 7 | DenoisePoolDeconv | 35.04 | 5618 | 38% |
| 8 | DenoiseLinearMix | 35.00 | 4561 | 50% |
| 9 | DenoiseSelfAttn | 34.98 | 2112 | 77% |
| 10 | DenoiseThunder | 34.98 | 1274 | **86%** |

All 10 loaded, zero dropped by the seed MAC gate.

**Why not the pre-existing `BEST10_DB`.** It was written 2026-07-27 and states its own
limit: *"the rest await the populate_db pass for measured PSNR before they can be ranked
best."* That pass has since run and inverts it — the old list contains `DenoiseUnet3F`
(33.18), `6F` (33.76) and `1F` (34.37), the bottom three of 25, and excludes the top five.
It is also 6 variants of one U-Net lineage, not 10 families. `BEST10_DB` was left in place
for reproducibility of earlier runs; `BEST10_V2_DB` was added alongside.

**Do not over-read the ranking.** These 10 span 0.45 dB; measured within-model
epoch-to-epoch jitter is 0.14 dB. Ranks 2–10 are not separable. The justification for v2 is
"stop seeding with the measured-worst", not "SubPixel beats Thunder".

**`--epoch 5` is a liveness screen, not a quality ranking.** ep5→ep50 Spearman on this
corpus is **+0.263**, not the 0.90 assumed when that budget was set.

**Predictors off.** `--prof-predictor` measured ρ=−0.140 / MAE 0.553 dB at n=18 — 4.7x
worse than predicting the corpus mean. Its ρ=+0.84 at n=6 did not survive the larger
sample. The built-in `--predictor` is kept: it only re-ranks within a generation and falls
back to measurement.

---

## Run 2026-07-30 — LLM `denoise-llmgen-1`

- **Job** `k8s/denoise-llmgen.json` → `llm_gen_job.sh` → `llm_denoise.py`
- **Model** `Qwen/Qwen2.5-Coder-7B-Instruct` (32k ctx; the 4096-ctx configs in `conf/llm/`
  cannot hold the ~22k-char seeded prompt). Weights cached at `/a/mm/hf_cache`, 14.5 GB.
- **Arms** `cold` (no reference code) and `seeded` (1 primary + 3 supporting), n=5 each
- **Seed set** `--seed-set named` = the hand-written F seeds ∪ the 17 built families

**Final corpus: 24 references = 7 seeds + 17 families** (`run_id 20260730-220434`,
nn-gpt `b3be3648`, nn-dataset `b4af36b9de`). dB from the canonical 50-epoch sweep where
available.

seeds (7): DenoiseUnet9F, DenoiseUnet8F, DenoiseUnet5F, DenoiseUnet3F, DenoiseUnet1F,
DenoiseUnet6F, LiteDenoiseNet

families (17): DenoiseSubPixel, DenoiseDilatedFlat, DenoiseLPPool, DenoiseELUCtx,
DenoiseMBConv, DenoisePoolDeconv, DenoiseLinearMix, DenoiseSelfAttn, DenoiseThunder,
DenoiseGroupNorm, DenoiseChanShuffle, DenoiseSplitterNet, DenoiseMFDNet,
DenoiseSplitterNetF, DenoiseParallelMR, DenoiseReflectPad, DenoiseLapPyr

**RESOLVED — the live gate caught the one my denylist missed.** Launch intent was 25;
runtime produced 24:

```
[seeds]  DROP over-cap seed DenoiseUnet4F: 12585 MMac > 9053 cap
[seeds] dropped 1 over-cap seed(s): DenoiseUnet4F
```

No over-cap model reached the prompts, so this run's seeded arm is sound and does not need
redoing. Note what saved it: not the static `KNOWN_OVER_CAP` list (which named only
2F/7F/10F, per a code comment), but the *live* measurement — working here only because the
`mutator` sys.path fix landed before launch. Static lists transcribed from comments go
stale; the measurement is the authority. Still add 4F to `KNOWN_OVER_CAP` so environments
without a working cost backend are also protected.

Prompt configs, pinned by sha256 in `run_meta.json`:
`Denoise_gen_cold.json`, `Denoise_alter_seeded.json`. The TASK / CONSTRAINTS /
INTERFACE CONTRACT / OUTPUT blocks are byte-identical across the two arms and verified by
`--check-contract` before generation; the arms must differ **only** in the presence of
reference code or the ablation measures wording instead of conditioning.

Every prompt is saved: `cand/<id>/prompt_system.txt`, `prompt_user.txt`, `response_raw.txt`,
`candidate.py`, `meta.json`, plus an append-only `prompts.jsonl` and `run_meta.json`
carrying the LLM name, sampling params, argv, config sha256s and both repo git revisions.

---

---

## Diversity gate across runs — CORRECTION

An earlier note in `launch_distinct10_v2.sh` claimed v2 "differs from launch_distinct10.sh
only in the seed pool". **That is wrong.** v1 had no diversity gate at all; v2 adds one.

| launcher | gate |
|---|---|
| `launch_distinct10.sh` (v1) | none |
| `launch_distinct_full.sh` | none |
| `launch_production.sh` | none |
| `launch_production_v2.sh` | `--diversity-gate --cka-max 0.80` |
| `launch_distinct10_v2.sh` | `--diversity-gate` |

Only R3/R4/R5/WLVAL left `*_wlhashes.pt` stores, so only those ran gated. The
`A0/B0/C0/D0/BEAT1/P1/P2` model dirs have no store — those runs were **ungated**.

Consequences worth stating plainly:

- **`--cka-max 0.80` is inert.** `DiversityGate.__init__` documents its own signature:
  the CKA-era args "are accepted but unused by the WL gate". `launch_production_v2.sh`
  believed it gated at CKA 0.80; it did exact WL-hash matching like everything else.
- **No run has ever used `--diversity-graded`.** The standard everywhere is EXACT graph-hash
  novelty. That accepts "same architecture, rescaled channel widths" (cosine >=0.985).
- If the 131 `DenoiseGA-*` models came from the ungated launchers, their distinctness was
  never enforced at generation time — it was asserted afterwards by the offline cosine-0.95
  clustering recorded at the top of `seeds_distinct.sh`. Not the same claim.

`screen_llm_candidates.py` therefore gates on the exact hash too, and reports cosine as
advisory only, so LLM output is judged by the same bar as GA output.

Gate confirmed FIRING on generated children (run d10v2, gen 2):
`DIVERSITY-REJECT DenoiseELUCtx+m -> graph-clone`, and
`DenoiseSelfAttnXDenoiseSelfAttn` — a model crossed with itself, which can only be a clone.

---

## LLM run 20260730-220434 — real end-to-end yield

Parse rate was 9/10. **End-to-end yield was 1/10.** The gap is the point: "parsed" only
means a `class Net` was recoverable from the text.

| cause | n | whose fault |
|---|---:|---|
| built + forwarded + correct output shape | 1 | — (`0002_cold`, 1246 MMac, cos 0.757 to nearest) |
| genuine model errors | 6 | the LLM |
| `out_shape=(1000,)` harness trap | 2 | **ours** |
| markdown fences inside `<nn>` tags | 2 | **ours** (fixed; both then failed on model errors) |

Model errors seen: `F.clamp` (does not exist — it is `torch.clamp`), channel mismatch
32 vs 16, `conv2d()` handed a list, matmul 1x32 @ 64x4, `SyntaxError` mid-file, and a class
named `DenoiserNet` despite the prompt demanding `Net` twice.

### The out_shape trap (affects the GA too)

`mutator/lemur_io.py` injects classification defaults regardless of task:

```python
DEFAULT_IN_SHAPE  = (1, 3, 224, 224)
DEFAULT_OUT_SHAPE = (1000,)   # "Output shape for classification"
```

A denoiser that writes `out_shape[1]` — reasonable for image-to-image — gets
`IndexError: tuple index out of range`. None of our hand-written seeds index `out_shape`
(0 of them), so this has been invisible. **It is latent for the GA**: any mutation that
introduces `out_shape[1]` is scored DEAD for a harness reason, not an architectural one.
Fix belongs at a run boundary, not mid-run.

### Prompt defect (mine)

The INTERFACE CONTRACT block shows `in_shape=(1, 3, 512, 512)`, copied from
DenoiseUnet9F, while the TASK block states `[B, 3, 256, 256]` and the loader actually
passes `(1, 3, 224, 224)`. **Three different spatial sizes in one prompt.** Models told to
size layers from `in_shape` were set up to fail. Fix before the next LLM run, and keep the
edit identical in both arm configs or `--check-contract` will (correctly) refuse to run.

### Sizing

At ~20-30% build success, 2000 usable models needs ~7,000-10,000 prompts, not ~2,200.
Generation is cheap (16 s cold / 35 s seeded) so that is 15-25 GPU-hours — but the LLM
path is not the shortcut over the GA it first appeared to be.

---

## Changing the seeds next run

1. Re-measure — `python mac_audit.py` in-container. Never inherit a MAC number.
2. Re-rank on `last5` from the canonical sweep DBs, not the master DB (mixed recipes).
3. Collapse families before taking a top-N, or the list fills with one lineage.
4. Prefer seeds with MAC headroom for the GA; a seed at the cap can only shrink.
5. Append a block here **before** launching, and record the selection rule.

---

## Run: GA NORM-v3 — launched 2026-07-31 15:44 UTC (`denoise-ga-normv3`)

**Question it answers:** does giving the GA norm-diverse seeds plus a mobile-safe norm
mutation target move the normalisation axis at all? Baseline to beat is d10v2's
**88 norm-free / 1 BatchNorm out of 89** generated sources.

Two changes from d10v2, and only two, so the result is attributable:

1. **Seeds span five norm regimes.** d10v2's cause was never a missing operator —
   `layer_type` existed and ran at weight 0.15 — it was that `layer_type` can only *swap a
   norm that is already there*, and 9 of 10 seeds had no norm module at all. A no-op
   operator looks identical to a missing one in the output.
2. **`BRUTE_NORM_SAFE=1`** restricts norm targets to `{BatchNorm2d, Identity}` and
   `config.py` now lists `Identity` as both source and target, so the axis moves in **both**
   directions. Previously it was a one-way ratchet toward norm-free. `layer_type` weight
   raised 0.15 → 0.25.

Deliberately **not** `--edge-strict`: it would reject most descendants of the
LayerNorm/GroupNorm/InstanceNorm seeds, which is the very thing being measured. Observe
first, label deployability after.

Fresh `seed_ga_NORMv3_*` checkpoint/cache/models/results — reusing d10v2's paths would
resume that population at gen 8 with the old seeds and prove nothing.

### Seeds as *requested* (11) vs as *run* (9)

Two were dropped in-container by the seed audit:

```
DROP over-cap seed DenoiseUNet2:  54618 MMac > 9053 cap
DROP over-cap seed DenoiseUNet:   55692 MMac > 9053 cap
```

**This retracts a claim made in `seeds_distinct.sh` BEST10_V3.** DenoiseUNet2 was written up
there as "the BEST model measured (35.368 last10)" and edge-legal. It is ~6× the edge
budget. The check that cleared it was source-level — transpose padding and norm modules,
both of which it passes — and **a source-level read cannot see MACs**. Its 35.368 dB must
not be quoted as a target for edge models; it was never in that weight class. Rule going
forward: nothing enters a seed list until `mac_audit.py` has printed a number for it.

The norm balance survives the drop, because `DenoiseResBN` also carries BatchNorm:

| seed | norm | MMac | Mp |
|---|---|---|---|
| DenoiseSubPixel | none | 4246.7 | 0.26 |
| DenoiseLPPool | none | 4442.8 | 1.05 |
| DenoiseELUCtx | none | 3964.7 | 0.85 |
| DenoiseThunder | none | 1274.0 | 0.18 |
| DenoiseSelfAttn | none | 2111.8 | 0.35 |
| DenoiseGroupNorm | GroupNorm | 3756.0 | 0.73 |
| DenoiseResBN | BatchNorm | 3756.0 | 0.73 |
| DenoiseLayerNorm | LayerNorm | 3756.0 | 0.73 |
| DenoiseInstNorm | InstanceNorm | 3756.0 | 0.73 |

→ norm-free 5, BatchNorm 1, GroupNorm 1, LayerNorm 1, InstanceNorm 1. Budget (auto = median
seed) = 3756.0 MMac.

**Unplanned bonus the drop exposed:** the last four all measure 3756.0 MMac / 0.73 Mp. They
are one backbone with only the norm swapped — the cos 0.94–0.998 "norm clone" cluster that
`seeds_distinct.sh` deliberately collapsed down to GroupNorm for diversity reasons. For a
*norm* experiment that redundancy is the point: it is a controlled 4-way contrast where norm
is the only variable. Reintroducing them was accidentally the right call.

### Success criterion

Norm distribution over generated sources materially away from 88/1. Count constructor
calls, **not** `isinstance` lines — the first scan of d10v2 reported "100% BatchNorm" because
it matched `nn.BatchNorm2d` inside an `isinstance()` NaN-guard tuple; the true figure was the
opposite.

---

## Run: LLM generation `denoise-llmgen2` — completed 2026-07-31 07:39 UTC

2 pods, 1 GPU each, 4h19m wall.

| arm | prompts | parseable code | yield |
|---|---|---|---|
| cold (index 0) | 150 | 150 | 100% |
| seeded (index 1) | 400 | 366 | 92% |
| **total** | **550** | **516** | **94%** |

Prompt sizes: cold ~4.4–4.6 kch, seeded ~21–29 kch (the reference blocks dominate). Every
prompt, system prompt and raw response is archived per candidate under
`cand/<id>/{prompt_user,prompt_system,response_raw}.txt` — re-extraction never needs a GPU.

**"Parseable" ≠ "trains".** The smoke test is the calibration: of 4 candidates, 2 raised at
build/first-forward (`expected input[16,3,256,256] to have 32 channels`; `'Net' object has no
attribute 'device'`). Treat 94% as an upper bound and expect the real figure from the screen.

### Stage 1 screen — `denoise-llmscreen2`, launched 2026-07-31 15:47 UTC

8 shards × 1 GPU, **`--screen-only`**. Sharded runs are forbidden from writing to the DB —
`screen_and_train.py` now refuses `--num-shards>1` without `--screen-only`, because N writers
on the NFSv3-hosted SQLite file corrupt it rather than blocking. Survivors get full-trained
afterwards by a single pod.

Candidates are strided (`i % num_shards`), not blocked: ids are emitted in strategy
round-robin order, so contiguous blocks would hand each shard a skewed strategy mix and make
per-shard pass rates incomparable.

**The gate is a label here, not a filter.** `results_shard*.json` records ep1 for every
candidate that trains at all. The default 32.7 is the measured floor that retains all six
canonical families finishing ≥35 dB — but that was measured on 18 *hand-written* models, and
LLM output has a far wider spread (the one smoke survivor screened at 31.5, which 32.7 would
reject). Once this pass lands, the full ep1 distribution of all 516 is on disk and the
production gate can be chosen from data and re-applied for free.

### Screen aborted and relaunched — the `self.device` prompt defect (mine)

The first screen launch was killed ~2 min in. The first candidates on three separate shards
all died the same way:

```
ERROR - Evaluation Failed: 'Net' object has no attribute 'device'
```

**Cause: the INTERFACE CONTRACT told the model to USE `self.device` twice and never once to
ASSIGN it.** It showed `self.to(self.device)` in `__init__` and "move both to self.device" in
`learn()`. `nn.Module` has no `.device` attribute, so a model writing the natural PyTorch
line `self.to(device)` raises on the first batch and scores zero regardless of architecture.

Measured across all 516:

| arm | uses `self.device`, never assigns | assigns | never touches |
|---|---|---|---|
| cold | **122 / 150 (81.3%)** | 20 | 8 |
| seeded | 3 / 366 (0.8%) | 363 | — |

The arms differ that starkly only because the seeded arm sees reference code containing
`self.device = device`. The cold arm had nothing but the contract, and the contract was
wrong. **Screening as-is would have produced the headline "zero-reference prompting yields
5× more broken models" — a pure artefact of my prompt bug.** Any cold-vs-seeded yield number
from before this fix is void.

Also caught in the same sweep, and *not* worth acting on: only 12/516 index `out_shape[...]`
(the known `(1000,)` trap) and 1/516 has a syntax error. Neither is a systematic defect.

**Two fixes applied:**

1. `Denoise_shared_blocks.json` — the contract now states `self.device = device` is required
   as the first line after `super().__init__()`, and names the exact AttributeError. This
   changes the shared-block sha, which is recorded per candidate in `meta.json`.
2. `repair_device_attr.py` — inserts the missing line into the 125 affected candidates,
   after `super().__init__()` so it is legal. All 125 were repairable; none needed
   regeneration. Originals kept as `candidate.py.orig`, and `meta.json` gains `_repaired`,
   so repaired models can be excluded from any claim about *raw* prompt yield.

Repair, not regeneration, because the missing line is harness boilerplate: it changes no
layer, no topology, no MAC count. Discarding 122 architectures over a line the LLM was never
asked to write would be the wrong trade. Verified after: **0 / 516 still broken.**

### Second abort: shared LEMUR scratch dir (the sharding hazard nobody would have caught)

The post-repair relaunch died on:

```
FileNotFoundError: '/a/mm/nn-dataset/out/nn/tmp/__init__.py'
```

`check_nn` writes each candidate into a shared scratch package `out/nn/tmp/` and imports it.
Eight pods doing that against one NFS directory delete each other's module mid-import. Same
shared-writer hazard as the DB, one directory over — and it is a **race, so it cannot
reproduce on a single-GPU smoke test**. The earlier smoke test was correct and still could
not have found this.

Fix: pod-local LEMUR root. `Const.py:__project_root_path` picks the root by walking up from
CWD for a directory holding both `ab/` and `README.md`, so a private root is just a directory
with those symlinked in — code, DB and data stay shared and read-only, `out/` becomes real,
local and per-pod on an emptyDir. Confirmed in-pod: `LEMUR root /scratch/root`. Failures
after the fix are genuine architecture errors (tensor-shape mismatches, bad `conv2d` args).

**Generalises beyond this job:** any future multi-pod LEMUR job needs the private root, not
just a private DB.

### Tooling added

`norm_census.py` — counts which norms a corpus **constructs**, via AST `Call` nodes only.
The earlier "GA output is 100% BatchNorm" error came from a grep matching `nn.BatchNorm2d`
inside an `isinstance()` NaN-guard tuple, which mentions the class without constructing one;
nearly every denoise template has such a line, so the false positive was near-universal.
Counting only `ast.Call` makes that error *unrepresentable* rather than guarded against.

Validated against the known d10v2 answer before use: it reports **92 norm-free / 1 BatchNorm**
of 93 (the run had generated 4 more models since the 88/1 count). Baseline confirmed.

`analyse_screen.py` — pools `results_shard*.json`; reports run/fail split, the ep1
distribution, survivor counts at eight candidate gates, cold-vs-seeded, per-strategy medians,
and bucketed failure causes. Reports each arm twice, with and without the 125
device-repaired candidates, so the arm comparison can be quoted without my prompt bug in it.

---

## Prompt audit 2026-07-31 — driven by measurement, not intuition

Full static audit of all 516 candidates plus the partial screen. Counts are what motivated
each fix; nothing was added to the prompt speculatively, because the prompt is already
~4.5 kch and every unmeasured line dilutes the rules that matter.

| defect | n / 516 | fix |
|---|---|---|
| failed at build/first forward | ~70% (screen, n=134) | **CPU preflight** (below) |
| tensor-size mismatch | 41 of first 95 failures | constraints 5–7 (shape discipline) |
| `groups` divisibility | 14 of first 95 failures | constraint 8 |
| hardcoded 256 in the model body | 58 | constraint 9 |
| used `F.*` without importing it | 31 | REQUIRED IMPORTS block |
| indexed `out_shape` | 12 | restated in self-check |
| never clamped the output | 11 | self-check |
| referenced `self.scheduler`, never assigned | 7 | self-check |
| no class `Net` | 2 | self-check |
| torchvision/timm | 1 | already constraint 2 |

Prompt grew 4430 → 7130 ch. `--check-contract` still passes, so both arms remain identical
by construction.

**Two of my own audit checks were wrong and are worth recording**, because both would have
produced a confidently false headline:

- *"503 candidates hardcode 256"* — the regex meant to strip the mandated `__init__`
  signature used `[^)]*`, which stops at the first `)`, i.e. inside `(1, 3, 256, 256)`.
  Re-done with AST, exempting signature defaults: the real figure is **58**, and 457 have 256
  only where the contract requires it.
- *"7 still call `.to(device)` without `self.device`"* — those 7 never *reference*
  `self.device`; they bind a local `device = next(self.parameters()).device`, which is
  correct. Not a defect.

Same failure mode as the original `isinstance` grep: a pattern that matches the *mention* of
a thing rather than its *use*. Static audits of generated code need AST, not regex.

## preflight.py — the actual compute fix

The prompt fixes address maybe 12% of candidates. The 70% build/forward failure rate is not
a prompt problem to begin with, it is a **scheduling** problem: each dead candidate was
consuming a GPU slot, a dataset load and an epoch start (~80 s) to report a shape error.

`preflight.py` builds and forwards every candidate **on CPU** — no GPU requested by the job
at all — checking parse → static contract → build → `train_setup` → forward (shape equal to
input, all finite). Measured: **~50 candidates per 4 s**, so all 516 in well under a minute
against ~11.5 GPU-hours for the same verdict.

It deliberately judges nothing else: no training, no PSNR, no MAC. Everything it passes still
goes through the real 1-epoch screen. A pass means "not obviously dead", never "will train" —
a model can pass at 256×256 and still fail at another resolution.

Wired in as `screen_and_train.py --only-ids <pass_ids.txt>`.

**The current screen was deliberately NOT restarted to use it.** It is ~1/3 done; killing it
would discard ~6 GPU-hours of completed work to save about the same, and would lose the one
chance to validate preflight against ground truth — the screen's pass/fail set on the very
same 516 candidates. Preflight is being validated against it first, and applies from the next
run.

---

## What diversity threshold is defensible for LLM output? (2026-07-31, measured)

`calibrate_diversity.py` re-derives the threshold on LLM output using the same two-sided
design that produced 0.95, with anchors that are established independently of the embedding:

* **POSITIVE (must be called duplicate)** — pairs inside an exact-WL-hash collision group.
  n=24 pairs. Cosine = **1.0000 for every one of them.**
* **NEGATIVE (must be called distinct)** — pairs of different canonical hand-written
  families. n=120 pairs, median 0.777, **max 0.9213**.

### The positive anchor is degenerate, and that is the crux

All 24 positives sit at exactly 1.0 because the embedding is a *function of the graph*:
identical graphs necessarily give identical vectors. So the positive anchor proves only that
"identical implies 1.0", which was never in doubt. **It says nothing about where
near-duplicates sit**, which is precisely the question. There is no ground truth for
"different graph, same architecture in substance" in this corpus.

### A correction to what I said earlier

I claimed 0.95 "was calibrated on mutants, so it does not transfer to LLM output." That is
half wrong, and the measurement shows how. `diversity_gate.py` records the calibration as
*trivial variation ≥0.985, distinct architectures ≤0.921*. My independently measured negative
anchor is **0.9213** — the same number. The negative side was derived from distinct
architectures, not mutants, and it **does** transfer. Only the positive side (≥0.985, from
same-family mutants) is generator-specific and does not.

Consequence: 0.95 is *safe* for LLM output — it never falsely merges two architectures we
call distinct (FP=0 at every threshold ≥0.93). What is unjustified is the region between
0.921 and 0.985, where LLM near-duplicates live and where we have no ground truth at all.

### LLM corpus, 186 traceable

| thr | clusters (order-independent) | greedy median | greedy range | false merges |
|---|---|---|---|---|
| 0.921 | 54 | 85 | 79–88 | 2 |
| 0.930 | 67 | 91 | 87–92 | **0** |
| 0.950 | 91 | 111 | 105–113 | 0 |
| 0.970 | 124 | 127 | 127–129 | 0 |
| 0.985 | 135 | 138 | 137–138 | 0 |

### Recommendation

1. **Quote single-linkage cluster count, never the greedy accept count.** `DiversityGate` is
   greedy, so its answer depends on arrival order: at 0.95 it returns anywhere from 105 to
   113 over 20 shuffles of the same set. Clusters are order-independent.
2. **Use 0.93 as the floor, 0.95 as the operating point.** 0.93 is the lowest threshold with
   zero false merges against the distinct-family anchor; 0.95 keeps margin.
3. **Report the curve, not one number.** Because the positive anchor is degenerate, no single
   threshold is fully justified, and the honest artifact is the table above.
4. **Keep the exact hash, stop quoting it as diversity.** It is a useful de-dup safety net
   (it caught 10 collision groups) but it certified the cold arm at 98.6% "distinct".
5. **A behavioural anchor is what is actually missing.** Structural embedding cannot see
   "different graph, same function". Residual CKA on trained models would supply the
   near-duplicate ground truth this calibration lacks — worth doing once the survivors are
   trained, since untrained CKA is dominated by init noise.

**Headline defensible number: 91 distinct architectures of 186 traceable (516 generated),
single-linkage at 0.95.**

---

## Stage-2 epoch budget corrected: 50 -> 5 (2026-07-31)

Prompted by the question "why 1-epoch screen then full 50, when the GA only needs 5-epoch
stats?" — checked against the DB rather than defended, and the challenge was right.

**What the DB actually holds.** `stat.epoch` is `epoch_max`, a CONFIG KEY, not a per-epoch
curve index:

| kind | models | DB rows each | where the curve lives |
|---|---|---|---|
| GA generated (`DenoiseGA-*`) | 223 | **1** (final dB @ epoch_max=5) | `_epoch_stats/<hash>/{1..5}.json` on disk |
| other generated | 384 | 1 | on disk |
| hand-written seeds | 16 | 50 | 50 separate epoch_max configs |
| LLM (smoke) | 1 | 1 @ epoch_max=2 | — |

Verified directly: `DenoiseGA-01d8d358...` has exactly one row, epoch 5, 33.794 dB, while
`DenoiseUnet1` has 50 rows tracing 15.835 -> 34.885 dB.

**The docstring in `screen_and_train.py` was wrong on both halves** — it claimed survivors
would "land in the DB with per-epoch curves exactly like GA models". GA models have no
per-epoch DB curve, and at 50 epochs the LLM entries would not have matched them anyway.

**Cost**, from the GA's own `duration` field (40.7 s/epoch, cumulative 40.6/81.4/122.3/163.3/
204.3 s over 5 epochs):

| | per model | x90 survivors |
|---|---|---|
| 5 epochs | ~3.5 min | **~6 h** |
| 50 epochs | ~34 min | ~52 h |

**Decision: stage 2 defaults to `--epochs 5`.** Nine times cheaper AND directly comparable to
the 223 GA models already in the DB, which is the whole point of running both generators.

A third stage is now named explicitly rather than left implicit: a small top-K (~10-20)
re-trained to 50 epochs for any headline dB claim, because 5 epochs is a triage floor and not
a ranking (ep5 -> ep50 Spearman +0.263 on the canonical corpus).

---

## Screen COMPLETE — `denoise-llmscreen2`, 516 candidates, 8 shards

| | all | cold | seeded |
|---|---|---|---|
| candidates | 516 | 150 | 366 |
| ran | 146 (28.3%) | 43 (28.7%) | 103 (28.1%) |
| **median ep1** | 30.21 | **13.05** | **31.23** |
| ≥ 32.7 dB | **26** | 2 | 24 |
| ≥ 30.0 dB | 74 | 3 | 71 |
| max ep1 | 33.32 | 33.32 | 33.12 |

**The two arms fail at the SAME rate and succeed at completely different levels.** Both run
~28% of the time, so references do not help the model produce *valid* code. But of what runs,
the cold median is 13.05 dB (near-noise) against 31.23 dB seeded — references determine
whether the architecture is any good, not whether it compiles.

**Nothing reached 33.5 dB at one epoch**, so a "34 dB at ep1" gate would have approved zero
models, exactly as predicted from the canonical corpus.

Cold arm excluding the 122 device-repaired candidates: n=28, 6 ran, **0 clear 32.7**. Both
cold survivors are repaired models — without the repair the cold arm contributes nothing at
all, which is worth stating plainly whenever the arm comparison is quoted.

### 5-epoch stage-2 cost (single pod; DB writes cannot be sharded safely)

Measured 40.7 s/epoch + ~40 s setup.

| gate | survivors | 1 epoch | 5 epochs | extra |
|---|---|---|---|---|
| **32.7** | **26** | 0.6 h | **1.8 h** | +1.2 h |
| 32.0 | 45 | 1.0 h | 3.0 h | +2.0 h |
| 31.0 | 60 | 1.3 h | 4.1 h | +2.7 h |
| 30.0 | 74 | 1.7 h | 5.0 h | +3.3 h |
| all that ran | 146 | 3.3 h | 9.9 h | +6.6 h |

5 epochs at the 32.7 gate costs **+1.2 h** over 1 epoch and buys GA parity. Cheap; do it.
Note the screen wrote nothing to the DB (`--screen-only`), so a stage-2 pass is required
regardless — the choice is only 1 vs 5 epochs, not whether to run it.

## Labelling and git tracking — `dataset_manifest.py`

Read-only. Labels every denoise model in the DB by provenance and reports what git has not
seen. **It never runs git add/commit/push** — it writes a path list for the user to review.

| provenance | models | committed | not yet |
|---|---|---|---|
| ga-legacy (`DnSeedGA-`/`DnSkip-`/`DnNewFam-`/`DnSearch-`/`DenoiseSkip-`) | 384 | 1 | **383** |
| ga (`DenoiseGA-`) | 228 | 131 | **97** |
| seed (hand-written) | 36 | 19 | **17** |
| llm (`LLM_Denoise-`) | 1 | 0 | 1 |
| **total** | **649** | 151 | **498** |

Git delta on denoise paths: **751 new files**, 11 modified.

The cold/seeded split is **not recoverable from the DB name** — both arms write the same
`LLM_Denoise-*` prefix. The manifest joins the arm in from the generation candidate ids via
the eval results and writes it to disk, because after the fact it is otherwise
unreconstructible. (The single LLM model currently in the DB is from the smoke test and
predates that join, hence `llm-unknown`.)

### Deletion hazard — do not use `git add -A` or `git commit -a` in this worktree

`git status` reports **252,227 deleted files** under `ab/nn/stat/train/` (1,611 of them
denoise). They are tracked in HEAD and genuinely absent from disk — the tree holds 116
directories where HEAD has 252k files, all other tasks' training stats. A blanket add would
record a quarter-million deletions as a commit.

`push_list.txt` therefore contains **adds only**, every path verified to exist on disk;
deletions are counted and reported but never listed. Suggested use:

```
git -C nn-dataset add --pathspec-from-file=push_list.txt
```

---

## Stage 2 running — `denoise-llmstage2`, 26 survivors, 5 epochs

Launched 2026-07-31 16:52 UTC. Single pod, 1 GPU — the only step that writes the LEMUR DB,
and NFSv3 makes concurrent writers a corruption risk, so it stays single by design.

`--only-ids` cut 516 → 26 before touching the GPU. `--skip-screen` reuses the sharded run's
stage-1 verdict rather than re-screening survivors (~35 min of GPU saved).

### Arm is now baked into the DB name

Both arms previously wrote the same `LLM_Denoise-*` stem, which made "was this cold or
seeded?" unanswerable once the run dirs age out — the DB name is the only thing that
survives. Names are now:

```
LLM_Denoise_cold-<hash>      LLM_Denoise_seeded-<hash>
```

26 survivors = 2 cold + 24 seeded. First result:
`LLM_Denoise_cold-e4dae4216cd5c2460fc61163a684a41f`, **33.527 dB at epoch_max=5**.

Its curve peaks at epoch 4 (34.145) and settles to 33.527 — worth remembering that the DB
records the FINAL epoch, not the best, so `MAX(accuracy)` over a model's rows is not "peak
during training".

### Bug found and fixed: `curve_from` read the wrong JSON shape

`check_nn` writes each per-epoch file as a **list wrapping one dict**, not a bare dict.
`curve_from` called `d.get("accuracy")` on the list, which raised `AttributeError` straight
into a bare `except Exception: pass` — so every curve came back empty and every `last5` was
`nan`, while the DB rows were perfectly fine. It looked exactly like "no per-epoch stats were
produced" when the files were on disk and correct all along.

Fixed to unwrap the list, and the bare except now prints what it skipped instead of
swallowing it. **The DB contribution for the current run is unaffected**; only the `last5`
field in `results.json` is nan, and it can be re-derived from the on-disk JSONs without
retraining.

## Which models are already on git

| | count | contents |
|---|---|---|
| commit `befad57a9a` "132 generated models" | **132** | 131 `DenoiseGA-*` + 1 `DenoiseSkip-*` |
| all denoise models ever committed | 156 | the 132 above + 25 hand-written families |

Exact name lists written to `sweep_out/manifest/pushed_132.txt` and `pushed_all.txt`, so the
next push can diff against them instead of re-adding what is already in.

---

## Generations and per-generation yield — GA vs LLM

### The LLM has no generations, by construction

This is a structural difference, not a tuning choice. The GA is **iterative with selection**:
generation N+1 is built from generation N's survivors, so quality compounds across the run.
The LLM run is **one-shot i.i.d. sampling**: 550 independent prompts, and prompt 550 knows
nothing about prompt 1. There is no feedback, no selection, and no improvement over the run.

So "how many generations for the LLM" is **1**, and it cannot currently be more. The seeded
arm references *hand-written* seeds — never the LLM's own output — so even it is not a
feedback loop.

### GA per-generation yield (measured, pop=50)

| gen | distinct models trained |
|---|---|
| 1 | 9–10 |
| 2–6 | 15–18 (mean ~16) |

Only ~16 of a 50-member population are newly trained each generation; the rest are elites and
unchanged individuals served from cache. Projected full run: **~473 models over 30
generations**, at ~30 min/generation → **~15.4 GPU-h**.

### Throughput, measured end to end

| | GPU-h | models into DB | models/GPU-h |
|---|---|---|---|
| GA (30 gens, projected) | 15.4 | ~473 | **30.7** |
| LLM generation (2 pods × 4h20m) | 8.7 | — | — |
| LLM screen (8 pods × 48m) | 6.4 | — | — |
| LLM preflight | 0.0 (CPU) | — | — |
| LLM stage 2 (1 pod) | 1.8 | 26 | — |
| **LLM total** | **16.8** | **26** | **1.55** |

**The GA produces ~20x more DB entries per GPU-hour.** The LLM's cost is dominated by the
94% of samples that never reach the DB — 516 parseable, 146 that run, 26 that clear the gate.

Two caveats before this is read as "the LLM is not worth running":

* It measures **entries per GPU-hour, not value per entry**. The GA explores around its
  seeds; the LLM produced architectures no mutation operator can reach. Diversity is the
  reason for running it, and that is measured separately (91 distinct of 186 traceable).
* The 26 came from the **pre-fix prompt**. The measured defects now addressed cover ~12% of
  candidates statically, and preflight removes the 70% build/forward failures from the GPU
  budget entirely — the same 550 prompts would now cost ~10.4 GPU-h instead of 16.8.

### The obvious missing experiment

Give the LLM generations: feed its own survivors back as the reference block, so the seeded
arm conditions on *its own best output* rather than only on hand-written seeds. That is the
closest analogue to the GA's selection pressure, and it is the one thing that would make the
two approaches comparable rather than merely co-run. Not yet built.

---

## LLM EVOLVE — the LLM finally gets generations (2026-07-31)

Closes the structural gap: the LLM run was one-shot i.i.d. sampling with no selection, while
the GA compounds. Generation N+1 is now conditioned on generation N's own trained survivors.

### Pipeline, one pod, one GPU, sequential

| step | tool | notes |
|---|---|---|
| 1 | `elite_archive.py` | rank by last-3-epoch mean (not peak), admit greedily under the WL/cosine gate, keep K=8 |
| 2 | `llm_denoise.py --seed-set elite` | generate N=150 conditioned on that pool |
| 3 | `preflight.py` | CPU build+forward; what makes a multi-generation loop affordable |
| 4 | `screen_and_train --screen-only` | 1-epoch screen of preflight survivors only |
| 5 | `screen_and_train --skip-screen` | 5-epoch train into the DB, prefix `LLM_Eg<N>_<arm>` |

Sequential and single-GPU by necessity, not preference: steps 4–5 write the NFS-hosted
SQLite DB where concurrent writers corrupt, and generation N+1 depends on N's *trained*
results, so the stages could not be overlapped anyway.

### Two design choices that keep it honest

**Elites are diversity-gated, not just top-K by dB.** A pool picked purely on score
collapses — the top-K of a converging population are near-copies, and conditioning on K
copies of one architecture is the LLM analogue of a GA losing population diversity. Elites
are admitted greedily in dB order subject to the same WL/cosine gate the GA uses.

**Hand-written seeds stay eligible as a floor** (`--floor-seeds 4`). If the LLM's own output
is worse than the seed corpus, conditioning purely on LLM output walks the population
downhill. The smoke test confirmed this is not hypothetical: the selected pool was

```
35.246 dB  seed    DenoiseSubPixel
35.147 dB  seed    DenoiseDilatedFlat
33.903 dB  cold    0058_cold_rci
33.362 dB  seeded  0058_seeded_direct_instruction
```

The LLM's best is **1.34 dB below** the best hand-written family. `elite_archive.py` prints
an explicit warning if a pool ever becomes all-hand-written, because at that point the next
generation is conditioning, not evolving, and an "improved across generations" claim would
be false.

### Smoke test first — and it caught two real bugs

Run at GENS=1 N=4 with `PRIVATE_DB=1` (pod-local DB copy, so a smoke test cannot collide
with the production writer — a smoke test sharing the DB is not a safe smoke test).

1. **`analyse_screen.py` only globbed `results_shard*.json`.** The evolve loop cannot shard
   (it owns the DB), so its screen writes plain `results.json` — step 5 would have failed
   every single generation while the file sat right there. Now reads both.
2. **Generation ran with the wrong CWD.** `ab/gpt/util/Const.py` resolves `conf/llm/*.json`
   relative to CWD, so running it from the LEMUR root looked for
   `/scratch/root/ab/gpt/conf/...` and died. There are genuinely **two roots** here —
   generation needs `cwd=/a/mm/nn-gpt`, train/screen needs `cwd=$ROOT` — and each step now
   sets its own rather than inheriting one.

Smoke result, all five steps green: 4/4 generated → 2/4 preflight → 2/2 screened (31.57,
32.47 dB) → 2 trained as `LLM_Eg1_seeded-*`, and `last5=33.082 n_ep=2` confirms the
`curve_from` list-unwrap fix.

### Launch

`GENS=5 N=150 K=8 FLOOR=4 GATE=32.7 EPOCHS=5`, ~2.5 h/generation, ~12 h total.
**Queued behind `denoise-llmstage2`** rather than launched immediately — both write the real
DB, and overlapping them is the corruption case the whole single-writer rule exists to
prevent.

---

## One-shot LLM approach STOPPED (2026-07-31) — and a MAC gap it exposed

`denoise-llmstage2` stopped by request after 8 complete models (of 26 planned). The call was
right on the numbers: every one lands **1.3–2.2 dB below the hand-written corpus**.

| candidate | epochs | last-3 dB | MMac |
|---|---|---|---|
| 0058_cold_rci | 5 | 33.903 | 2642 |
| 0015_cold_cot | 5 | 33.893 | 5058 |
| 0096_seeded_constraint_based | 5 | 33.816 | **10787 OVER** |
| 0121_seeded_conversational | 5 | 33.399 | 1274 |
| 0058_seeded_direct_instruction | 5 | 33.362 | 4452 |
| 0066_seeded_direct_instruction | 5 | 33.259 | 1463 |
| 0150_seeded_direct_instruction | 5 | 33.256 | 5040 |
| 0026_seeded_direct_instruction | 5 | 33.055 | 6744 |
| 0170_seeded_direct_instruction | 2 (partial) | 32.713 | **16968 OVER** |

Best hand-written family: **DenoiseSubPixel 35.246 dB**. Best LLM: 33.903.

### The gap: nothing in the LLM path ever checked the MAC budget

The GA enforces 9053 MMac twice — `--seed-mac-cap` on seeds, `--edge-filter` on children.
The LLM path enforced it **nowhere**: `screen_and_train.py` contains no MAC check
(`grep -c 'mac_cap\|MAC\|edge_filter'` → 0), and `elite_archive.py` printed
`MAC CAP NOT ENFORCED: cost backend unimportable` because nn-dataset's `ab` package shadows
nn-gpt's under the pod-local LEMUR root.

**2 of 9 trained models are over cap** — one by 19%, one by 87%. Both are already in the DB.
Worse, the 10787 MMac model had been **selected into the gen-1 elite pool** and the evolve
loop was running on it: it would have propagated an illegal architecture into every
subsequent generation as a reference. The loop was stopped mid-generation to prevent that.

This is the same failure mode as the DenoiseUNet2 retraction earlier the same day: *the
prompt states the budget, but a prompt is a request, not a constraint*, and source-level
reading cannot see MACs.

### Fixed in two places, both using the GA's own counter

`edge_cost._mac_hooks`, not thop/fvcore — neither profiler is installed (the first audit
reported all 9 UNMEASURABLE), and a different counter would give numbers not comparable to
the GA's, which is the entire point of measuring against the same cap.

1. **`preflight.py`** — MAC is now a rejection stage, so an over-cap model never reaches a
   GPU at all. Free: it already builds and forwards on CPU.
2. **`elite_archive.py`** — over-cap models are ineligible as elites. **Unmeasurable is
   treated as ineligible, never as legal** — failing open is exactly how these got in.

Verified on relaunch: `[skip] 0096_seeded_constraint_based: 10787 MMac > 9053 cap`, and the
slot went to `0121_seeded_conversational` (33.399 dB, 1274 MMac).

### Open item for the dataset

The 2 over-cap models are **in the DB now**. They are not edge-legal and should not be
counted in any edge claim. Not removed — deleting DB rows is destructive and is the user's
call. `sweep_out/manifest/llm_macs.json` lists them by id.

Also still cosmetically unresolved: `elite_archive` cannot MAC-check the **hand-written**
floor seeds in-pod (`ab.gpt` shadowed). Harmless today — all 4 selected are measured legal
(4247 / 4443 / 3965 MMac and DilatedFlat) — but it is the same fail-open shape and should be
closed before the floor pool is ever changed.

---

## The 9053 MMac cap does not survive a literature check (2026-07-31)

### Where it came from

`9053` is **exactly `DenoiseUnet9F`'s own measured cost** — this ledger records it as
"OK — 0.4 MMac of headroom". The cap was set to the largest seed we happened to own, then
every later decision (seed selection, `--edge-filter`, the LLM prompt, `preflight.py`) cited
it as *the* edge budget. It is circular: it says "our models fit our cap" because the cap was
drawn around our models. No hardware measurement was ever involved.

### Normalising properly

Published mobile denoisers report at 720p or Full HD, ours at 256×256. These networks are
fully convolutional, so MACs scale linearly with pixel count and **MAC/pixel is the only
valid comparison**.

| model | reported | kMAC/px | equivalent MMac @256×256 |
|---|---|---:|---:|
| MFDNet-S | 2.34 G @720p | 2.54 | 166 |
| NAFNet | 3.81 G @720p | 4.13 | 271 |
| MAI'26 NPU student (34 ms, Dimensity 9500) | 14.13 G @FullHD | 6.76 | 443 |
| MFDNet | 11.46 G @720p | 12.43 | 815 |
| PMRID | 15.07 G @720p | 16.35 | 1072 |
| RIDNet | 20.39 G @720p | 22.12 | 1450 |
| MFDNet-L (heaviest published) | 21.81 G @720p | 23.67 | 1551 |
| **our cap (DenoiseUnet9F)** | **9053 M @256** | **138.14** | **9053** |

**Our "edge budget" is 5.8× the heaviest published mobile denoiser, 20× the model that
actually runs at 34 ms on a 2026 flagship NPU, and 54× MFDNet-S.** At 720p our cap
corresponds to **127 GMac**, which is not a mobile workload by any reading.

### Consequence

Almost nothing we have built is edge-legal by literature standards:

| model | MMac@256 | kMAC/px | verdict |
|---|---:|---:|---|
| DenoiseThunder | 1274 | 19.4 | within MFDNet-L tier |
| LLM `0121_seeded_conversational` | 1274 | 19.4 | within MFDNet-L tier |
| LLM `0066_seeded_direct_instruction` | 1463 | 22.3 | within MFDNet-L tier |
| DenoiseSelfAttn | 2112 | 32.2 | 1.4× over heaviest published |
| LiteDenoiseNet | 3530 | 53.9 | 2.3× over |
| DenoiseSubPixel (best seed, 35.25 dB) | 4247 | 64.8 | 2.7× over |
| DenoiseUnet9F | 9053 | 138.1 | 5.8× over |

Only **3 of our models** land inside the published mobile envelope, and the best-scoring seed
is 2.7× outside it. `DenoiseThunder` — literature-derived, and the one we ranked 10th of 10
on PSNR — is the most defensible edge model we own.

### Recommended caps (measured, not inherited)

| tier | kMAC/px | MMac @256 | anchor |
|---|---:|---:|---|
| aggressive | 2.5–4.1 | **170–270** | MFDNet-S / NAFNet |
| **NPU-realistic** | **6.8** | **~450** | MAI'26 student, 34 ms Dimensity 9500 |
| permissive | 23.7 | **~1550** | MFDNet-L, heaviest published |

**Not changed unilaterally** — re-capping at 450 or 1550 invalidates the current seed pool,
both running GA arms, and the LLM elite pool, and is a project-scope decision. But the 9053
figure should not be described as an edge budget in the thesis without this caveat, and
"9053 because that is what 9F measured" is not defensible under review.

Sources: MFDNet (arXiv 2211.04687), MAI 2026 NPU denoising (arXiv 2605.03680),
Mobile AI 2021 denoising challenge (arXiv 2105.08629).

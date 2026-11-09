# LLM generation and fine-tuning — measured state

Reference sheet. Every number here was computed from files on disk; the command that produces it
is given so it can be re-derived rather than trusted. Read this before re-deriving anything.

Last updated 2026-08-07.

---

## 1. Paths and invariants

| thing | value |
|---|---|
| host root | `/shared/ssd/home/b-b-ahmed/thesis` |
| same path inside a pod | `/a/mm` |
| generator | `nn-gpt/ab/gpt/brute/ga/denoise/llm_denoise.py` |
| base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| adapter flag | `llm_denoise.py --adapter <dir>` (line 777) |
| adapters | `sweep_out/run3/cycle<k>/adapter`, k = 1..20 |
| KTO warm start | `nn-gpt/out/qlora-kto/final` |
| cycle coordinator | `run3_evolve.sh` |
| SFT | `run3_sft.py` |
| preflight | `preflight.py` |
| torch on the host | **only** in `thesis/venv_tflite/bin/python` (2.8.0+cpu). The system python has no torch. |

**`meta.json` records `llm` as the BASE model id even when an adapter is applied.** Do not read the
`llm` field and conclude a run used base weights — it never says otherwise. The adapter is recorded
in the run's `gen.log` as `[llm] adapter applied from <path>` and in `meta.json` under `adapter`
(`llm_denoise.py:968`). This misled an earlier analysis in this repo; check `gen.log`.

---

## 2. Fine-tuning: what it does and does not fix

`run3_sft.py:1` — *"Run-3 per-cycle SFT: score-conditioned fine-tune on the >=bar corpus."*
It trains on models that **already passed preflight and scored above the quality bar**. It is
positives-only. `run3_evolve.sh` chains adapters cycle to cycle (merge_and_unload + fresh LoRA);
nothing is overwritten, every cycle is kept at `cycle<k>/adapter`.

### Yield by adapter — the only clean comparison

`llmprod` generated ~425 candidates per adapter from cycles 1/5/10/12 **simultaneously**, against a
frozen DB, with one prompt config. That simultaneity is what makes it clean: the duplicate gate
gets harder as the corpus grows, so sequential comparisons are confounded (see §3).

| adapter | n | pass | yield | 95% Wilson |
|---|---|---|---|---|
| cycle1 | 417 | 45 | 10.8% | [8.2, 14.1] |
| cycle5 | 430 | 52 | 12.1% | [9.3, 15.5] |
| **cycle10** | 425 | 86 | **20.2%** | [16.7, 24.3] |
| cycle12 | 427 | 66 | 15.5% | [12.3, 19.2] |

cycle10 − cycle1 = **+9.4 pp [+4.6, +14.3]** (Newcombe), resolved. cycle10 − cycle12 = +4.7 pp:
**the progression turns over after cycle 10**, so "more SFT cycles" is not automatically better.

**cycle10 is the production warm start** (`run3_evolve.sh` `PEFT0`). Caveat that must travel with
that choice: cycle10 is the best of the **four tested**, not established best of twenty.

### Where the gain came from — NOT from writing valid code

Conditional failure rate (denominator = candidates reaching that stage):

| stage | cycle1 | cycle5 | cycle10 | cycle12 |
|---|---|---|---|---|
| `forward` | 51% | 44% | 50% | 47% |
| `build` | 6% | 5% | 9% | 8% |
| `mac` | 9% | 9% | 7% | 8% |
| `arch-dup-gate` | **68%** | 71% | **41%** | 59% |

The entire gain is in `arch-dup-gate`. Forward failures are flat. Confirmed independently on
run3's own 20 cycles (741 candidates): forward-fail cycles 1–10 **63.8%** vs 11–20 **64.7%**,
difference **−0.9 pp [−8.1, +6.3], not resolved**.

> **Twelve to twenty cycles of positives-only SFT taught the generator to stop repeating itself
> and taught it nothing about channel arithmetic.** This is the expected result: no broken model
> appears in the training data, so there is no gradient for avoiding one. Fixing validity needs
> negatives, which positives-only SFT cannot supply. **We are NOT pursuing this** — channel repair
> was built, measured at 1.7% (no better than the old pass) and abandoned; see `ABLATIONS.md §7`.
> The rate is a property of the base model's codegen, and generating more candidates is cheaper
> than repairing them.

---

## 3. Never rank adapters on raw per-cycle yield

`run3/cycle<k>/pre/preflight.json` exists for all 20 cycles, but:

- **n ≈ 37 per cycle** (40 generated). Wilson intervals are 20–30 pp wide. Every cycle overlaps
  the 29.0% pooled rate; nothing is distinguishable.
- **The dup gate is corpus-dependent.** Its conditional rate climbs ~monotonically **9% (cycle1)
  → 75% (cycle17/18)** as the DB fills. Raw yield falls 26.3% → 2.8% across cycles, which is the
  gate getting harder, *not* the adapter degrading.

For a corpus-independent comparison use "cleared static/exec/build/train_setup/forward/mac", i.e.
reached the dup gates. Pooled = **215/741 = 29.0% [25.9, 32.4]**, and no cycle separates from it.
The best-vs-worst pair (cycle13 48.7% vs cycle4 21.6%, +27.1 pp) is the extreme of twenty
comparisons with no multiplicity correction — do not act on it.

**40 candidates per cycle is upstream repo behaviour, not our choice. Keep it.**

---

## 4. The preflight funnel

Stages run in **strict sequence** (`preflight.py:504-830`):

```
static → exec → build → train_setup → forward → mac → dead-code → arch-dup-db → arch-dup-gate → ok
```

**Any per-stage rate must use a conditional denominator** — candidates that reached that stage. A
candidate dying at `forward` is never tested for duplication, so marginal rates make high-mortality
arms look like low-duplication arms. This error was made once and corrected; it flipped conclusions.

llmprod totals (n=1699, 249 pass = 14.7%):

| stage | lost | share | recoverable? |
|---|---|---|---|
| `forward` | 739 | 43.5% | yes — see §6 |
| `arch-dup-gate` | 375 | 22.1% | no, this is the diversity budget |
| `build` | 117 | 6.9% | partly |
| `arch-dup-db` | 116 | 6.8% | no |
| `mac` | 67 | 3.9% | yes — put the MMac number in the prompt |
| `static` / `exec` / `train_setup` | 36 | 2.1% | partly |

Conditional pass rate once `forward` is cleared: **249/807 = 30.9%**. Use this to convert recovered
candidates into expected extra passes.

---

## 5. Prompt styles

Config `Denoise_alter_seeded_strategies.json`. Four styles, recorded in `meta.json.strategy` and
**encoded in the candidate id suffix** (`p10_0000_seeded_constraint_based`). The id parse was
validated against `meta.json` on all 1,752 candidates: **0 disagreements**, so the id is a safe key.

Styles are **balanced within every arm** (102–110 each per cycle), so style is not confounded with
adapter version.

Yield is flat — 13.3–16.0%, **none of the six pairwise comparisons resolves** at n≈425 each. What
differs is *where* candidates die (conditional rates, vs the other three styles pooled):

| stage | style | rate | others | diff | 95% CI |
|---|---|---|---|---|---|
| `forward` | Step-by-step | 59.1% | 44.1% | +14.9 pp | [+9.2, +20.5] |
| `forward` | Constraint-based | 42.0% | 49.7% | −7.8 pp | [−13.4, −2.0] |
| `arch-dup-db` | Step-by-step | 9.2% | 17.2% | −8.1 pp | [−13.0, −1.6] |
| `arch-dup-db` | Conversational | 20.9% | 13.8% | +7.1 pp | [+1.1, +13.9] |
| `mac` | Direct instruction | 11.8% | 7.0% | +4.8 pp | [+0.5, +10.0] |
| `mac` | Conversational | 4.9% | 9.5% | −4.6 pp | [−8.0, −0.2] |

Reading: step-by-step explores further (fewer exact duplicates) and breaks more (more forward
failures); constraint-based is the most reliably runnable; conversational is most derivative;
direct instruction over-builds.

**No style claim about PSNR or latency is supported.** `db_name` is null for all 249 screened
llmprod rows, so nothing converged or was profiled. The 1-epoch screen is useless as a quality
proxy here: median **32.72 dB against a 33.021 dB no-op floor**, only **19 of 238** rows above it.

`LLM_Pg<N>` in the DB is the **generation index**, not a prompt group
(`llm_evolve_par.sh:125` `for g in $(seq 1 "$GENS")`, line 237 `--prefix LLM_Pg${g}`). It cannot be
used to join prompt style to measured PSNR/latency. This looks available and is not.

Reproduce: `python ablation_prompt.py`

---

## 6. Forward failures — the census

`sweep_out/ablation/forward_census.json`, n=434 classified:

| class | n | share | repairable without topology change |
|---|---|---|---|
| conv `in_channels` vs actual input | 191 | 44% | **yes** |
| other RuntimeError | 68 | 16% | mixed |
| elementwise add/mul shape mismatch | 70 | 16% | no (needs a projection) |
| missing attribute, module never created | 38 | 9% | no (real bug) |
| concat non-channel dim mismatch | 22 | 5% | no (needs interpolation) |
| TypeError / NameError / other | 26 | 6% | no |
| view/reshape, zero-sized, downsampled too far, output-shape | 13 | 3% | partly |

The old repair (`preflight.repair_channels`) regexes the error text and needs positional integer
literals in the constructor. Measured on 120 real failures (`chan_repair_eval.json`):
**v1 fired 2 / fixed 0; v2 fired 7 / fixed 2 → 2 fixes in 120 = 1.7%.** Decline reasons: 42 "could
not build model", 38 "conv already has matching channels" (misdiagnosis), ~12 "could not isolate
the conv inside `self.<attr>`".

**The 44% figure above is from a DIFFERENT POPULATION** (the ablation's generated variants) and
does NOT hold for llmprod. Measured on llmprod's own failures: elementwise mismatch **46.7%**,
concat **12.5%**, conv channel only **9.2%**. See `ABLATIONS.md §7`.

`repair_shapes.py` was built to target the conv class and **measured at 1.7%, the same as the pass
it replaced. Channel repair is ABANDONED — do not revisit.** It is not wired into `preflight.py`
and must not be; it is kept as a validated negative result and a diagnostic:

```
venv_tflite/bin/python repair_shapes.py --list <paths.txt> --out out.json
```

Forward failure is flat across 20 SFT cycles and across all four prompt styles, i.e. a property of
the base model's codegen. Generation is the cheap step — generate more rather than repair.

---

## 7. Accuracy prediction

Use `predict_final.py`. It enforces the split; do not call a predictor directly.

| predictor | Spearman ρ | MAE | top-50 overlap | use |
|---|---|---|---|---|
| in-house (n=326) | **−0.178** | — | — | **never** |
| stock ABrain `acc_client.py` | +0.226 | 1.616 | 57% | no |
| fine-tuned on denoising | +0.629 | **0.509 dB** | 74% | **calibration** |
| epoch-3 measured PSNR | **+0.728** | 2.559 | **82%** | **selection** |

`rank_for_selection()` uses epoch-3 only and calls no model. `estimate_final_db()` uses the
fine-tune. `acc_client.py` hardcodes 50 epochs, which is wrong for this DB — 701 of 727 curves are
3–5 epochs (+0.30 dB ep3→best) vs 26 long ones (+2.01 dB).

The predictor flags are **deliberately neutralised in all 14 GA launchers**. See `GA.md §4`.

---

## 8. Running generation jobs: the local-root wrapper, and why it exists

`refab_gen.sh` runs generation from a pod-local LEMUR root. **There is no upstream pattern for this**
— checked against `main` (2026-08-08): every `ab/gpt/act/` runner uses `ab_root_path` directly, with
no `/scratch` root, no sqlite `backup()`, and no `chdir`. The only sanctioned env overrides,
`AB_GPT_NNGPT_DIR` / `NNGPT_DIR_OVERRIDE`, redirect **output** only. No DB-path override exists.

Upstream doesn't need one: single jobs absorb the shared-DB read in minutes. Ours runs beside 13
training pods on one filesystem.

**The cost, measured.** `cd /a/mm/nn-gpt` makes `__project_root_path()` resolve the root to nn-gpt,
so `Const.db_file` becomes `/a/mm/nn-gpt/db/ab.nn.db` — **1,464 MB, 19,473 models, 22 of them
denoise**. It is the upstream LEMUR DB (780k img-classification stat rows); classification runs need
it, we never touch it. Our references come from nn-dataset's **43 MB** DB via `db_path()`.

| condition | startup to first candidate |
|---|---|
| llmprod, quiet cluster (Aug 5) | **~9 min** (`run_meta` 06:35:41, first candidate 06:36:19) |
| refab, farm submitter active | 2.5 h, **zero candidates** — killed |
| refab, submitter done | 0.7 MB/s → ~35 min for the DB read alone |

**Do not time these runs by directory mtime.** Preflight later writes `candidate.py.orig` into
candidate dirs, pushing mtimes hours past generation. That artifact made a 9-minute startup look
like 2 h and nearly caused a healthy run to be killed. Use `meta.json`'s `ts` field. Steady-state
generation is **61.9 s/candidate**, median gap 60 s, max 142 s.

**Local root contents** — each item earned its place by breaking something:

| item | why |
|---|---|
| `README.md` + `version` + `pyproject.toml` | `__project_root_path()` needs `ab/`+`README.md`; then `get_version()` (`Const.py:22`) demands `version` AND `pyproject.toml` or raises at import |
| `ab/nn` minus `nn/`, `stat/` | 328 MB of model sources and 13.8k stat JSONs, needed by neither — `nn_dir()` resolves absolutely to `/a/mm`. Excluding them: 392 files, 1.7 MB |
| `ab/gpt` minus `brute/` | `ab/gpt/util/Const.py:1` takes `ab_root_path` from nn-dataset's Const, so both share ONE root; a root without `ab/gpt` dies on `<root>/ab/gpt/conf/llm/*.json` |

Final root: **71 MB**, and both `Const` modules resolve correctly (validated on the host).

**rsync excludes anchor at the transfer root, which is the PARENT of the source arg.** Copying
`$SRC/ab/nn` makes paths read `nn/...`, so `ab/nn/nn` is `'/nn/nn'` — `'/nn'` matches the top-level
directory and excludes *everything*. Likewise `'/gpt/brute'`, not `'/brute'`. And an unanchored
`'db'` matches any component named `db`, silently deleting `ab/nn/util/db/` (a package) and
producing `ModuleNotFoundError: No module named 'ab.nn.util.db'`.

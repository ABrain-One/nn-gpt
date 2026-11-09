# Naming convention for everything we contribute to LEMUR

Adopted 2026-08-11, after the batch-1 review. Applies to all new work.

## The rule

**No underscores in any name that reaches the LEMUR repo. Use hyphens.**

That means all four of these:

| field   | where it comes from            | good                          | bad                     |
|---------|--------------------------------|-------------------------------|-------------------------|
| task    | passed to `check_nn`           | `img-denoising`               | `img_denoising`         |
| dataset | `ab/nn/loader/<dataset>.py`    | `denoise-awgn`, `awgn-color`  | `denoise_awgn`          |
| metric  | `ab/nn/metric/<metric>.py`     | `psnr`, `psnrrgb`, `ssimrgb`  | `psnr_rgb`, `mai_psnr`  |
| model   | `ab/nn/nn/<model>.py`, DB `nn` | `LLM-Denoise-cold-<hash>`     | `LLM_Denoise_cold-<h>`  |

## Why

A training record lives in a directory named by joining those four fields with `_`:

```
ab/nn/stat/train/<task>_<dataset>_<metric>_<model>/
```

and `ab/nn/util/Util.py:61` reads it back with a positional split:

```python
task, dataset, metric, nn = tuple(c.split(config_splitter))   # config_splitter = '_'
```

One underscore inside any single field shifts every later field by one, and the
import dies with `too many values to unpack (expected 4)`. **The failure is not
local** — it aborts the whole import, so a single malformed name withholds an
entire 500-model batch.

This is not a style preference. Upstream had been demonstrating the convention all
along: every multi-word name it owns is hyphenated (`cifar-10`, `celeba-gender`,
`imagenette`, `psnrrgb` — note it spells that one without a separator rather than
`psnr_rgb`). Our generators drifted from it.

### What it cost us

- **372 models** minted as `LLM_Pg23_seeded-<hash>` had to be renamed on disk after
  review, along with their stat dirs — 744 paths.
- **111 stat dirs** under dataset `denoise_awgn` had to be withdrawn from a commit
  that was already pushed.
- The rename then desynchronised disk from the DB (470 rows still carry the old
  names), which needed an aliasing layer to keep uploads working.

## How it is enforced

Three layers, so this cannot regress silently. All of them live in `naming.py`.

**1. Mint time — names are born clean.** The generator prefixes now use hyphens,
and `naming.check()` raises if one somehow does not:

| file                  | was                       | now                       |
|-----------------------|---------------------------|---------------------------|
| `screen_and_train.py` | `--prefix LLM_Denoise`, `+"_"+arm` | `--prefix LLM-Denoise`, `+"-"+arm` |
| `llm_evolve.sh`       | `--prefix LLM_Eg${g}`     | `--prefix LLM-Eg${g}`     |
| `llm_evolve_par.sh`   | `--prefix LLM_Pg${g}`     | `--prefix LLM-Pg${g}`     |
| `llm_stage2.sh`       | `--prefix LLM_Denoise`    | `--prefix LLM-Denoise`    |
| `run3_db_import.py`   | `f"LLM_R3c{cycle}-..."`   | `f"LLM-R3c{cycle}-..."`   |

**2. Export time — `export_db_stats.py`** builds every stat dir through
`naming.stat_dir()` and asserts `naming.check_stat_dir()`. Model names from legacy
DB rows are canonicalised `_`→`-` and every rewrite is printed; task/dataset/metric
are **hard errors**, because each names a real module file that must be renamed at
source (renaming it only in the path would produce a stat dir whose loader cannot
be found on re-import).

**3. Upload time — `upload_batches.py --plan`** refuses to write a batch list at all
if any path would fail the 4-way split. A plan that fails here is a generator bug,
not something to work around.

Plus a standalone pre-push audit:

```bash
python3 naming.py --repo /shared/ssd/home/b-b-ahmed/thesis/nn-dataset
# exit 0 = safe to push; violations in upstream-owned files are reported but do not fail
```

## Reading legacy names

Minting and export are strict. **Reading history stays permissive** — 470 DB rows
predate the rename and rejecting them would hide the corpus rather than fix it. Use:

- `naming.matches_family(name, "LLM-")` — matches `LLM-` *and* `LLM_`
- `naming.arm_of(id_or_name)` — `cold`/`seeded` from either separator
- `naming.alias_both_ways(dict)` — adds canonical aliases to a name-keyed dict
- SQL: `nn GLOB 'LLM[-_]*'`, **not** `nn LIKE 'LLM_%'` — in `LIKE`, `_` is a
  single-character wildcard, so the old pattern was matching by accident. Any query
  typed with the new prefix (`LIKE 'LLM-Denoise%'`) would otherwise return zero rows
  against a DB that still holds the old spelling, and the export would look
  already-done.

### Candidate ids are a different namespace — leave them alone

Generation-run ids like `p10_0000_seeded_constraint_based` keep their underscores.
They are run-local scratch, never become a model name, and never reach a path.
Canonicalising them would break the arm join for every historical run.

## Loaders renamed under this convention

`ab/nn/loader/`: `denoise_awgn.py`→`denoise-awgn.py`, `awgn_color.py`→`awgn-color.py`,
`sidd_bench.py`→`sidd-bench.py`. All three were untracked, so no git history cost.

A hyphenated module name is fine for LEMUR: it loads loaders through
`__import__(mod_string, fromlist=[...])` (`ab/nn/util/db/Util.py:25`), which takes the
name as a *string* and so bypasses Python's identifier syntax — exactly how upstream
loads `cifar-10.py`. Only the `import a.b.c-d` *statement* form is a `SyntaxError`;
`sidd_bench_eval.py` was switched to the `__import__` form for this reason.

Two upstream files still violate the rule — `ab/nn/loader/imdb_wiki.py` and
`ab/nn/metric/mai_psnr.py`. Both are real hazards for whoever uses them (an
`imdb_wiki` dataset cannot produce a parseable stat dir), but they are not ours to
rename. The audit reports them separately and does not fail on them.

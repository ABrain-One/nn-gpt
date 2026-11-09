# Handoff: what the professor needs to run the denoise GA

Audited 2026-08-11. **The GA is currently NOT runnable from the repo.** `run_seed_ga.py` is tracked,
but 10 of the 11 modules it imports are not, so a fresh clone fails on `import ga_core` before doing
any work. The seed architectures the launchers reference by path are also absent.

`docs/handoff_ga_files.txt` is the fix: **31 paths**, all verified present on disk and none currently
tracked. Adding them makes the GA runnable end to end.

SCOPED TO THE CURRENT GA ONLY. The manifest carries what `launch_run5.sh` (the production config
that produced the results we report) and `launch_random_search.sh` (its matched-budget baseline)
actually reference -- both use the identical seed set. Ten seed files present on disk are NOT
shipped because no current launcher references them:
`new_seeds/{DenoiseGroupNorm,DenoiseLayerNorm,DenoiseMFDNet,DenoiseResBN,DenoiseSplitterNet,
DenoiseSplitterNetF,DenoiseSplitterNetL,DenoiseThunder}.py` and
`promoted_pool_r3/{8901067b4e1ec298,e11d622b58e7cbe7}.py`. Shipping unused seeds would imply they
were part of the search space that produced our numbers, and they were not.

---

## 1. Current state

| | on disk | tracked |
|---|---|---|
| files in `ab/gpt/brute/ga/denoise/` | 212 | **14** |
| dependency modules of `run_seed_ga.py` | 11 | **1** (the entry point itself) |
| `new_seeds/*.py` | 22 | **0** |
| `promoted_pool_r3/*.py` | 6 | **0** |
| `README.md` (193 lines, documents the pipeline) | yes | **no** |

Branch: `brute-skip-operator-and-cli`. The 14 tracked files are `run_seed_ga.py`,
`run_denoise_ga.py`, `populate_db.sh`, `seeds_distinct.sh`, and the `launch_{A0,B0,C0,D0,E0,
distinct10,distinct_full,production,production_v2,wlval}.sh` launchers.

---

## 2. Already upstream — nothing to do

- **`ab/gpt/brute/ast/mutator/`** — 39 files, tracked. `seed_mutation.py` imports
  `mutator.config`, `mutator.planning.ModelPlanner` and `mutator.execution.code_mutator` from it.
- **`ab/gpt/util/{LLM,LLMUtil,KTO,Const,acc_client}.py`** — all present after the 2026-08
  refactoring. Verified against the current `main`.
- **`ab.gpt.act.kto.kto_data_manager`** — the refactoring moved this out of `ab.gpt.kto_pipeline`.
  `build_kto_dataset.py` now tries the new path first and falls back, so it works either side of the
  refactor. This was the only import in the whole denoise package the refactoring touched.

---

## 3. The dependency closure of the GA

Computed by walking imports from `run_seed_ga.py`, then verified by importing each module:
**9/10 import cleanly**, and the tenth (`seed_mutation`) fails only on the missing `psutil`
package — a dependency to install, not a code gap.

```
run_seed_ga.py            entry point                    (tracked)
ga_core.py                population, tournament selection, elitism, mutation loop
seed_mutation.py          AST mutation via the mutator package (needs psutil)
skip_operators.py         GLOBAL / LATERAL / BLOCKRES skip operators
block_crossover.py        block-level recombination
arch_uid2.py              WL architecture certificate = the distinctness key
arch_embed.py             graded architecture embedding (cosine similarity)
diversity_gate.py         novelty gate, exact + graded
edge_cost.py              MACs / params measurement
functional_similarity.py  behavioural similarity check
```

`arch_uid2.py` matters beyond the GA: it is the definition of distinctness used for the model
uploads, and the DB's `arch_uid` column is exactly this function's output (verified 25/25).

---

## 4. Commit command

```bash
cd /shared/ssd/home/b-b-ahmed/thesis/nn-gpt
git add --pathspec-from-file=ab/gpt/brute/ga/denoise/docs/handoff_ga_files.txt
git diff --cached --name-status | awk '$1!="A"'      # expect no output
git commit -m "denoise GA: add dependency modules, seeds and README so the pipeline runs from a clone"
```

31 paths: 10 dependency modules, `README.md`, `launch_run5.sh` (current production config),
`launch_random_search.sh` (the matched-budget baseline), the **14** `new_seeds/*.py` and **4**
`promoted_pool_r3/*.py` those launchers reference.

Check `git diff --cached --name-status` in the **nn-gpt** repo the same way as for nn-dataset —
this tree may also have tracked files missing from disk, and `--pathspec-from-file` with a
directory pathspec would stage those as deletions.

---

## 5. How to run it, once committed

Two sibling repos under one root, as `README.md` §1 describes:

```
<root>/nn-dataset      datasets, training, metrics, the DenoiseUnet* seeds
<root>/nn-gpt          this GA layer + the brute mutation engine
```

```bash
export PYTHONPATH=<root>/nn-dataset:<root>/nn-gpt:<root>/nn-gpt/ab/gpt/brute/ast
pip install psutil
cd <root>/nn-dataset          # CWD must be under nn-dataset: seed_mutation.py requires it
bash <root>/nn-gpt/ab/gpt/brute/ga/denoise/launch_run5.sh
```

`brute/ast` must be on `PYTHONPATH` because `seed_mutation.py` imports `mutator` as a top-level
package, not as `ab.gpt.brute.ast.mutator`.

Production configuration, from `launch_run5.sh`:

```
--pop-size 50 --gens 30 --epoch 5 --elitism 5 --tournament 3
--crossover block --skip-ops --skip-prob 0.3
--diversity-gate --diversity-graded --graded-max 0.95
--edge-filter --mac-cap 9053
--cost-metric macs --cost-mode log --cost-weight 0.3
BRUTE_WEIGHTS="dimension:0.10,activation:0.15,layer_type:0.15,kernel_size:0.10,stride:0.10,scale:0.15,skip:0.15"
```

= 1,500 candidate evaluations at 5 epochs. `launch_random_search.sh` runs the same space, same
gates and same budget with the guidance removed (`--gens 1 --elitism 0 --tournament 1`), which is
the baseline a NAS paper is rejected for omitting.

**The predictor flags are deliberately absent.** All 14 launchers carry a header explaining why:
the in-house zero-cost surrogate ranks **anti-correlated** with truth (Spearman −0.178 over 326
models, against the +0.693 its own cross-validation claimed), so `--predict-keep 0.5` was
discarding the better half of every generation. Use epoch-3 measured PSNR for selection instead
(+0.728, 82% top-half overlap) — see `predict_final.py`. Do not re-enable them.

---

## 6. Superseded launchers ALREADY in the repo

Ten of the 14 currently-tracked files are launchers from earlier runs that nothing uses now:
`launch_{A0,B0,C0,D0,E0,distinct10,distinct_full,production,production_v2,wlval}.sh`, plus
`seeds_distinct.sh` and the older driver `run_denoise_ga.py`. They were pushed before
`launch_run5.sh` existed.

They are NOT removed here -- deleting tracked files is a separate decision, and they do document
the run history. But anyone picking this up should start from `launch_run5.sh`, and the earlier
launchers reference seed sets and flag combinations that do not match the results we report. If you
want them gone, that is a deliberate `git rm` of those 12 paths, not something this manifest does.

## 7. Not part of this handoff

The 212 files on disk include analysis and one-off tooling that nobody else needs to run the GA:
ablation drivers, re-export and profiling pipelines, upload tooling, log files, and per-run
model/checkpoint directories. Shipping all of it would obscure the runnable core. If more is
wanted later, the natural next tranches are the ablation scripts
(`ablation_factorial.py`, `ablation_placement.py`, `ablation_prompt.py`) and the on-device
pipeline (`npu_gate.py`, `bulk_profile.py`), each with its own dependency check first.

Excluded deliberately, as recorded in `GIT_UPLOAD.md`: the `*.pre-*` backups, `sweep_out/`
artifacts, and the 460 MB `data_sidd_bench/*.mat`.

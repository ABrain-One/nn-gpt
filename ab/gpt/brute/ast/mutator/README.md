# Brute-force architecture-mutation search

An algorithmic ("brute-force", non-LLM) neural-architecture search. It takes seed
models from the **LEMUR** database (the `nn-dataset` repo), applies single-point
**AST mutations** to their source, validates each candidate (forward/backward +
output-shape check), and writes the valid ones out as new `.py` model files. A
separate driver then trains/ranks the candidates.

It is a **search space of operators**: each operator is one kind of architectural
edit (change a conv width, swap an activation, add a skip connection, …). A run
repeatedly picks an operator at random (weighted), applies it to a seed, and keeps
what survives validation.

```
seed model (LEMUR)  ->  ModelPlanner picks an operator  ->  plan (what to edit + where)
     ->  in-memory validation (deepcopy + forward/backward)
     ->  CodeMutator rewrites the source AST
     ->  emitted-code validation (re-instantiate the written source, shape-check)
     ->  save candidate .py  (+ optional train/save to LEMUR DB)
```

## Requirements

- **Two repos, side by side:** this repo (`nn-gpt`) and `nn-dataset`. The engine
  imports `ab.nn.api` from `nn-dataset` and reads seeds from its LEMUR DB.
- Seeds must **exist in the LEMUR DB** (with source in the `nn` table).
- A Python env with `nn-dataset` installed / importable (`torch`, etc.).

## Quickstart (CLI)

Run from this directory (`ab/gpt/brute/ast/mutator`). `cli.py` needs no edits to
`config.py`:

```bash
PY=/path/to/nn-dataset/.venv/bin/python

# list the available operators and their current weights
$PY cli.py --list-operators

# isolate ONE operator on ONE seed (e.g. the global-residual skip experiment)
$PY cli.py --lemur-root /path/to/nn-dataset \
    --models DenoiseUnet2F --only skip --attempts 2 --debug

# a mixed search over several seeds
$PY cli.py --lemur-root /path/to/nn-dataset \
    --models DenoiseUnet7F,DenoiseUnet9F --weights skip:1.0,activation:0.5 --attempts 50
```

CLI flags (all optional; unset flags fall back to `config.py` defaults):

| flag | meaning |
|------|---------|
| `--lemur-root` | path to the `nn-dataset` repo (LEMUR root). Also `$NN_DATASET_ROOT`. |
| `--models` | comma-separated seed names (LEMUR `nn` names) |
| `--attempts` | mutation attempts per model |
| `--only` | isolate a single operator (weight 1.0, others 0.0) |
| `--weights` | per-operator weights, e.g. `skip:1.0,activation:0.5` |
| `--output` | output root for generated `.py` files |
| `--workers` | max parallel worker processes |
| `--debug` | single-process + verbose (**use this for first runs**) |

Under the hood the CLI sets `BRUTE_*` environment variables that `config.py` reads.
Because env vars are inherited by child processes, overrides **also work in parallel
mode** (plain `config` attribute assignment does not — workers re-import `config`).

You can also run the engine directly with the config-file defaults:
`python -m mutator.run_mutation` (from `ab/gpt/brute/ast`, with that dir on `PYTHONPATH`).

## Output & evaluation

Generated candidates are written to `config.MUTATED_MODELS_OUTPUT_ROOT`
(default `mutator/nn-dataset/mutated_models/<seed>/<seed>-ast-<op>-<checksum>.py`);
plan/skip records go to `mutation_plans/`. Both are generated artifacts (git-ignored).

Generation is **decoupled from training**. To evaluate candidates, use
`../denoise_search.py` (proxy-screen at few epochs, rank, then full-train the top-K).

## Operators

| operator | edits | kind |
|----------|-------|------|
| `dimension` | conv/linear/norm channel widths | constructor kwarg |
| `activation` | activation module type | constructor |
| `layer_type` | norm/pool type | constructor |
| `kernel_size` | Conv2d kernel | constructor |
| `stride` | Conv2d stride | constructor |
| `scale` | sub-network width/depth kwargs | constructor |
| `architectural` | ConvNeXT block/stage layout | constructor (ConvNeXT-only) |
| `skip` | global input→output residual in `forward()` | **`forward()` dataflow** |

All operators except `skip` edit **constructor arguments** in `__init__`. `skip` is
the first operator that rewrites **`forward()` dataflow** (a `Return` statement).

## Adding an operator

The pattern is consistent. An operator = a **planner** (decides *what* + *where* to
edit) + an **apply function** (does the AST edit) + **registration** in three files.
Use the `skip` operator as the worked example.

1. **Apply function** — `execution/apply_<name>.py`
   A function `apply_<name>_modification(node, mod)` that mutates an AST node in place.
   (`skip` edits an `ast.Return`; constructor operators edit an `ast.Call`.)

2. **Planner** — `planning/<name>_planner.py`
   A class `Xplanner(model_planner)` with:
   - `plan_<name>_mutation() -> dict` returning `{ key: { "mutation_type": "<name>",
     "source_location": {"lineno":.., "col_offset":..}, ...params } }`
   - `can_apply() -> bool` (used for applicability).
   `source_location` must be the (lineno, col_offset) of the AST node to edit, from
   parsing the **same** `source_code` the engine will parse.

3. **Register in `planning/base_planner.py`** (3 spots):
   - import + instantiate the planner in `__init__`
   - dispatch it in `plan_random_mutation()` (`elif selected_type == '<name>': ...`)
   - applicability in `_is_mutation_type_applicable()` (`elif mutation_type == '<name>': ...`)

4. **Register in `execution/code_mutator.py`**:
   - import `apply_<name>_modification`
   - add `schedule_<name>_modification(...)` (appends a `mod` dict with `'type': '<name>'`)
   - dispatch it in the right visitor. Constructor edits go in `visit_Call`; a new
     node kind needs its own visitor (see `visit_Return` for `skip`).

5. **Register dispatch in `orchestrator.py`**:
   in the `for full_module_name, details in plan.items()` loop, add
   `elif mutation_type == "<name>": code_mutator.schedule_<name>_modification(...)`.
   Note: `get_submodule` is only called for constructor operators — a `forward()`
   operator's plan key is not a real module name (see the `skip` guard).

6. **Register in `config.py`**:
   add the operator to `MUTATION_TYPE_WEIGHTS` (start at `0.0`) and a line in
   `OPERATOR_DESCRIPTIONS`.

Then smoke-test with `python cli.py --models <seed> --only <name> --attempts 1 --debug`.

> **Roadmap note (planned refactor):** the 3-file registration above is boilerplate.
> The intended next step is an operator **registry** (each operator self-registers its
> planner + apply + schedule via a decorator) so adding one is a single file. Until
> then, follow the 6 steps above.

## Known gotchas

- **cwd = nn-dataset** before the orchestrator imports `ab.nn` (the CLI handles this
  via `--lemur-root`). Otherwise the LEMUR root is resolved from the wrong directory.
- **Runtime overrides must go through env vars** (`BRUTE_*` / the CLI), not by setting
  `config.X` in a script, if you use parallel mode — workers re-import `config`.
- **`--debug` runs single-process**; it's the reliable path for first runs and for
  reading rejection reasons.
